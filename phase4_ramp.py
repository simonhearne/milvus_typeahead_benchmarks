"""Phase 4: open-loop QPS ramp for one query form until failure or a ceiling.

Rate starts at --start and rises by --step every --step-seconds until --max or the first failing
step. Load is spread over --workers processes, each with its own AsyncMilvusClient, sending
uniform interleaved arrivals, so the aggregate is a steady target rate. Prefixes are drawn
from the phase-2 workload (popularity-weighted, all lengths mixed).

A step FAILS if: error rate > --max-error-pct, or completed/target < --min-achieved, or
p95 > --max-p95-ms. A step is flagged CLIENT-LIMITED (not a server result) if the p99
scheduling lag exceeds --max-lag-ms or client CPU is saturated.

Usage: python phase4_ramp.py [--state results/<ts>/phase2_state.json] [--form ngram.like_prefix]
                             [--start 100 --step 100 --max 2000 --step-seconds 60 --workers 8] [--keep]
"""
import argparse
import asyncio
import csv
import gzip
import json
import multiprocessing as mp
import os
import random
import time
from pathlib import Path

import config
from phase3_bench import latest_state, pct
from variants import BY_KEY, call_kwargs

CONSISTENCY = "Bounded"
REQUEST_TIMEOUT_S = 5.0


def worker(wid, args, schedule, t0, prefixes, coll, stop, out_q):
    form = _form(args.form)
    rng = random.Random(1000 + wid)

    async def run():
        from pymilvus import AsyncMilvusClient
        ac = AsyncMilvusClient(uri=config.ZILLIZ_URI, token=config.ZILLIZ_TOKEN, db_name=config.ZILLIZ_DB)
        await ac.query(coll, filter="id >= 0", limit=1)  # connect before t0
        loop = asyncio.get_running_loop()
        # Map wall-clock t0 onto the loop clock.
        base = loop.time() + (t0 - time.time())
        finalisers = []
        for k, rate in enumerate(schedule):
            if stop.is_set():
                break
            start = base + k * args.step_seconds
            n = int(rate * args.step_seconds / args.workers)
            lat, errs, lags = [], [], []
            tasks = []
            for i in range(n):
                planned = start + (i * args.workers + wid) / rate
                d = planned - loop.time()
                if d > 0:
                    await asyncio.sleep(d)
                lags.append((loop.time() - planned) * 1000)
                tasks.append(asyncio.create_task(one(ac, rng.choice(prefixes), lat, errs)))
            finalisers.append(asyncio.create_task(report(k, rate, tasks, lat, errs, lags)))
        await asyncio.gather(*finalisers)
        await ac.close()

    async def one(ac, p, lat, errs):
        method, kw = call_kwargs(form, p, CONSISTENCY)
        t = time.perf_counter()
        try:
            await getattr(ac, method)(coll, timeout=REQUEST_TIMEOUT_S, **kw)
            lat.append((time.perf_counter() - t) * 1000)
        except Exception as e:  # noqa: BLE001 - recorded, no retry
            errs.append(f"{type(e).__name__}[{getattr(e, 'code', None)}]: {str(e)[:200]}")

    async def report(k, rate, tasks, lat, errs, lags):
        await asyncio.gather(*tasks)
        out_q.put({"step": k, "rate": rate, "wid": wid, "lat": lat, "errs": errs, "lags": lags})

    asyncio.run(run())


def _form(key):
    vkey, fname = key.split(".")
    return next(f for f in BY_KEY[vkey].forms if f.name == fname)


def cpu_sample():
    try:
        import psutil
        return psutil.cpu_percent(interval=None)
    except ImportError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", type=Path, default=None)
    ap.add_argument("--form", default="ngram.like_prefix")
    ap.add_argument("--start", type=int, default=100)
    ap.add_argument("--step", type=int, default=100)
    ap.add_argument("--max", type=int, default=2000)
    ap.add_argument("--step-seconds", type=float, default=60)
    ap.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 2) - 1))
    ap.add_argument("--max-error-pct", type=float, default=1.0)
    ap.add_argument("--min-achieved", type=float, default=0.95)
    ap.add_argument("--max-p95-ms", type=float, default=50.0)
    ap.add_argument("--max-lag-ms", type=float, default=10.0)
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    state_path = args.state or latest_state()
    state = json.loads(Path(state_path).read_text())
    p2 = Path(state_path).parent
    workload = json.loads((p2 / "workload.json").read_text())
    prefixes = [p for L in sorted(workload["string"]) for p in workload["string"][L]]
    coll = state["collections"]["main"]["name"]
    schedule = list(range(args.start, args.max + 1, args.step))

    ts = config.stamp()
    out = config.run_dir(ts)
    sync = config.client()
    meta = {"phase": 4, "run": ts, "phase2_state": str(state_path), "form": args.form, "schedule": schedule,
            "step_seconds": args.step_seconds, "workers": args.workers, "arrivals": "uniform, interleaved across workers",
            "consistency": CONSISTENCY, "request_timeout_s": REQUEST_TIMEOUT_S, "rows": state.get("rows_loaded"),
            "failure_rule": {"max_error_pct": args.max_error_pct, "min_achieved": args.min_achieved,
                             "max_p95_ms": args.max_p95_ms},
            "client_limited_rule": {"max_lag_ms": args.max_lag_ms, "cpu_pct": 90},
            "client_cpu_count": os.cpu_count(), **config.run_meta(sync)}
    print(f"server={meta['server_version']} build={meta['cluster_db_version']} client={meta['client_hostname']} "
          f"cpus={os.cpu_count()} connect={meta['tcp_connect_ms_median']}ms")
    print(f"{args.form} on {coll} ({meta['rows']} rows); {len(schedule)} steps x {args.step_seconds:.0f}s, "
          f"{args.workers} workers")

    ctx = mp.get_context("spawn")  # gRPC channels do not survive fork
    stop, out_q = ctx.Event(), ctx.Queue()
    t0 = time.time() + 8  # time for workers to start and connect
    procs = [ctx.Process(target=worker, args=(w, args, schedule, t0, prefixes, coll, stop, out_q), daemon=True)
             for w in range(args.workers)]
    for p in procs:
        p.start()
    cpu_sample()

    steps, raw = [], []
    pending = {}
    expected_steps = len(schedule)
    k_done = 0
    while k_done < expected_steps:
        try:
            msg = out_q.get(timeout=args.step_seconds + REQUEST_TIMEOUT_S + 30)
        except Exception:  # noqa: BLE001
            print("timed out waiting for workers")
            break
        pending.setdefault(msg["step"], []).append(msg)
        if len(pending.get(k_done, [])) < args.workers:
            continue
        parts = pending.pop(k_done)
        lat = [x for m in parts for x in m["lat"]]
        errs = [x for m in parts for x in m["errs"]]
        lags = [x for m in parts for x in m["lags"]]
        rate = parts[0]["rate"]
        issued = len(lat) + len(errs)
        cpu = cpu_sample()
        s = {"step": k_done, "target_qps": rate, "issued": issued,
             "achieved_qps": round(len(lat) / args.step_seconds, 1),
             "achieved_ratio": round(len(lat) / (rate * args.step_seconds), 4),
             "error_pct": round(100 * len(errs) / issued, 3) if issued else None,
             **{f"p{q}": round(pct(lat, q), 2) if lat else None for q in (50, 90, 95, 99, 99.9)},
             "max": round(max(lat), 2) if lat else None,
             "sched_lag_p99_ms": round(pct(lags, 99), 2) if lags else None,
             "client_cpu_pct": cpu, "error_sample": errs[0] if errs else ""}
        fail = []
        if s["error_pct"] is not None and s["error_pct"] > args.max_error_pct:
            fail.append(f"errors {s['error_pct']}%")
        if s["achieved_ratio"] < args.min_achieved:
            fail.append(f"achieved {s['achieved_ratio']:.0%} of target")
        if s["p95"] is not None and s["p95"] > args.max_p95_ms:
            fail.append(f"p95 {s['p95']}ms > {args.max_p95_ms}ms")
        client = []
        if s["sched_lag_p99_ms"] and s["sched_lag_p99_ms"] > args.max_lag_ms:
            client.append(f"sched lag p99 {s['sched_lag_p99_ms']}ms")
        if cpu is not None and cpu > 90:
            client.append(f"client CPU {cpu}%")
        s["fail"] = "; ".join(fail)
        s["client_limited"] = "; ".join(client)
        steps.append(s)
        raw += [(k_done, rate, round(x, 3), "") for x in lat] + [(k_done, rate, "", e) for e in errs]
        print(f"  step {k_done:2d} target={rate:5d} achieved={s['achieved_qps']:7.1f} p50={s['p50']} p95={s['p95']} "
              f"p99={s['p99']} p99.9={s['p99.9']} err={s['error_pct']}% lag99={s['sched_lag_p99_ms']} cpu={cpu}"
              + (f"  FAIL: {s['fail']}" if fail else "") + (f"  CLIENT-LIMITED: {s['client_limited']}" if client else ""))
        k_done += 1
        if fail or client:
            stop.set()
            meta["stopped_at_step"] = s["step"]
            meta["stop_reason"] = s["fail"] or s["client_limited"]
            # Workers finish the step they are already in; collect it for completeness, then stop.
            expected_steps = min(expected_steps, k_done + 1)

    for p in procs:
        p.join(timeout=args.step_seconds + REQUEST_TIMEOUT_S + 30)
    with open(out / "phase4_steps.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(steps[0]))
        w.writeheader()
        w.writerows(steps)
    with gzip.open(out / "phase4_requests.csv.gz", "wt", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["step", "target_qps", "latency_ms", "error"])
        w.writerows(raw)
    meta["utc_end"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    config.write_json(out / "phase4_meta.json", meta)
    if not args.keep and sync.has_collection(coll):
        sync.drop_collection(coll)
        print("dropped:", coll)
    print(f"\nresults: {out}")


if __name__ == "__main__":
    main()
