"""Phase 3: open-loop latency benchmark at a fixed request rate.

Arrivals are UNIFORM (deterministic 1/qps spacing), scheduled independently of response time.
Latency = perf_counter from the moment the request is issued to the moment the response is
parsed. Scheduling lag (issue time minus planned time) is recorded separately.

Usage: python phase3_bench.py [--state results/<ts>/phase2_state.json] [--qps 20]
                              [--warmup 10] [--duration 60] [--lengths 3,4,5,6] [--only k1,k2] [--keep]
"""
import argparse
import asyncio
import csv
import json
import random
import statistics
import time
from pathlib import Path

from pymilvus import AsyncMilvusClient

import config
from terms import score_topk
from variants import BY_KEY, call_kwargs, normalise_hits

CONSISTENCY = "Bounded"


def latest_state():
    states = sorted((config.ROOT / "results").glob("*/phase2_state.json"))
    if not states:
        raise SystemExit("no phase-2 state found; run phase2_load.py first")
    return states[-1]


def pct(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


async def open_loop(qps, n, make_call, on_done):
    """Fire n requests at uniform 1/qps spacing; never wait for responses before sending."""
    loop = asyncio.get_running_loop()
    t0 = loop.time() + 0.05
    tasks = []
    for i in range(n):
        planned = t0 + i / qps
        delay = planned - loop.time()
        if delay > 0:
            await asyncio.sleep(delay)
        lag_ms = (loop.time() - planned) * 1000
        tasks.append(asyncio.create_task(_one(i, lag_ms, make_call, on_done)))
    await asyncio.gather(*tasks)
    return loop.time() - t0


async def _one(i, lag_ms, make_call, on_done):
    t = time.perf_counter()
    try:
        res = await make_call(i)
        on_done(i, (time.perf_counter() - t) * 1000, lag_ms, res, None)
    except Exception as e:  # noqa: BLE001 - recorded, no retry
        on_done(i, (time.perf_counter() - t) * 1000, lag_ms, None, e)


def summarise(recs, measured_s):
    lat = [r["latency_ms"] for r in recs if not r["error"]]
    errs = sum(1 for r in recs if r["error"])
    ov = [r["top10_overlap"] for r in recs if r["top10_overlap"] != ""]
    ex = [r["exact_order"] for r in recs if r["exact_order"] != ""]
    return {"n": len(recs), "errors": errs, "error_pct": round(100 * errs / len(recs), 3) if recs else None,
            "achieved_qps": round(len(recs) / measured_s, 2) if measured_s else None,
            **{f"p{p}": round(pct(lat, p), 2) if lat else None for p in (50, 90, 95, 99)},
            "max": round(max(lat), 2) if lat else None,
            "mean": round(statistics.fmean(lat), 2) if lat else None,
            "sched_lag_p99_ms": round(pct([r["sched_lag_ms"] for r in recs], 99), 2) if recs else None,
            "top10_overlap_mean": round(statistics.fmean(ov), 4) if ov else None,
            "exact_order_pct": round(100 * sum(ex) / len(ex), 2) if ex else None,
            "results_mean": round(statistics.fmean(r["n_results"] for r in recs if not r["error"]), 2) if lat else None}


async def run_one(ac, coll, form, mode, L, prefixes, gt, row_by_id, args, rng):
    n_warm, n_meas = int(args.warmup * args.qps), int(args.duration * args.qps)
    seq = [rng.choice(prefixes) for _ in range(n_warm + n_meas)]
    recs = []

    async def make_call(i):
        method, kw = call_kwargs(form, seq[i], CONSISTENCY)
        return method, await getattr(ac, method)(coll, **kw)

    def on_done(i, ms, lag, res, err):
        if i < n_warm:
            return
        p = seq[i]
        rec = {"variant": form_key(form), "mode": mode, "L": L, "prefix": p, "latency_ms": round(ms, 3),
               "sched_lag_ms": round(lag, 3), "n_results": "", "top10_overlap": "", "exact_order": "",
               "error": ""}
        if err is not None:
            rec["error"] = f"{type(err).__name__}[{getattr(err, 'code', None)}]: {str(err)[:300]}"
        else:
            method, raw = res
            hits = normalise_hits(form, method, raw)
            ids = [h["id"] for h in hits]
            ov, exact = score_topk(ids, gt[p], row_by_id, p, mode)
            rec.update(n_results=len(ids), top10_overlap=round(ov, 4), exact_order=int(exact))
        recs.append(rec)

    elapsed = await open_loop(args.qps, n_warm + n_meas, make_call, on_done)
    measured = max(elapsed - args.warmup, 1e-9)
    return recs, measured


def form_key(form):
    return form._key


async def baseline(ac, coll, ids, args, rng, label):
    n_warm, n_meas = int(args.warmup * args.qps), int(args.baseline_duration * args.qps)
    seq = [rng.choice(ids) for _ in range(n_warm + n_meas)]
    recs = []

    async def make_call(i):
        return await ac.get(coll, ids=[seq[i]], output_fields=["id"], consistency_level=CONSISTENCY)

    def on_done(i, ms, lag, res, err):
        if i >= n_warm:
            recs.append({"variant": label, "mode": "get", "L": "", "prefix": seq[i], "latency_ms": round(ms, 3),
                         "sched_lag_ms": round(lag, 3), "n_results": len(res) if res else "",
                         "top10_overlap": "", "exact_order": "",
                         "error": "" if err is None else f"{type(err).__name__}: {str(err)[:300]}"})

    elapsed = await open_loop(args.qps, n_warm + n_meas, make_call, on_done)
    return recs, max(elapsed - args.warmup, 1e-9)


async def amain(args):
    state_path = args.state or latest_state()
    state = json.loads(Path(state_path).read_text())
    p2 = Path(state_path).parent
    ts = config.stamp()
    out = config.run_dir(ts)
    sync = config.client()
    meta = {"phase": 3, "run": ts, "phase2_state": str(state_path), "qps": args.qps, "arrivals": "uniform",
            "warmup_s": args.warmup, "duration_s": args.duration, "consistency": CONSISTENCY,
            **config.run_meta(sync)}
    print(f"server={meta['server_version']} pymilvus={meta['pymilvus_version']} "
          f"client={meta['client_hostname']} connect={meta['tcp_connect_ms_median']}ms")
    if "warning" in meta:
        print("WARNING:", meta["warning"])

    with open(p2 / state["terms_csv"], encoding="utf-8") as f:
        from terms import rows_from_terms
        rows = rows_from_terms((r["term"], int(r["popularity"])) for r in csv.DictReader(f))
    row_by_id = {r["id"]: r for r in rows}
    workload = json.loads((p2 / "workload.json").read_text())
    gt = json.loads((p2 / "ground_truth.json").read_text())
    lengths = [str(x) for x in args.lengths.split(",")]
    colls = {role: info["name"] for role, info in state["collections"].items()}

    runs = []
    only = set(filter(None, (args.only or "").split(",")))
    for key, modes in state["selected_forms"].items():
        if only and key not in only:
            continue
        vkey, fname = key.split(".")
        form = next(f for f in BY_KEY[vkey].forms if f.name == fname)
        form._key = key
        for mode in modes:
            for L in lengths:
                runs.append((form, mode, L))
    rng = random.Random(args.seed)
    rng.shuffle(runs)
    meta["run_order"] = [(form_key(f), m, L) for f, m, L in runs]
    est = (len(runs) * (args.warmup + args.duration) + 2 * (args.warmup + args.baseline_duration)) / 60
    print(f"{len(runs)} runs, est. {est:.0f} min")

    ac = AsyncMilvusClient(uri=config.ZILLIZ_URI, token=config.ZILLIZ_TOKEN, db_name=config.ZILLIZ_DB)
    all_recs, summary = [], []
    fields = ["variant", "mode", "L", "prefix", "latency_ms", "sched_lag_ms", "n_results", "top10_overlap",
              "exact_order", "error"]
    try:
        ids = list(row_by_id)
        for label in ("baseline_get_start",):
            recs, measured = await baseline(ac, colls["main"], ids, args, rng, label)
            summary.append({"variant": label, "mode": "get", "L": "", **summarise(recs, measured)})
            all_recs += recs
            print(f"  {label:34s} p50={summary[-1]['p50']} p95={summary[-1]['p95']} p99={summary[-1]['p99']}")
        for i, (form, mode, L) in enumerate(runs, 1):
            coll = colls[form.collection]
            recs, measured = await run_one(ac, coll, form, mode, int(L), workload[mode][L], gt[mode][L],
                                           row_by_id, args, rng)
            s = {"variant": form_key(form), "mode": mode, "L": int(L), **summarise(recs, measured)}
            summary.append(s)
            all_recs += recs
            print(f"  [{i}/{len(runs)}] {form_key(form):32s} {mode:6s} L={L} p50={s['p50']} p95={s['p95']} "
                  f"p99={s['p99']} err={s['error_pct']}% top10={s['top10_overlap_mean']} qps={s['achieved_qps']}")
        recs, measured = await baseline(ac, colls["main"], ids, args, rng, "baseline_get_end")
        summary.append({"variant": "baseline_get_end", "mode": "get", "L": "", **summarise(recs, measured)})
        all_recs += recs
    finally:
        await ac.close()
        with open(out / "phase3_requests.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(all_recs)
        if summary:
            with open(out / "phase3_summary.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(summary[-1]))
                w.writeheader()
                w.writerows(summary)
        config.write_json(out / "phase3_meta.json", meta)

    if not args.keep:
        for n in colls.values():
            if sync.has_collection(n):
                sync.drop_collection(n)
        print("dropped:", ", ".join(colls.values()))
    print(f"\nresults: {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", type=Path, default=None)
    ap.add_argument("--qps", type=float, default=config.PHASE3_QPS)
    ap.add_argument("--warmup", type=float, default=10)
    ap.add_argument("--duration", type=float, default=60)
    ap.add_argument("--baseline-duration", type=float, default=60)
    ap.add_argument("--lengths", default="3,4,5,6")
    ap.add_argument("--only", default="", help="comma-separated variant.form keys")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--keep", action="store_true")
    asyncio.run(amain(ap.parse_args()))


if __name__ == "__main__":
    main()
