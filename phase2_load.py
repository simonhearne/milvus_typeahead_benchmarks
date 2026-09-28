"""Phase 2: load N e-commerce terms into a fresh collection holding only the phase-1 winners.

Usage: python phase2_load.py [--rows 10000] [--phase1 results/<ts>] [--extra-forms edge.fuzzy_1,...]
Writes results/<ts>/phase2_state.json, which phase3_bench.py consumes.
"""
import argparse
import csv
import json
from pathlib import Path

import config
from collection import setup_collection
from terms import ground_truth, load_terms, rows_from_terms, sample_prefixes
from variants import BY_KEY, resolve

BASELINE_FORMS = ["raw.like_prefix"]  # unindexed baseline, always included
DEFAULT_EXTRA = "edge.fuzzy_1,edge_lc.fuzzy_1"  # typo-tolerant forms kept on purpose


def latest_phase1():
    runs = sorted(p.parent for p in (config.ROOT / "results").glob("*/capability_matrix.csv"))
    if not runs:
        raise SystemExit("no phase-1 results found; run phase1_capability.py first")
    return runs[-1]


def select_forms(p1_dir, extra):
    """{ "variant.form": ["string"] or ["string", "word"] } from the phase-1 matrix."""
    sel = {}
    with open(p1_dir / "capability_matrix.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key = f"{r['variant']}.{r['form']}"
            workloads = []
            if r["PASS_string_prefix"] == "True":
                workloads.append("string")
            if r["word_start_n"] and r["word_start_errors"] == "0" and r["word_start_exact"] == r["word_start_n"]:
                workloads.append("word")
            if workloads:
                sel[key] = workloads
    for key in BASELINE_FORMS:
        sel.setdefault(key, ["string"])
    for key in filter(None, extra.split(",")):
        sel.setdefault(key, ["string", "word"])
    return sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=10_000)
    ap.add_argument("--phase1", type=Path, default=None)
    ap.add_argument("--extra-forms", default=DEFAULT_EXTRA)
    ap.add_argument("--forms", default="",
                    help="explicit selection, skips phase 1: 'ngram.like_prefix:string,ngram.like_wordstart:string+word'")
    ap.add_argument("--per-len", type=int, default=2000)
    ap.add_argument("--lengths", default="3,4,5,6")
    args = ap.parse_args()

    p1 = None if args.forms else (args.phase1 or latest_phase1())
    ts = config.stamp()
    out = config.run_dir(ts)
    c = config.client()
    meta = {"phase": 2, "run": ts, "phase1_dir": str(p1) if p1 else None, "rows_requested": args.rows, **config.run_meta(c)}
    print(f"server={meta['server_version']} pymilvus={meta['pymilvus_version']} connect={meta['tcp_connect_ms_median']}ms")

    if args.forms:
        selected = {k: w.split("+") for k, w in (x.split(":") for x in args.forms.split(","))}
    else:
        selected = select_forms(p1, args.extra_forms)
    keys = sorted({k.split(".")[0] for k in selected})
    variants = resolve(keys)
    meta["selected_forms"] = selected
    meta["variants_in_collection"] = [v.key for v in variants]
    print("selected forms:")
    for k, w in selected.items():
        print(f"  {k:32s} {','.join(w)}")

    terms_csv = out / f"terms_{args.rows // 1000}k.csv"
    print(f"generating {args.rows:,} terms...")
    pairs = load_terms(args.rows, seed=42, out_csv=terms_csv)
    rows = rows_from_terms(pairs)
    meta["rows_loaded"] = len(rows)
    meta["terms_csv"] = terms_csv.name

    main_vars = [v for v in variants if v.key != "vec_pop"]
    name = config.collection_name(f"p2_main_{args.rows}", ts)
    print(f"building {name} ({len(rows)} rows, {len(main_vars)} variants)...")
    log, applied = setup_collection(c, name, main_vars, rows, vec_mode="null", vec_nullable=True)
    meta["collections"] = {"main": {"name": name, "setup_log": log, "applied_indexes": applied,
                                    "count": c.query(name, filter="", output_fields=["count(*)"],
                                                     consistency_level="Strong")[0]["count(*)"]}}
    if any(v.key == "vec_pop" for v in variants):
        vname = config.collection_name(f"p2_vec_{args.rows}", ts)
        print(f"building {vname}...")
        vlog, vapplied = setup_collection(c, vname, resolve(["inv"]), rows, vec_mode="pop", vec_nullable=True)
        meta["collections"]["vec"] = {"name": vname, "setup_log": vlog, "applied_indexes": vapplied}
    for role, info in meta["collections"].items():
        for n, a in info["applied_indexes"].items():
            print(f"  [{role}] index {n:12s} -> {a.get('index_type')} state={a.get('state')} "
                  f"indexed={a.get('indexed_rows')}/{a.get('total_rows')}")

    lengths = tuple(int(x) for x in args.lengths.split(","))
    wl = sample_prefixes(rows, lengths=lengths, per_len=args.per_len, seed=7)
    gt = {mode: {str(L): ground_truth(rows, ps, mode) for L, ps in by_len.items()}
          for mode, by_len in wl.items()}
    (out / "workload.json").write_text(json.dumps({m: {str(L): ps for L, ps in d.items()} for m, d in wl.items()},
                                                  ensure_ascii=False))
    (out / "ground_truth.json").write_text(json.dumps(gt, ensure_ascii=False))
    meta["workload"] = {"lengths": lengths, "per_len": args.per_len, "seed": 7,
                        "weighting": "popularity-weighted term choice"}
    config.write_json(out / "phase2_state.json", meta)
    print(f"\nstate: {out / 'phase2_state.json'}")


if __name__ == "__main__":
    main()
