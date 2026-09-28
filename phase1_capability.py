"""Phase 1: capability matrix on 10 heterogeneous probe terms.

Usage: python phase1_capability.py [--keep] [--variants k1,k2,...]
"""
import argparse
import csv
import time
import traceback
from collections import defaultdict

from pymilvus import DataType

import config
from collection import SetupError, setup_collection
from terms import is_cjk, probe_prefixes, probe_rows
from variants import BY_KEY, VARIANTS, expr_of, resolve, run_form

COLUMNS = ["string_prefix", "word_start", "multi_word", "case_orig", "typo", "cjk"]


def neg_dim1(c, ts):
    name = config.collection_name("dim1", ts)
    s = c.create_schema(auto_id=False)
    s.add_field("id", DataType.INT64, is_primary=True)
    s.add_field("v", DataType.FLOAT_VECTOR, dim=1, nullable=True)
    try:
        c.create_collection(name, schema=s)
    except Exception as e:  # noqa: BLE001
        return {"rejected": True, "error": f"{type(e).__name__}: {e}"}
    c.drop_collection(name)
    return {"rejected": False, "error": None, "note": "dim=1 was ACCEPTED - verified fact is wrong"}


def is_vec_count_error(e):
    m = str(e).lower()
    return "vector field" in m and ("maximum" in m or "exceed" in m or "num" in m)


def build_main(c, ts, variants, rows, meta):
    """Try all-null nullable vectors first; fall back to populated vectors; split on vector-field limit."""
    main_vars = [v for v in variants if v.key != "vec_pop"]
    attempts = [("null", True), ("pop", True), ("pop", False)]
    for vec_mode, nullable in attempts:
        name = config.collection_name(f"p1_main_{vec_mode}{'' if nullable else '_nn'}", ts)
        try:
            log, applied = setup_collection(c, name, main_vars, rows, vec_mode=vec_mode, vec_nullable=nullable)
            meta["main_collection"] = {"name": name, "vec_mode": vec_mode, "vec_nullable": nullable,
                                       "setup_log": log, "applied_indexes": applied}
            return {"main": name}
        except SetupError as e:
            meta.setdefault("main_setup_failures", []).append(
                {"name": name, "vec_mode": vec_mode, "vec_nullable": nullable, "step": e.step,
                 "error": f"{type(e.err).__name__}: {e.err}"})
            if c.has_collection(name):
                c.drop_collection(name)
            if is_vec_count_error(e.err):
                meta["split_reason"] = str(e.err)
                raise
    raise RuntimeError("main collection could not be created in any vector mode")


def build_vec(c, ts, rows, meta):
    name = config.collection_name("p1_vec", ts)
    try:
        log, applied = setup_collection(c, name, resolve(["inv"]), rows, vec_mode="pop", vec_nullable=True)
        meta["vec_collection"] = {"name": name, "setup_log": log, "applied_indexes": applied}
        return name
    except SetupError as e:
        meta["vec_collection"] = {"name": name, "error": f"{e.step}: {type(e.err).__name__}: {e.err}"}
        return None


def score(returned, expected):
    inter = len(returned & expected)
    recall = inter / len(expected) if expected else 1.0
    precision = inter / len(returned) if returned else (1.0 if not expected else 0.0)
    return precision, recall


def column_of(case, prefix):
    return "cjk" if is_cjk(prefix) else case


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--variants", default=",".join(v.key for v in VARIANTS))
    args = ap.parse_args()

    ts = config.stamp()
    out = config.run_dir(ts)
    c = config.client()
    meta = {"phase": 1, "run": ts, **config.run_meta(c)}
    print(f"server={meta['server_version']} pymilvus={meta['pymilvus_version']} "
          f"endpoint={meta['endpoint_host']} connect={meta['tcp_connect_ms_median']}ms")
    if "warning" in meta:
        print("WARNING:", meta["warning"])

    stale = [n for n in c.list_collections() if n.startswith(config.COLLECTION_PREFIX)]
    for n in stale:
        c.drop_collection(n)
    meta["dropped_stale_collections"] = stale

    meta["neg_dim1"] = neg_dim1(c, ts)
    print("dim=1 negative test:", meta["neg_dim1"])

    variants = resolve(args.variants.split(","))
    rows = probe_rows()
    colls = build_main(c, ts, variants, rows, meta)
    if any(v.key == "vec_pop" for v in variants):
        colls["vec"] = build_vec(c, ts, rows, meta)
    config.write_json(out / "phase1_meta.json", meta)
    print("main:", meta["main_collection"]["name"], "vec_mode=", meta["main_collection"]["vec_mode"])
    for n, a in meta["main_collection"]["applied_indexes"].items():
        print(f"  index {n:12s} -> {a.get('index_type')} {a.get('params') or ''} state={a.get('state')}")

    norm_by_id = {r["id"]: r["term_norm"] for r in rows}
    probes = probe_prefixes(rows)
    records = []
    requested = {k for k in args.variants.split(",")}
    for v in variants:
        if v.key not in requested:
            continue
        for form in v.forms:
            coll = colls.get(form.collection)
            for case, rid, prefix, expect in probes:
                expected = {i for i, n in norm_by_id.items() if expect(n)}
                rec = {"variant": v.key, "form": form.name, "case": case, "column": column_of(case, prefix),
                       "probe_row": rid, "prefix": prefix, "expr": expr_of(form, prefix),
                       "expected_ids": sorted(expected)}
                if coll is None:
                    rec.update(ok=False, error="collection unavailable")
                    records.append(rec)
                    continue
                t = time.perf_counter()
                try:
                    hits = run_form(c, coll, form, prefix, consistency="Strong")
                    rec["latency_ms"] = round((time.perf_counter() - t) * 1000, 2)
                    got = [h["id"] for h in hits]
                    p, r = score(set(got), expected)
                    rec.update(ok=True, error="", returned_ids=got,
                               returned_terms=[h.get("term") for h in hits],
                               precision=round(p, 3), recall=round(r, 3),
                               intended_hit=rid in got)
                except Exception as e:  # noqa: BLE001
                    rec["latency_ms"] = round((time.perf_counter() - t) * 1000, 2)
                    code = getattr(e, "code", None)
                    rec.update(ok=False, error=f"{type(e).__name__}[{code}]: {e}")
                records.append(rec)

    with open(out / "phase1_queries.csv", "w", newline="", encoding="utf-8") as f:
        keys = ["variant", "form", "case", "column", "probe_row", "prefix", "expr", "ok", "error",
                "latency_ms", "precision", "recall", "intended_hit", "expected_ids", "returned_ids",
                "returned_terms"]
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(records)

    matrix = aggregate(records)
    write_matrix(out, matrix, meta)
    config.write_json(out / "phase1_meta.json", meta)

    if not args.keep:
        for n in [meta["main_collection"]["name"], (meta.get("vec_collection") or {}).get("name")]:
            if n and c.has_collection(n):
                c.drop_collection(n)
    print(f"\nresults: {out}")
    print((out / "capability_matrix.md").read_text())


def aggregate(records):
    g = defaultdict(list)
    for r in records:
        g[(r["variant"], r["form"], r["column"])].append(r)
    matrix = defaultdict(dict)
    for (v, f, col), rs in g.items():
        errs = [r for r in rs if not r["ok"]]
        oks = [r for r in rs if r["ok"]]
        cell = {"n": len(rs), "errors": len(errs), "error_sample": errs[0]["error"][:300] if errs else ""}
        if oks:
            cell["recall_min"] = min(r["recall"] for r in oks)
            cell["precision_min"] = min(r["precision"] for r in oks)
            cell["exact"] = sum(1 for r in oks if r["recall"] == 1 and r["precision"] == 1)
            cell["intended_hits"] = sum(1 for r in oks if r["intended_hit"])
            cell["latency_ms_median"] = sorted(r["latency_ms"] for r in oks)[len(oks) // 2]
        matrix[(v, f)][col] = cell
    return matrix


def cell_text(col, cell):
    if not cell:
        return "–"
    if cell["errors"] == cell["n"]:
        return "ERR"
    ok_n = cell["n"] - cell["errors"]
    if col == "typo":
        # typo-tolerant view: did the intended (un-typo'd) row come back?
        return f"{'✅' if cell['intended_hits'] == ok_n else ('◐' if cell['intended_hits'] else '❌')} {cell['intended_hits']}/{ok_n}"
    mark = "✅" if cell["exact"] == ok_n and not cell["errors"] else ("◐" if cell["exact"] else "❌")
    s = f"{mark} {cell['exact']}/{cell['n']}"
    if cell["errors"]:
        s += f" ({cell['errors']} err)"
    return s


def write_matrix(out, matrix, meta):
    rows = []
    for (v, f), cols in matrix.items():
        sp = cols.get("string_prefix") or {}
        passed = bool(sp) and not sp["errors"] and sp.get("exact") == sp["n"]
        row = {"variant": v, "form": f, "PASS_string_prefix": passed}
        for col in COLUMNS:
            cell = cols.get(col) or {}
            row[f"{col}_exact"] = cell.get("exact")
            row[f"{col}_n"] = cell.get("n")
            row[f"{col}_errors"] = cell.get("errors")
            row[f"{col}_recall_min"] = cell.get("recall_min")
            row[f"{col}_precision_min"] = cell.get("precision_min")
            row[f"{col}_intended_hits"] = cell.get("intended_hits")
            row[f"{col}_error_sample"] = cell.get("error_sample")
        row["latency_ms_median_strong"] = (sp or {}).get("latency_ms_median")
        rows.append(row)
    with open(out / "capability_matrix.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    hdr = ["variant", "form", "PASS", "string-prefix", "word-start", "multi-word (word + partial)",
           "case-insensitive", "typo (intended row returned)", "CJK"]
    lines = [f"Server: `{meta['server_version']}` · pymilvus {meta['pymilvus_version']} · "
             f"endpoint {meta['endpoint_host']} · run {meta['run']}", "",
             "| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for r in rows:
        cols = matrix[(r["variant"], r["form"])]
        cells = [cell_text(col, cols.get(col)) for col in COLUMNS]
        lines.append(f"| {r['variant']} | {r['form']} | {'✅' if r['PASS_string_prefix'] else '❌'} | "
                     + " | ".join(cells) + " |")
    lines += ["", "Cells: exact = recall 1.0 and precision 1.0 per probe (x/n). "
              "PASS = every non-CJK string-prefix probe exact. Typo column counts probes where the "
              "intended (un-typo'd) row was returned.", ""]
    errs = [(k, col, cell["error_sample"]) for k, cols in matrix.items() for col, cell in cols.items()
            if cell.get("errors")]
    if errs:
        lines.append("Errors (first sample per cell):")
        seen = set()
        for (v, f), col, msg in errs:
            if (v, f, msg) in seen:
                continue
            seen.add((v, f, msg))
            lines.append(f"- `{v}.{f}` [{col}]: {msg}")
    (out / "capability_matrix.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
