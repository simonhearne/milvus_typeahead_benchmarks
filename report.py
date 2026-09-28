"""Build report.md + charts from a phase-3 run (and the phase-2 / phase-1 runs it points to).

Usage: python report.py [results/<phase3_ts>]
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import config  # noqa: E402

# Reference palette (dataviz skill), validated: ordinal blue ramp for L, categorical slots 1-5.
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
L_COLORS = {3: "#86b6ef", 4: "#3987e5", 5: "#1c5cab", 6: "#0d366b"}
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
CDF_HIGHLIGHT = ["baseline_get_start", "raw.like_prefix", "inv.like_prefix", "ngram.like_prefix",
                 "edge_lc.text_match_and"]

CAVEATS = {
    "raw": "No index: every query scans the field. Case-sensitive (store/query normalised text).",
    "inv": "Whole value is one term; `like 'p%'` is a prefix scan of the term dictionary. Case-sensitive. "
           "`like '%p%'` returns substrings, not just word starts.",
    "trie": "Prefix only. Case-sensitive.",
    "ngram": "Prefix/infix/word-start via LIKE; literal must be >= min_gram or it falls back to brute force. "
             "Case-sensitive.",
    "edge": "Word-start semantics (any word), not whole-string prefix. Needs client-side edge-ngram "
            "expansion at write time. No lowercase filter: case-sensitive.",
    "edge_lc": "As edge, but the analyzer lowercases both sides, so raw-case input works.",
    "bm25_edge": "Ranking is BM25 relevance, not popularity; over-fetch 100 + client sort recovers part of it.",
    "arr_prefix": "Array of edge-ngrams; `array_contains_all` = AND over typed words. Case-sensitive.",
    "vec_pop": "Ranking from IP on [log1p(pop), 0]; needs a populated vector and a second collection here.",
}

VERBATIM_CAVEATS = [
    "At 10k rows, index vs scan differences are small because everything fits in one or a few segments. "
    "{scale_note} Re-run with `python phase2_load.py --rows 1000000` then `phase3_bench.py` to repeat at 1M "
    "with generated terms.",
    "20 QPS is far below real keystroke traffic. It measures latency, not capacity.",
]


def read_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def main():
    p3 = Path(sys.argv[1]) if len(sys.argv) > 1 else sorted(
        p.parent for p in (config.ROOT / "results").glob("*/phase3_summary.csv"))[-1]
    m3 = json.loads((p3 / "phase3_meta.json").read_text())
    p2 = Path(m3["phase2_state"]).parent
    if not p2.exists():  # state path recorded on another machine
        p2 = config.ROOT / "results" / p2.name
    m2 = json.loads((p2 / "phase2_state.json").read_text())
    p1 = Path(m2["phase1_dir"]) if m2.get("phase1_dir") else None  # None when phase 2 ran with --forms
    if p1 and not p1.exists():
        p1 = config.ROOT / "results" / p1.name
    summary = read_csv(p3 / "phase3_summary.csv")
    reqs = read_csv(p3 / "phase3_requests.csv")

    base = [s for s in summary if s["mode"] == "get"]
    base_p50 = min(fnum(s["p50"]) for s in base)
    base_p95 = min(fnum(s["p95"]) for s in base)
    runs = [s for s in summary if s["mode"] != "get"]

    chart_p95(runs, base_p95, p3 / "p95_vs_L.png")
    chart_cdf(reqs, p3 / "cdf_L3.png")

    out = []
    out.append("# Milvus typeahead benchmark: capability and latency\n")
    out.append("## 1. Setup\n")
    out.append(f"- Server version: `{m3['server_version']}`; cluster build (control plane): "
               f"`{m3.get('cluster_db_version') or 'not recorded'}`; pymilvus {m3['pymilvus_version']}")
    out.append(f"- Cluster endpoint: `{m3['endpoint_host']}` (region `{m3['endpoint_region']}`)")
    out.append(f"- Client: `{m3['client_hostname']}` ({m3['client_platform']}); median TCP connect "
               f"{m3['tcp_connect_ms_median']} ms" + (f". **{m3['warning']}**" if m3.get("warning") else ""))
    out.append(f"- Rows: {m2['rows_loaded']:,} (server count {m2['collections']['main']['count']:,}); "
               f"phase-3 load: {m3['qps']} QPS open loop, {m3['arrivals']} arrivals, {m3['warmup_s']} s warm-up + "
               f"{m3['duration_s']} s measured per variant × L, consistency {m3['consistency']}")
    out.append(f"- Runs: phase 1 `{p1.name if p1 else 'skipped (--forms)'}`, phase 2 `{p2.name}`, phase 3 `{p3.name}`\n")
    out.append("Applied index per field (from `describe_index` after the build finished):\n")
    out.append("| collection | field | index | params | indexed / total |\n|---|---|---|---|---|")
    for role, info in m2["collections"].items():
        for n, a in info["applied_indexes"].items():
            params = {k: v for k, v in a["raw"].items() if k in ("min_gram", "max_gram", "metric_type")}
            out.append(f"| {role} | {n} | {a.get('index_type')} | {params or ''} | "
                       f"{a.get('indexed_rows')}/{a.get('total_rows')} |")
    out.append("\nText-match fields (`enable_match=True`) carry an implicit match index that "
               "`describe_index` does not list.\n")

    out.append("## 2. Capability matrix (phase 1, 10 probe terms)\n")
    if p1 and (p1 / "capability_matrix.md").exists():
        out.append((p1 / "capability_matrix.md").read_text())
    else:
        out.append(f"Phase 1 skipped: forms chosen explicitly ({', '.join(m2['selected_forms'])}).\n")

    out.append("## 3. Latency (phase 3)\n")
    out.append(f"Baseline `get(pk)` RTT at the same rate: p50 {base_p50:.2f} ms, p95 {base_p95:.2f} ms. "
               "The `−RTT` columns subtract the baseline p50 and are secondary: they approximate server-side "
               "cost, not user-visible latency.\n")
    out.append("| variant.form | workload | L | p50 | p95 | p99 | max | p50 −RTT | p95 −RTT | err % | "
               "top-10 overlap | exact order % | qps |\n|" + "---|" * 13)
    for s in sorted(runs, key=lambda s: (s["variant"], s["mode"], int(s["L"]))):
        out.append(f"| {s['variant']} | {s['mode']} | {s['L']} | {s['p50']} | {s['p95']} | {s['p99']} | "
                   f"{s['max']} | {fnum(s['p50']) - base_p50:.2f} | {fnum(s['p95']) - base_p50:.2f} | "
                   f"{s['error_pct']} | {s['top10_overlap_mean']} | {s['exact_order_pct']} | {s['achieved_qps']} |")
    for s in base:
        out.append(f"| {s['variant']} | get | – | {s['p50']} | {s['p95']} | {s['p99']} | {s['max']} | – | – | "
                   f"{s['error_pct']} | – | – | {s['achieved_qps']} |")
    out.append("\n`top-10 overlap`: share of the expected top-min(10, matches) rows by popularity that came back "
               "(ties at the boundary are accepted). `exact order`: the returned id list equals the ground-truth "
               "top-10 order exactly; only expected for ORDER BY variants.\n")

    out.append("## 4. Charts\n")
    out.append("![p95 latency vs prefix length](p95_vs_L.png)\n")
    out.append("![Latency CDF at L=3](cdf_L3.png)\n")

    out.append("## 5. Findings\n")
    by_var = defaultdict(list)
    for s in runs:
        by_var[s["variant"]].append(s)
    for key in sorted(by_var):
        ss = by_var[key]
        p95s = [fnum(s["p95"]) for s in ss if s["p95"]]
        ov = {s["mode"]: [] for s in ss}
        for s in ss:
            ov[s["mode"]].append(fnum(s["top10_overlap_mean"]) or 0)
        ov_txt = ", ".join(f"{m} top-10 {min(v):.2f}–{max(v):.2f}" for m, v in ov.items())
        errs = sum(int(s["errors"]) for s in ss)
        out.append(f"- **{key}**: p95 {min(p95s):.1f}–{max(p95s):.1f} ms across L; {ov_txt}; "
                   f"{errs} errors. {CAVEATS.get(key.split('.')[0], '')}")
    out.append("")

    out.append("## 6. Caveats\n")
    scale_note = ("Conclusions about which variants are *correct* hold at any scale; conclusions about "
                  "relative *latency* are not expected to hold at 1M+ rows, where unindexed scans and "
                  "broad 3-character prefixes (many matches to sort) diverge from indexed lookups.")
    for c in VERBATIM_CAVEATS:
        out.append("- " + c.format(scale_note=scale_note))
    if m3.get("warning"):
        out.append(f"- {m3['warning']}")
    (p3 / "report.md").write_text("\n".join(out) + "\n")
    print(p3 / "report.md")


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def chart_p95(runs, base_p95, path):
    modes = ["string", "word"]
    keys = sorted({s["variant"] for s in runs},
                  key=lambda k: next((fnum(s["p95"]) for s in runs
                                      if s["variant"] == k and s["mode"] == "string" and s["L"] == "3"), 1e9))
    fig, axes = plt.subplots(1, 2, figsize=(12, 0.32 * len(keys) + 1.6), sharey=True, facecolor=SURFACE)
    xmax = max(fnum(s["p95"]) for s in runs) * 1.08
    for ax, mode in zip(axes, modes):
        _style(ax)
        for y, k in enumerate(keys):
            for s in runs:
                if s["variant"] == k and s["mode"] == mode:
                    L = int(s["L"])
                    ax.plot(fnum(s["p95"]), y, "o", ms=8, color=L_COLORS.get(L, INK2),
                            markeredgecolor=SURFACE, markeredgewidth=1.5, label=f"L={L}")
        ax.axvline(base_p95, color=INK2, linestyle="--", linewidth=1)
        ax.text(base_p95, len(keys) - 0.3, " get(pk) p95", color=INK2, fontsize=8, va="bottom")
        ax.set_xlim(0, max(xmax, base_p95 * 1.2))
        ax.set_title(f"{mode}-prefix workload", color=INK, fontsize=10, loc="left")
        ax.set_xlabel("p95 latency (ms), client-side", color=INK2, fontsize=9)
    axes[0].set_yticks(range(len(keys)), keys, fontsize=8, color=INK)
    axes[0].set_ylim(-0.7, len(keys) - 0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    order = sorted(uniq, key=lambda s: int(s.split("=")[1]))
    fig.legend([uniq[o] for o in order], order, loc="upper right", ncol=4, frameon=False, fontsize=8,
               labelcolor=INK2)
    fig.suptitle("p95 latency by prefix length (rows sorted by string L=3)", x=0.01, ha="left",
                 color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def chart_cdf(reqs, path):
    series = defaultdict(list)
    for r in reqs:
        if r["error"]:
            continue
        if r["mode"] == "get" and r["variant"] == "baseline_get_start":
            series[r["variant"]].append(float(r["latency_ms"]))
        elif r["mode"] == "string" and r["L"] == "3":
            series[r["variant"]].append(float(r["latency_ms"]))
    fig, ax = plt.subplots(figsize=(9, 5), facecolor=SURFACE)
    _style(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    hi = []
    for k, xs in series.items():
        if k in CDF_HIGHLIGHT:
            continue
        xs = sorted(xs)
        ax.plot(xs, [(i + 1) / len(xs) for i in range(len(xs))], color="#c9c8c3", linewidth=1)
    for i, k in enumerate(k for k in CDF_HIGHLIGHT if k in series):
        xs = sorted(series[k])
        hi.append(xs[int(0.995 * (len(xs) - 1))])
        ax.plot(xs, [(j + 1) / len(xs) for j in range(len(xs))], color=CATEGORICAL[i], linewidth=2, label=k)
    if hi:
        ax.set_xlim(left=0, right=max(hi) * 1.15)
    ax.set_ylim(0, 1.005)
    ax.set_xlabel("client latency (ms)", color=INK2, fontsize=9)
    ax.set_ylabel("share of requests", color=INK2, fontsize=9)
    ax.plot([], [], color="#c9c8c3", linewidth=1, label="other variants")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="lower right")
    ax.set_title("Latency CDF, string prefix, L = 3", color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    main()
