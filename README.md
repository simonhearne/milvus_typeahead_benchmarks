# Milvus Typeahead Benchmarks

This project tests which Milvus field, index and query combinations return correct prefix matches for e-commerce typeahead. It then measures their latency at a steady 20 QPS on Zilliz Cloud. Every capability claim is treated as a hypothesis, and the test results are the source of truth.

## Run

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
cp .env.example .env   # set ZILLIZ_URI, ZILLIZ_TOKEN, ZILLIZ_DB (read from ./.env or ../.env)

.venv/bin/python phase1_capability.py            # capability matrix on 10 probe terms
.venv/bin/python phase2_load.py [--rows 10000]   # load terms, keep only the phase-1 winners
.venv/bin/python phase3_bench.py                 # 20 QPS open loop, drops collections unless --keep
.venv/bin/python report.py                       # report.md + charts in the phase-3 results dir
.venv/bin/python phase4_ramp.py --form ngram.like_prefix   # optional: open-loop QPS ramp until failure
```

Each phase writes to `results/<UTC timestamp>/` (gitignored). Phase 2 reads the latest phase-1 matrix, and phase 3 reads the latest phase-2 state, unless you pass `--phase1` or `--state`. Collections are named `ta_bench_<tag>_<ts>`. Phase 1 drops any `ta_bench_*` collections left over from earlier runs.

For a quick smoke test, run `phase3_bench.py --warmup 1 --duration 3 --baseline-duration 3 --lengths 3,6`. The full default (22 forms, 1–2 workloads each, 4 prefix lengths, 70 s per run) takes about 3 h.

### In-region client VM (required for meaningful latency)

The cluster is in AWS eu-west-1. From outside the region, every number is dominated by WAN round-trip time (≈17 ms from a laptop in the UK).

```bash
./vm.sh up        # c7i.large, AL2023, eu-west-1a, SSH open to this IP only
./vm.sh push      # copies the harness and .env, installs pinned deps, prints the TCP floor
./vm.sh run       # phase1 → phase2 → phase3 → report (PHASE2_ARGS / PHASE3_ARGS pass through)
./vm.sh status    # follow the log
./vm.sh fetch     # copies results back and verifies the phase-3 files arrived
./vm.sh down      # terminates the VM and deletes the SG and key pair; refuses until fetch is verified
```

## Variants

Each variant is its own field in one collection, because a field can hold only one index. Every row also stores `id`, `term`, `term_norm` (NFKC, lowercased, trimmed, whitespace collapsed), `popularity` (INT64, STL_SORT index) and `pop_vec`. `pop_vec` is a nullable FLOAT_VECTOR with dim=2, all null, with an AUTOINDEX/IP index. Milvus requires at least one vector field, and dim=1 is rejected.

| key | field contents | index | how it matches |
|---|---|---|---|
| raw | term_norm | none | `like "p%"` / `like "%p%"` (brute-force baseline) |
| inv | term_norm | INVERTED | `like` prefix / infix |
| trie | term_norm | TRIE | `like` prefix |
| ngram | term_norm | NGRAM 2–3 | `like` prefix / infix / word-start (`p%` OR `% p%`) |
| ana_std / ana_custom | original term | analyzer + match index | `text_match`, `phrase_match`, `text_match_fuzzy` (whole tokens only) |
| edge | client-side edge-ngrams (1..15) of each word | whitespace analyzer + match index | `text_match`; `_and` / `_msm` handle multi-word input |
| edge_lc | same as edge | whitespace + lowercase analyzer | as edge, and case-insensitive |
| bm25_std / bm25_edge | sparse BM25 function on t_std / t_edge | SPARSE_INVERTED_INDEX | `search`; `_over100_popsort` over-fetches and re-sorts by popularity on the client |
| arr_prefix | ARRAY<VARCHAR> of edge-ngrams | INVERTED | `array_contains` / `array_contains_all` |
| vec_pop | second collection, `pop_vec=[log1p(pop),0]` | AUTOINDEX IP | ANN ranking plus a `t_inv like "p%"` filter |

Filter-only variants run `query(limit=10, order_by_fields=["popularity:desc"])`. To add a variant, append one `Variant(...)` to `variants.py`.

## Results

These results come from runs on 2026-09-25. The cluster was a Zilliz Cloud Dedicated cluster in AWS eu-west-1, and the client was pymilvus 3.0.2 on a c7i.large VM in the same region (TCP connect ≈1.6 ms). All data was synthetic terms from `terms.py`, and results are top-10 by popularity. Raw per-run output is not in the repo. Run the phases to regenerate it.

### Correctness

"Overlap" is the share of the true top-10 that came back. Phase 3 measured it at 10k rows over 1,200 requests per variant × workload × prefix length (string-prefix and word-start workloads, L = 3–6).

| need | works (overlap ≥ 0.996) | doesn't |
|---|---|---|
| whole-string prefix (`sony w…`) | `like "p%"` on raw / INVERTED / TRIE / NGRAM; `vec_pop` ANN + `like` filter | analyzer `text_match` / `phrase_match` and `bm25_std` match whole tokens only (3/9 probes); `bm25_edge` ranks by relevance, not popularity (0.07–0.25, or 0.45–0.58 with over-fetch + client sort) |
| any-word prefix (`wh…` → "Sony WH-1000XM5") | NGRAM `like_wordstart` (`p%` OR `% p%`); edge / edge_lc `text_match`, `_and`, `_msm`; `arr_prefix.array_contains_all` | `like "% p%"` alone misses the first word; `like "%p%"` returns substrings (0.84–0.93) |
| raw-case input | `edge_lc` (lowercase analyzer) | every `like` variant is case-sensitive, so normalise on write and on query |
| typos | `edge*.fuzzy_1/2` return the intended row (9/9 probes) | at the cost of precision (overlap 0.41–0.98), so fuzzy is a fallback, not the primary query |

### Latency

- **10k rows, 20 QPS, all 22 forms:** p95 was 3.6–5.8 ms for every variant, against a `get(pk)` baseline p95 of 3.5 ms. Server-side cost was under ≈2 ms for everything, and at this size index choice makes no measurable difference. `vec_pop` was the slowest (p95 4.7–5.8 ms).
- **100k rows, 100 QPS:** `ngram.like_prefix` p95 was 4.0 ms at every L, and `ngram.like_wordstart` p95 was 4.2–4.6 ms. There were 0 errors.
- **Capacity ramp (`phase4_ramp.py`, `ngram.like_prefix`, 100k rows, 100 → 2000 QPS in 60 s steps):** there were 0 errors at every step, and the client stayed below 15 % CPU with 1.1 ms p99 scheduling lag. p95 stayed at or below 3.6 ms through 1,700 QPS. The knee is at about 1,800 QPS: p95 was 4.5 ms at 1,800, 13 ms at 1,900 and 29 ms at 2,000. The 50 ms p95 failure rule never tripped within the 2,000 QPS ceiling.

**Recommendation:**
- For whole-string prefix, use an NGRAM index on a normalised field with `like "p%"` and `order_by popularity:desc`.
- For match-any-word, use the same field with `like_wordstart`.
- Use `edge_lc` only if you can't normalise the query on the client.

Re-test at 1M+ rows before relying on relative latency.

## Known caveats

- `get_server_version()` on Zilliz Cloud returns only "Compatible with Milvus 3.0". The exact build version comes from the Zilliz control plane. Set `CLUSTER_DB_VERSION` so the build is recorded in run metadata.
- Phase 1 has 10 rows, so prefix `LIKE` and infix `LIKE` can't be told apart there. Phase 2 and 3 correctness at 10k rows is the real test.
- The edge variants match whenever any word starts with the prefix, not only the whole string. Their top-10 overlap against a strict string-prefix ground truth is below 1 by design.
- Phase 3 uses uniform (deterministic) arrivals. Latency is measured from issuing the request; scheduling lag is recorded separately in `sched_lag_ms`.
- At 10k rows, index and scan perform about the same. Re-run with `--rows 1000000` before drawing conclusions about latency at scale.

## License

Apache License 2.0. See [LICENSE](LICENSE).
