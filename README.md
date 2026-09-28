# typeahead_bench

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

## Known caveats

- `get_server_version()` on Zilliz Cloud returns only "Compatible with Milvus 3.0". The exact build version comes from the Zilliz control plane. Set `CLUSTER_DB_VERSION` so the build is recorded in run metadata.
- Phase 1 has 10 rows, so prefix `LIKE` and infix `LIKE` can't be told apart there. Phase 2 and 3 correctness at 10k rows is the real test.
- The edge variants match whenever any word starts with the prefix, not only the whole string. Their top-10 overlap against a strict string-prefix ground truth is below 1 by design.
- Phase 3 uses uniform (deterministic) arrivals. Latency is measured from issuing the request; scheduling lag is recorded separately in `sched_lag_ms`.
- At 10k rows, index and scan perform about the same. Re-run with `--rows 1000000` before drawing conclusions about latency at scale.

## License

Apache License 2.0. See [LICENSE](LICENSE).
