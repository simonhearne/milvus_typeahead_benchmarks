"""Variant registry: one entry per field/index strategy, with its query forms.

Adding a variant = appending one Variant(...) to VARIANTS.
"""
import math
from dataclasses import dataclass, field
from typing import Callable

from pymilvus import DataType, Function, FunctionType

from config import NGRAM_MAX, NGRAM_MIN, VARCHAR_MAX
from terms import edge_ngrams

TOP_K = 10
OUT_FIELDS = ["term", "popularity"]
ORDER_BY = ["popularity:desc"]
CUSTOM_ANALYZER = {"tokenizer": "standard",
                   "filter": ["lowercase", "asciifolding", {"type": "stemmer", "language": "english"}]}
EDGE_ANALYZER = {"tokenizer": "whitespace"}
EDGE_LC_ANALYZER = {"tokenizer": "whitespace", "filter": ["lowercase"]}


# ------------------------------------------------------------------ literals
def q(s):
    """Milvus double-quoted string literal."""
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def like_escape(s):
    return s.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def like(fld, pattern_fmt, p):
    return f"{fld} like {q(pattern_fmt.format(p=like_escape(p)))}"


# ------------------------------------------------------------------ model
@dataclass
class Form:
    name: str
    kind: str                      # "query" (filter + ORDER BY) | "search" (ANN / BM25)
    build: Callable[[str], dict]   # prefix -> kwargs fragment
    collection: str = "main"       # which collection it runs against
    limit: int = 10                # server-side limit
    client_sort: bool = False      # re-rank hits by popularity client-side, then cut to TOP_K


@dataclass
class Variant:
    key: str
    fields: list                   # [(name, DataType, kwargs)]
    value: Callable[[dict], dict]  # base row -> {field: value}
    indexes: list = field(default_factory=list)    # add_index kwargs
    functions: list = field(default_factory=list)  # Function objects
    forms: list = field(default_factory=list)
    requires: list = field(default_factory=list)   # other variant keys whose fields are inputs
    note: str = ""


def _text_forms(fld):
    return [
        Form("text_match", "query", lambda p: {"filter": f"text_match({fld}, {q(p)})"}),
        Form("phrase_match", "query", lambda p: {"filter": f"phrase_match({fld}, {q(p)}, 0)"}),
        Form("fuzzy_1", "query", lambda p: {"filter": f"text_match_fuzzy({fld}, {q(p)}, max_edit_distance=1)"}),
        Form("fuzzy_2", "query", lambda p: {"filter": f"text_match_fuzzy({fld}, {q(p)}, max_edit_distance=2)"}),
        Form("like_prefix", "query", lambda p: {"filter": like(fld, "{p}%", p)}),
    ]


def _bm25_search(fld):
    return lambda p: {"data": [p], "anns_field": fld, "search_params": {"metric_type": "BM25"}}


def _edge_and(fld):
    def build(p):
        toks = p.split()
        return {"filter": " and ".join(f"text_match({fld}, {q(t)})" for t in toks) or "false"}
    return build


def _edge_msm(fld):
    def build(p):
        toks = p.split()
        return {"filter": f"text_match({fld}, {q(' '.join(toks))}, minimum_should_match={max(1, len(toks))})"}
    return build


def _edge_forms(fld):
    return [Form("text_match", "query", lambda p: {"filter": f"text_match({fld}, {q(p)})"}),
            Form("text_match_and", "query", _edge_and(fld)),
            Form("text_match_msm", "query", _edge_msm(fld)),
            Form("fuzzy_1", "query", lambda p: {"filter": f"text_match_fuzzy({fld}, {q(p)}, max_edit_distance=1)"}),
            Form("fuzzy_2", "query", lambda p: {"filter": f"text_match_fuzzy({fld}, {q(p)}, max_edit_distance=2)"})]


def _bm25_forms(fld):
    return [Form("bm25", "search", _bm25_search(fld)),
            Form("bm25_over100_popsort", "search", _bm25_search(fld), limit=100, client_sort=True)]


VARIANTS = [
    Variant("raw", [("t_raw", DataType.VARCHAR, {"max_length": VARCHAR_MAX})],
            lambda r: {"t_raw": r["term_norm"]},
            forms=[Form("like_prefix", "query", lambda p: {"filter": like("t_raw", "{p}%", p)}),
                   Form("like_infix", "query", lambda p: {"filter": like("t_raw", "%{p}%", p)})],
            note="term_norm, no index (brute-force baseline)"),
    Variant("inv", [("t_inv", DataType.VARCHAR, {"max_length": VARCHAR_MAX})],
            lambda r: {"t_inv": r["term_norm"]},
            indexes=[{"field_name": "t_inv", "index_type": "INVERTED"}],
            forms=[Form("like_prefix", "query", lambda p: {"filter": like("t_inv", "{p}%", p)}),
                   Form("like_infix", "query", lambda p: {"filter": like("t_inv", "%{p}%", p)})],
            note="term_norm, INVERTED"),
    Variant("trie", [("t_trie", DataType.VARCHAR, {"max_length": VARCHAR_MAX})],
            lambda r: {"t_trie": r["term_norm"]},
            indexes=[{"field_name": "t_trie", "index_type": "TRIE"}],
            forms=[Form("like_prefix", "query", lambda p: {"filter": like("t_trie", "{p}%", p)})],
            note="term_norm, TRIE"),
    Variant("ngram", [("t_ngram", DataType.VARCHAR, {"max_length": VARCHAR_MAX})],
            lambda r: {"t_ngram": r["term_norm"]},
            indexes=[{"field_name": "t_ngram", "index_type": "NGRAM",
                      "params": {"min_gram": NGRAM_MIN, "max_gram": NGRAM_MAX}}],
            forms=[Form("like_prefix", "query", lambda p: {"filter": like("t_ngram", "{p}%", p)}),
                   Form("like_infix", "query", lambda p: {"filter": like("t_ngram", "%{p}%", p)}),
                   Form("like_space_infix", "query", lambda p: {"filter": like("t_ngram", "% {p}%", p)}),
                   Form("like_wordstart", "query",
                        lambda p: {"filter": f"({like('t_ngram', '{p}%', p)}) or ({like('t_ngram', '% {p}%', p)})"})],
            note=f"term_norm, NGRAM min={NGRAM_MIN} max={NGRAM_MAX}"),
    Variant("ana_std", [("t_std", DataType.VARCHAR,
                         {"max_length": VARCHAR_MAX, "enable_analyzer": True, "enable_match": True})],
            lambda r: {"t_std": r["term"]},
            forms=_text_forms("t_std"),
            note="original term, default (standard) analyzer"),
    Variant("ana_custom", [("t_custom", DataType.VARCHAR,
                            {"max_length": VARCHAR_MAX, "enable_analyzer": True, "enable_match": True,
                             "analyzer_params": CUSTOM_ANALYZER})],
            lambda r: {"t_custom": r["term"]},
            forms=_text_forms("t_custom"),
            note="original term, standard+lowercase+asciifolding+english stemmer"),
    Variant("edge", [("t_edge", DataType.VARCHAR,
                      {"max_length": VARCHAR_MAX * 4, "enable_analyzer": True, "enable_match": True,
                       "analyzer_params": EDGE_ANALYZER})],
            lambda r: {"t_edge": " ".join(edge_ngrams(r["term_norm"]))},
            forms=_edge_forms("t_edge"),
            note="client-side edge-ngrams (1..15) of term_norm words, whitespace analyzer"),
    Variant("edge_lc", [("t_edge_lc", DataType.VARCHAR,
                         {"max_length": VARCHAR_MAX * 4, "enable_analyzer": True, "enable_match": True,
                          "analyzer_params": EDGE_LC_ANALYZER})],
            lambda r: {"t_edge_lc": " ".join(edge_ngrams(r["term_norm"]))},
            forms=_edge_forms("t_edge_lc"),
            note="as edge, whitespace analyzer + lowercase filter (query side lowercased by server)"),
    Variant("bm25_std", [("sp_std", DataType.SPARSE_FLOAT_VECTOR, {})],
            lambda r: {},
            indexes=[{"field_name": "sp_std", "index_type": "SPARSE_INVERTED_INDEX", "metric_type": "BM25"}],
            functions=[Function(name="bm25_std", function_type=FunctionType.BM25,
                                input_field_names=["t_std"], output_field_names=["sp_std"])],
            forms=_bm25_forms("sp_std"),
            requires=["ana_std"], note="BM25 over t_std"),
    Variant("bm25_edge", [("sp_edge", DataType.SPARSE_FLOAT_VECTOR, {})],
            lambda r: {},
            indexes=[{"field_name": "sp_edge", "index_type": "SPARSE_INVERTED_INDEX", "metric_type": "BM25"}],
            functions=[Function(name="bm25_edge", function_type=FunctionType.BM25,
                                input_field_names=["t_edge"], output_field_names=["sp_edge"])],
            forms=_bm25_forms("sp_edge"),
            requires=["edge"], note="BM25 over t_edge"),
    Variant("arr_prefix", [("a_prefix", DataType.ARRAY,
                            {"element_type": DataType.VARCHAR, "max_capacity": 256, "max_length": 64})],
            lambda r: {"a_prefix": edge_ngrams(r["term_norm"])},
            indexes=[{"field_name": "a_prefix", "index_type": "INVERTED"}],
            forms=[Form("array_contains", "query", lambda p: {"filter": f"array_contains(a_prefix, {q(p)})"}),
                   Form("array_contains_all", "query",
                        lambda p: {"filter": f"array_contains_all(a_prefix, [{', '.join(q(t) for t in p.split()) or q(p)}])"})],
            note="ARRAY<VARCHAR> of edge-ngrams, INVERTED"),
    # vec_pop runs in a second collection whose pop_vec = [log1p(pop), 0] and reuses t_inv.
    Variant("vec_pop", [], lambda r: {},
            forms=[Form("ann_filter_like", "search",
                        lambda p: {"data": [[1.0, 0.0]], "anns_field": "pop_vec",
                                   "search_params": {"metric_type": "IP"},
                                   "filter": like("t_inv", "{p}%", p)}, collection="vec")],
            requires=["inv"], note="IP search on [log1p(pop),0] + t_inv prefix filter"),
]
BY_KEY = {v.key: v for v in VARIANTS}


def resolve(keys):
    """Expand requires, preserving registry order."""
    need, stack = set(), list(keys)
    while stack:
        k = stack.pop()
        if k not in need:
            need.add(k)
            stack.extend(BY_KEY[k].requires)
    return [v for v in VARIANTS if v.key in need]


# ------------------------------------------------------------------ schema / data
def build_schema(client, variants, vec_nullable=True):
    s = client.create_schema(auto_id=False, enable_dynamic_field=False)
    s.add_field("id", DataType.INT64, is_primary=True)
    s.add_field("term", DataType.VARCHAR, max_length=VARCHAR_MAX)
    s.add_field("term_norm", DataType.VARCHAR, max_length=VARCHAR_MAX)
    s.add_field("popularity", DataType.INT64)
    s.add_field("pop_vec", DataType.FLOAT_VECTOR, dim=2, nullable=vec_nullable)
    for v in variants:
        for name, dt, kw in v.fields:
            s.add_field(name, dt, **kw)
        for fn in v.functions:
            s.add_function(fn)
    return s


def build_index_params(client, variants):
    ip = client.prepare_index_params()
    ip.add_index(field_name="popularity", index_type="STL_SORT", index_name="popularity")
    ip.add_index(field_name="pop_vec", index_type="AUTOINDEX", metric_type="IP", index_name="pop_vec")
    for v in variants:
        for ix in v.indexes:
            kw = dict(ix)
            kw.setdefault("index_name", kw["field_name"])
            ip.add_index(**kw)
    return ip


def build_rows(base_rows, variants, vec_mode="null"):
    out = []
    for r in base_rows:
        row = dict(r)
        row["pop_vec"] = None if vec_mode == "null" else [math.log1p(r["popularity"]), 0.0]
        for v in variants:
            row.update(v.value(r))
        out.append(row)
    return out


def indexed_fields(variants):
    return ["popularity", "pop_vec"] + [ix["field_name"] for v in variants for ix in v.indexes]


# ------------------------------------------------------------------ execution
def call_kwargs(form, prefix, consistency):
    """(method_name, kwargs) for MilvusClient / AsyncMilvusClient, identical for both."""
    kw = form.build(prefix)
    if form.kind == "query":
        return "query", dict(filter=kw["filter"], limit=form.limit, order_by_fields=ORDER_BY,
                             output_fields=OUT_FIELDS, consistency_level=consistency)
    return "search", dict(data=kw["data"], anns_field=kw["anns_field"], limit=form.limit,
                          filter=kw.get("filter", ""), search_params=kw["search_params"],
                          output_fields=OUT_FIELDS, consistency_level=consistency)


def normalise_hits(form, method, res):
    if method == "query":
        hits = list(res)
    else:
        hits = [{"id": h["id"], "score": h["distance"], **h["entity"]} for h in res[0]]
    if form.client_sort:
        hits = sorted(hits, key=lambda h: -h["popularity"])[:TOP_K]
    return hits


def run_form(client, coll, form, prefix, consistency):
    method, kw = call_kwargs(form, prefix, consistency)
    return normalise_hits(form, method, getattr(client, method)(coll, **kw))


def expr_of(form, prefix):
    kw = form.build(prefix)
    return kw.get("filter") or f"search({kw['anns_field']}, {kw['data']!r})"
