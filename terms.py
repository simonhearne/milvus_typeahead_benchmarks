"""Probe terms, normalisation, edge-ngram expansion and the 10k term generator."""
import csv
import math
import os
import random
import re
import unicodedata

from config import EDGE_MAX, EDGE_MIN

PROBE_TERMS = [
    "iPhone 15 Pro Max",
    "wireless headphones",
    "Sony WH-1000XM5",
    "men's running shoes",
    "café table",
    "USB-C charger 65W",
    "LEGO Star Wars",
    "4K TV 55 inch",
    "耳机 蓝牙",
    "Crème brûlée torch",
]


def normalize(s):
    s = unicodedata.normalize("NFKC", s).lower().strip()
    return re.sub(r"\s+", " ", s)


def is_cjk(s):
    return any("一" <= ch <= "鿿" for ch in s)


def edge_ngrams(norm):
    """Client-side edge-ngrams of every whitespace word, deduplicated, order kept."""
    out = []
    for w in norm.split(" "):
        for n in range(EDGE_MIN, min(EDGE_MAX, len(w)) + 1):
            g = w[:n]
            if g not in out:
                out.append(g)
    return out


def rows_from_terms(pairs):
    """pairs: iterable of (term, popularity) -> list of base row dicts."""
    rows = []
    for i, (term, pop) in enumerate(pairs, start=1):
        rows.append({"id": i, "term": term, "term_norm": normalize(term), "popularity": int(pop)})
    return rows


def probe_rows():
    # Distinct, rank-ordered popularity so ORDER BY results are deterministic.
    return rows_from_terms((t, 1000 // (i + 1)) for i, t in enumerate(PROBE_TERMS))


def probe_prefixes(rows):
    """Yield (case, row_id, query_prefix, expect_fn) per the phase-1 spec.

    expect_fn(term_norm) -> bool decides expected membership for each stored row.
    """
    out = []
    for r in rows:
        term, norm = r["term"], r["term_norm"]
        words = norm.split(" ")
        p1 = norm[:3]
        out.append(("string_prefix", r["id"], p1, _starts(p1)))
        if len(words) > 1:
            p2 = words[1][:3]
            out.append(("word_start", r["id"], p2, _word_starts(p2)))
            # completed first word + partial second word
            pm = words[0] + " " + words[1][:2]
            out.append(("multi_word", r["id"], pm, _starts(pm)))
        p3 = unicodedata.normalize("NFKC", term).strip()[:3]
        if p3 != p1:
            out.append(("case_orig", r["id"], p3, _starts(normalize(p3))))
        if len(p1) == 3 and not is_cjk(p1):
            typo = p1[0] + p1[2] + p1[1]
            if typo != p1:
                # strict = spec definition (rows whose term_norm starts with the typo'd string);
                # tolerant recall (does the intended row come back) is computed in phase 1.
                out.append(("typo", r["id"], typo, _starts(typo)))
        if is_cjk(norm):
            for n in (1, 2):
                out.append(("cjk", r["id"], norm[:n], _starts(norm[:n])))
    return out


def _starts(p):
    return lambda norm: norm.startswith(p)


def _word_starts(p):
    return lambda norm: any(w.startswith(p) for w in norm.split(" "))


# ---------------------------------------------------------------- 10k generator
BRANDS = ["Apple", "Samsung", "Sony", "LG", "Bose", "Nike", "Adidas", "Puma", "Lego", "Philips",
          "Dyson", "Bosch", "Canon", "Nikon", "Logitech", "Anker", "Xiaomi", "Lenovo", "Dell", "HP",
          "Asus", "Garmin", "Fitbit", "JBL", "Sennheiser", "Nespresso", "DeLonghi", "Tefal", "Braun",
          "Oral-B", "Levi's", "Under Armour", "New Balance", "Asics", "Ikea", "Hasbro", "Mattel",
          "Nintendo", "PlayStation", "Xbox", "GoPro", "DJI", "Kindle", "Kärcher", "Miele", "Siemens",
          "Crocs", "Birkenstock", "Ray-Ban", "Casio"]
PRODUCTS = ["headphones", "earbuds", "speaker", "smartwatch", "laptop", "tablet", "phone case",
            "charger", "usb-c cable", "monitor", "keyboard", "mouse", "running shoes", "sneakers",
            "hoodie", "t-shirt", "jeans", "backpack", "vacuum cleaner", "coffee machine",
            "air fryer", "toaster", "kettle", "blender", "electric toothbrush", "hair dryer",
            "camera", "lens", "drone", "tripod", "tv", "soundbar", "router", "power bank",
            "fitness tracker", "gaming chair", "controller", "desk lamp", "sunglasses", "watch"]
ATTRS = ["black", "white", "red", "blue", "green", "pink", "grey", "wireless", "bluetooth",
         "waterproof", "noise cancelling", "4k", "portable", "mini", "pro", "max", "ultra",
         "for kids", "for men", "for women", "2024", "2025", "size 42", "size 10", "xl", "large",
         "small", "65w", "20000mah", "1tb", "256gb", "55 inch", "65 inch", "stainless steel"]
LONG_TAIL = ["best {p} under 100", "{b} {p} replacement parts", "cheap {p} with free delivery",
             "{p} gift for dad", "how to clean {b} {p}", "{b} {p} vs {b2} {p}",
             "refurbished {b} {p}", "{p} deals today"]
ACCENTED = ["café", "crème", "brûlée", "naïve", "façade", "résumé", "jalapeño", "piñata",
            "über", "smørrebrød", "crêpe pan", "pâtisserie", "entrée", "fiancée", "déjà vu"]
CJK = ["耳机", "蓝牙", "手机壳", "充电器", "运动鞋", "咖啡机", "笔记本电脑", "智能手表", "无线", "音箱"]


def generate_terms(n=10_000, seed=42):
    rng = random.Random(seed)
    seen, out = set(), []

    def add(t):
        t = re.sub(r"\s+", " ", t).strip()
        k = normalize(t)
        if k and k not in seen:
            seen.add(k)
            out.append(t)

    n_accent, n_cjk = int(n * 0.02), int(n * 0.01)
    # The plain CJK / accented pools hold ~820 / ~1.9k unique terms. Past a run of duplicate
    # draws, widen the template so large n terminates; small n (<= 10k) never reaches it, so
    # its output is unchanged.
    stale = 0
    while len(out) < n_cjk:
        before = len(out)
        if stale < 500:
            add(" ".join(rng.sample(CJK, rng.randint(1, 3))))
        else:
            add(f"{' '.join(rng.sample(CJK, rng.randint(1, 2)))} {rng.choice(BRANDS)} {rng.choice(ATTRS)}")
        stale = 0 if len(out) > before else stale + 1
    stale = 0
    while len(out) < n_cjk + n_accent:
        before = len(out)
        a = rng.choice(ACCENTED)
        if stale < 500:
            add(rng.choice([f"{a} {rng.choice(PRODUCTS)}", f"{rng.choice(BRANDS)} {a}",
                            f"{a} {rng.choice(ATTRS)}", a]))
        else:
            add(f"{rng.choice(BRANDS)} {a} {rng.choice(PRODUCTS)} {rng.choice(ATTRS)}")
        stale = 0 if len(out) > before else stale + 1
    while len(out) < n:
        b, p, a = rng.choice(BRANDS), rng.choice(PRODUCTS), rng.choice(ATTRS)
        r = rng.random()
        if r < 0.30:
            add(f"{b} {p}")
        elif r < 0.55:
            add(f"{b} {p} {a}")
        elif r < 0.70:
            add(f"{p} {a}")
        elif r < 0.82:
            model = f"{rng.choice('ABCDEFGHKMRSTWXZ')}{rng.choice(['', '-'])}{rng.randint(10, 9999)}"
            add(f"{b} {model}{rng.choice(['', ' ' + a])}")
        elif r < 0.90:
            add(f"{b} {p} {a} {rng.choice(ATTRS)}")
        else:
            add(rng.choice(LONG_TAIL).format(b=b, b2=rng.choice(BRANDS), p=p))
    rng.shuffle(out)
    # Zipf popularity (s≈1.1) over a random rank order.
    ranks = list(range(1, n + 1))
    rng.shuffle(ranks)
    return [(t, max(1, int(1_000_000 / math.pow(rk, 1.1)))) for t, rk in zip(out, ranks)]


def load_terms(n, seed=42, out_csv=None):
    path = os.environ.get("TERMS_CSV")
    if path:
        with open(path, newline="", encoding="utf-8") as f:
            pairs = [(r["term"], int(r["popularity"])) for r in csv.DictReader(f)]
    else:
        pairs = generate_terms(n, seed)
    if out_csv:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["term", "popularity"])
            w.writerows(pairs)
    return pairs


# ---------------------------------------------------------------- phase-3 workload
def sample_prefixes(rows, lengths=(3, 4, 5, 6), per_len=2000, seed=7):
    """Popularity-weighted prefixes. Returns {"string": {L: [...]}, "word": {L: [...]}}.

    string: first L chars of term_norm. word: first L chars of a random non-first word
    (terms with a single word fall back to that word).
    """
    rng = random.Random(seed)
    weights = [r["popularity"] for r in rows]
    out = {"string": {}, "word": {}}
    for L in lengths:
        s_list, w_list = [], []
        while len(s_list) < per_len or len(w_list) < per_len:
            r = rng.choices(rows, weights=weights, k=1)[0]
            norm = r["term_norm"]
            if len(s_list) < per_len and len(norm) >= L:
                s_list.append(norm[:L])
            words = [w for w in norm.split(" ") if len(w) >= L]
            later = [w for w in words if w != norm.split(" ")[0]] or words
            if len(w_list) < per_len and later:
                w_list.append(rng.choice(later)[:L])
        out["string"][L], out["word"][L] = s_list, w_list
    return out


def ground_truth(rows, prefixes, mode, k=10):
    """{prefix: {"n_match": int, "top": [ids by popularity desc], "cut": popularity of k-th match}}.

    A returned row counts as correct if it matches the prefix and its popularity >= cut, so ties
    at the top-k boundary are not scored as misses. One pass over rows, so it scales to 1M+.
    """
    wanted = set(prefixes)
    lens = sorted({len(p) for p in wanted})
    hits = {p: [] for p in wanted}
    for r in rows:
        norm = r["term_norm"]
        cands = {norm[:L] for L in lens} if mode == "string" else \
            {w[:L] for w in norm.split(" ") for L in lens}
        for c in cands & wanted:
            hits[c].append((r["popularity"], r["id"]))
    gt = {}
    for p, m in hits.items():
        m.sort(key=lambda t: (-t[0], t[1]))
        cut = m[min(k, len(m)) - 1][0] if m else 0
        gt[p] = {"n_match": len(m), "top": [i for _, i in m[:k]], "cut": cut}
    return gt


def matches(norm, p, mode):
    return norm.startswith(p) if mode == "string" else any(w.startswith(p) for w in norm.split(" "))


def score_topk(returned_ids, gt_entry, row_by_id, p, mode, k=10):
    """(overlap, exact_order): overlap = correct returned / min(k, n_match); 1.0 if both empty."""
    need = min(k, gt_entry["n_match"])
    if need == 0:
        return (1.0 if not returned_ids else 0.0), not returned_ids
    ok = sum(1 for i in returned_ids[:k]
             if i in row_by_id and matches(row_by_id[i]["term_norm"], p, mode)
             and row_by_id[i]["popularity"] >= gt_entry["cut"])
    return min(ok, need) / need, list(returned_ids[:k]) == gt_entry["top"]
