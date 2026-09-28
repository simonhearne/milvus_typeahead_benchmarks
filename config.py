"""Environment, naming and run bookkeeping shared by every phase."""
import datetime as dt
import json
import os
import platform
import socket
import time
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
load_dotenv(ROOT.parent / ".env")

ZILLIZ_URI = os.environ["ZILLIZ_URI"]
ZILLIZ_TOKEN = os.environ["ZILLIZ_TOKEN"]
ZILLIZ_DB = os.environ.get("ZILLIZ_DB", "default")

COLLECTION_PREFIX = "ta_bench_"
VARCHAR_MAX = 1024
EDGE_MIN, EDGE_MAX = 1, 15
NGRAM_MIN = int(os.environ.get("NGRAM_MIN", 2))
NGRAM_MAX = int(os.environ.get("NGRAM_MAX", 3))
PHASE3_QPS = 20


def client():
    from pymilvus import MilvusClient
    return MilvusClient(uri=ZILLIZ_URI, token=ZILLIZ_TOKEN, db_name=ZILLIZ_DB)


def stamp():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def run_dir(ts):
    d = ROOT / "results" / ts
    d.mkdir(parents=True, exist_ok=True)
    return d


def collection_name(tag, ts):
    return f"{COLLECTION_PREFIX}{tag}_{ts}"


def tcp_connect_ms(samples=5):
    u = urlparse(ZILLIZ_URI)
    port = u.port or 443
    out = []
    for _ in range(samples):
        t = time.perf_counter()
        with socket.create_connection((u.hostname, port), timeout=5):
            pass
        out.append((time.perf_counter() - t) * 1000)
    return sorted(out)[len(out) // 2]


def run_meta(c):
    import pymilvus
    host = urlparse(ZILLIZ_URI).hostname
    connect_ms = tcp_connect_ms()
    meta = {
        "server_version": c.get_server_version(),
        # get_server_version() on Zilliz Cloud only says "Compatible with Milvus 3.0"; the exact build
        # comes from the control plane and is passed in by the operator.
        "cluster_db_version": os.environ.get("CLUSTER_DB_VERSION"),
        "pymilvus_version": pymilvus.__version__,
        "endpoint_host": host,
        "endpoint_region": host.split(".")[1] if host.count(".") > 2 else None,
        "client_hostname": socket.gethostname(),
        "client_platform": platform.platform(),
        "tcp_connect_ms_median": round(connect_ms, 2),
        "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    if connect_ms > 10:
        meta["warning"] = (f"TCP connect {connect_ms:.1f} ms > 10 ms: client is not co-located "
                           "with the cluster; latency numbers include WAN RTT.")
    return meta


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str))
