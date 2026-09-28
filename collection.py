"""Create / fill / index / load a benchmark collection with the build discipline the spec requires."""
import time

from variants import build_index_params, build_rows, build_schema, indexed_fields


class SetupError(Exception):
    def __init__(self, step, err):
        super().__init__(f"{step}: {err}")
        self.step, self.err = step, err


def _step(log, step, fn):
    t = time.perf_counter()
    try:
        out = fn()
    except Exception as e:  # noqa: BLE001 - recorded, then re-raised as SetupError
        log.append({"step": step, "ok": False, "error": f"{type(e).__name__}: {e}"})
        raise SetupError(step, e) from e
    log.append({"step": step, "ok": True, "s": round(time.perf_counter() - t, 2)})
    return out


def wait_indexes(c, coll, names, timeout=900):
    deadline = time.time() + timeout
    while True:
        states = {n: c.describe_index(coll, n) for n in names}
        if all(s.get("state") == "Finished" and not s.get("pending_index_rows") for s in states.values()):
            return states
        if any(s.get("state") == "Failed" for s in states.values()):
            bad = {n: s for n, s in states.items() if s.get("state") == "Failed"}
            raise RuntimeError(f"index build failed: {bad}")
        if time.time() > deadline:
            raise TimeoutError(f"index build not finished after {timeout}s: "
                               f"{ {n: s.get('state') for n, s in states.items()} }")
        time.sleep(2)


def setup_collection(c, name, variants, base_rows, vec_mode="null", vec_nullable=True, batch=2000):
    """Returns (log, applied_indexes). Raises SetupError with the failing step."""
    log = []
    if c.has_collection(name):
        c.drop_collection(name)
    schema = build_schema(c, variants, vec_nullable=vec_nullable)
    _step(log, "create_collection", lambda: c.create_collection(name, schema=schema))
    rows = build_rows(base_rows, variants, vec_mode=vec_mode)

    def insert():
        for i in range(0, len(rows), batch):
            c.insert(name, rows[i:i + batch])
    _step(log, "insert", insert)
    _step(log, "flush", lambda: c.flush(name))
    _step(log, "create_index", lambda: c.create_index(name, build_index_params(c, variants)))
    names = indexed_fields(variants)
    states = _step(log, "wait_index_build", lambda: wait_indexes(c, name, names))
    _step(log, "load", lambda: c.load_collection(name))
    applied = {n: {k: s.get(k) for k in ("field_name", "index_type", "metric_type", "params", "state",
                                           "indexed_rows", "total_rows", "pending_index_rows")
                   if k in s} | {"raw": s} for n, s in states.items()}
    return log, applied
