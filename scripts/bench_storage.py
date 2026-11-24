"""Benchmark: shared-catalog storage and accuracy.

Measures, at N = 100,000 shots:
  * resident bytes of one shared ShotIndex (python-object based, `byte_size()`)
  * traced bytes of the same index (tracemalloc ground truth)
  * per-video overlay cost with R live orders
  * V-video retained vs. pre-sharing layout (one full index per video id)
  * retrieval-vector error vs. stable full recomputation (math.fsum)
  * register/revoke latency at N=10k and N=100k

Run:  python scripts/bench_storage.py
"""

import math
import statistics
import sys
import time
import tracemalloc
from array import array

sys.path.insert(0, __file__.rsplit("/scripts/", 1)[0])

from app.segment_tree import DIM, RestrictionIndex, ShotIndex  # noqa: E402


def make_payload(n, seed=0):
    """Deterministic LCG payload (no 100k Python float objects needed)."""
    feats = array("d", bytes(8 * DIM * n))
    scores = array("d", bytes(8 * n))
    state = seed
    for i in range(n):
        base = i * DIM
        for k in range(DIM):
            state = (1103515245 * state + 12345) & 0x7FFFFFFF
            feats[base + k] = (state / 0x3FFFFFFF) - 1.0
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        scores[i] = (state / 0x3FFFFFFF) * 1000.0 - 1000.0
    return feats, scores


def traced(fn):
    tracemalloc.start()
    try:
        obj = fn()
        cur, _ = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return obj, cur


def reference_vector(feats, scores, hidden):
    vis = [i for i in range(len(scores)) if i not in hidden]
    m = max(scores[i] for i in vis)
    ws = [(i, math.exp(scores[i] - m)) for i in vis]
    s = math.fsum(w for _, w in ws)
    return [math.fsum(w * feats[i * DIM + k] for i, w in ws) / s for k in range(DIM)]


def main():
    n = 100_000
    print(f"building payload N={n} ...")
    feats, scores = make_payload(n)

    base, base_traced = traced(lambda: ShotIndex(feats, scores))
    print("\n== one shared index (N=100,000) ==")
    print(f"byte_size() (deep, python objects) : {base.byte_size():>12,} B"
          f"  ({base.byte_size() / 2**20:.1f} MiB)")
    print(f"tracemalloc current                : {base_traced:>12,} B"
          f"  ({base_traced / 2**20:.1f} MiB)")

    # overlay cost with R live orders
    for r in (0, 40, 200):
        def build(r=r):
            idx = RestrictionIndex(base)
            for j in range(r):
                s = (j * 7919) % (n - 5000)
                idx.register(f"o{j}", s, s + 5000)
            return idx
        idx, ov = traced(build)
        print(f"overlay, R={r:>3} orders: byte_size()={idx.byte_size():>10,} B"
              f"  traced={ov:>9,} B  (covers={len(idx.covers)},"
              f" overrides={len(idx.overrides)})")

    # V videos sharing the same content
    print("\n== V video ids sharing this content, pre- vs post-sharing ==")
    for v in (1, 2, 4, 8):
        naive = v * base.byte_size()
        shared = base.byte_size() + v * (sys.getsizeof(RestrictionIndex(base)))
        print(f"V={v}: one-copy-per-video={naive:>13,} B   shared={shared:>13,} B"
              f"   ratio={naive / shared:5.2f}x")

    # accuracy across videos with different orders (worst dim diff vs fsum)
    print("\n== accuracy (max |tree - stable full recompute| over 16 dims) ==")
    rng_state = 12345
    def rnd():
        nonlocal rng_state
        rng_state = (1103515245 * rng_state + 12345) & 0x7FFFFFFF
        return rng_state / 0x7FFFFFFF

    idx = RestrictionIndex(base)
    hidden = set()
    for j in range(200):
        s = int(rnd() * (n - 20000))
        e = s + 1 + int(rnd() * 20000)
        idx.register(f"o{j}", s, e)
        hidden.update(range(s, e))
    got = idx.retrieval_vector()
    ref = reference_vector(feats, scores, hidden)
    worst = max(abs(a - b) for a, b in zip(got, ref))
    print(f"visible={idx.visible_count():,}  max dim diff = {worst:.3e}  "
          f"(threshold 1e-9: {'PASS' if worst <= 1e-9 else 'FAIL'})")

    # latency
    print("\n== register/revoke latency (300 + 300 ops, median us/op) ==")
    for m in (10_000, 100_000):
        f2, s2 = make_payload(m, seed=7)
        big = ShotIndex(f2, s2)
        ix = RestrictionIndex(big)
        lat = []
        for j in range(300):
            s = (j * 6151) % (m - 2000)
            t0 = time.perf_counter()
            ix.register(f"x{j}", s, s + 1500)
            lat.append(time.perf_counter() - t0)
        for j in range(300):
            t0 = time.perf_counter()
            ix.revoke(f"x{j}")
            lat.append(time.perf_counter() - t0)
        print(f"N={m:>7,}: {statistics.median(lat) * 1e6:6.1f} us/op median,"
              f" {statistics.mean(lat) * 1e6:6.1f} us/op mean")


if __name__ == "__main__":
    main()
