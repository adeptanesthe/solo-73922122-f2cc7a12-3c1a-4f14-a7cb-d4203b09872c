"""Self-tests: hiding/recovery correctness, overlap semantics, idempotency,
conflict handling, withdrawn marker, <=1e-9 agreement with a stable full
recomputation, and O(log N) update timing."""

import math
import random
import time

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.segment_tree import DIM, VisibleSoftmaxTree


# ------------------------------------------------------------- reference

def brute_force_vector(features, scores, hidden):
    """Stable full recomputation: max-shift + math.fsum (exact-ish sums)."""
    vis = [i for i in range(len(scores)) if i not in hidden]
    if not vis:
        return None
    m = max(scores[i] for i in vis)
    ws = [(i, math.exp(scores[i] - m)) for i in vis]
    s = math.fsum(w for _, w in ws)
    return [math.fsum(w * features[i][k] for i, w in ws) / s for k in range(DIM)]


def assert_close(tree_vec, ref_vec, tol=1e-9):
    if ref_vec is None:
        assert tree_vec is None
        return
    assert tree_vec is not None
    diffs = [abs(a - b) for a, b in zip(tree_vec, ref_vec)]
    assert max(diffs) <= tol, f"max dim diff {max(diffs):.3e} > {tol}"


# ---------------------------------------------------- tree vs brute force

@pytest.mark.parametrize("seed", range(40))
def test_tree_matches_stable_full_recompute(seed):
    rng = random.Random(seed)
    n = rng.choice([1, 2, 3, 5, 17, 64, 100, 333, 1000])
    features = [[rng.uniform(-1, 1) for _ in range(DIM)] for _ in range(n)]
    scores = [rng.uniform(-1000, 1000) for _ in range(n)]
    tree = VisibleSoftmaxTree(features, scores)
    hidden = set()
    assert_close(tree.retrieval_vector(), brute_force_vector(features, scores, hidden))

    active = []  # (start, end) of live orders
    for _ in range(30):
        if active and rng.random() < 0.4:
            i = rng.randrange(len(active))
            s, e = active.pop(i)
            tree.add_cover(s, e, -1)
            for j in range(s, e):
                hidden.discard(j)  # rebuilt below anyway
        else:
            s = rng.randrange(0, n)
            e = rng.randrange(s + 1, n + 1)
            tree.add_cover(s, e, +1)
            active.append((s, e))
        # rebuild the hidden set from scratch (reference counting semantics)
        hidden = set()
        for s, e in active:
            hidden.update(range(s, e))
        assert tree.visible_count() == n - len(hidden)
        assert_close(tree.retrieval_vector(), brute_force_vector(features, scores, hidden))


def test_tree_extreme_scores_and_features():
    rng = random.Random(7)
    n = 500
    features = [[rng.choice([-1.0, 1.0]) for _ in range(DIM)] for _ in range(n)]
    scores = [rng.choice([-1000.0, 1000.0]) for _ in range(n)]
    tree = VisibleSoftmaxTree(features, scores)
    hidden = set()
    active = []
    for _ in range(20):
        s = rng.randrange(0, n)
        e = rng.randrange(s + 1, n + 1)
        tree.add_cover(s, e, +1)
        active.append((s, e))
        hidden.update(range(s, e))
        assert_close(tree.retrieval_vector(), brute_force_vector(features, scores, hidden))
    # revoke everything -> must recover the full-set vector exactly-ish
    for s, e in active:
        tree.add_cover(s, e, -1)
    assert tree.visible_count() == n
    assert_close(tree.retrieval_vector(), brute_force_vector(features, scores, set()))


def test_tree_all_hidden_returns_none():
    tree = VisibleSoftmaxTree([[0.5] * DIM, [-0.5] * DIM], [3.0, -3.0])
    tree.add_cover(0, 2, +1)
    assert tree.visible_count() == 0
    assert tree.retrieval_vector() is None
    tree.add_cover(0, 2, -1)
    assert tree.visible_count() == 2
    assert tree.retrieval_vector() is not None


def test_tree_large_n_accuracy():
    rng = random.Random(99)
    n = 100_000
    features = [[rng.uniform(-1, 1) for _ in range(DIM)] for _ in range(n)]
    scores = [rng.uniform(-1000, 1000) for _ in range(n)]
    tree = VisibleSoftmaxTree(features, scores)
    hidden = set()
    for _ in range(300):
        s = rng.randrange(0, n)
        e = min(n, s + rng.randrange(1, 5000))
        tree.add_cover(s, e, +1)
        hidden.update(range(s, e))
    assert_close(tree.retrieval_vector(), brute_force_vector(features, scores, hidden))


# ------------------------------------------------------------------- API

@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


def _shots(n, seed=0):
    rng = random.Random(seed)
    return [
        {"feature": [rng.uniform(-1, 1) for _ in range(DIM)],
         "score": rng.uniform(-1000, 1000)}
        for _ in range(n)
    ]


def test_api_hide_overlap_revoke_recover(client):
    shots = _shots(8, seed=1)
    r = client.post("/videos/v1/shots", json={"shots": shots})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["visible_count"] == 8
    initial = body["vector"]

    feats = [s["feature"] for s in shots]
    scores = [s["score"] for s in shots]

    # order A hides [2,6), order B hides [4,8)  (overlap [4,6))
    assert client.post("/videos/v1/restrictions",
                       json={"order_id": "A", "start": 2, "end": 6}).json()["visible_count"] == 4
    assert client.post("/videos/v1/restrictions",
                       json={"order_id": "B", "start": 4, "end": 8}).json()["visible_count"] == 2

    # duplicate registration of A: no state change
    dup = client.post("/videos/v1/restrictions", json={"order_id": "A", "start": 2, "end": 6})
    assert dup.status_code == 200 and dup.json()["visible_count"] == 2

    # same id, different range: conflict
    conflict = client.post("/videos/v1/restrictions", json={"order_id": "A", "start": 2, "end": 7})
    assert conflict.status_code == 409

    # revoke A: [2,4) playable again, [4,8) still hidden by B
    r = client.delete("/videos/v1/restrictions/A")
    body = r.json()
    assert body["visible_count"] == 4
    ref = brute_force_vector(feats, scores, set(range(4, 8)))
    assert max(abs(a - b) for a, b in zip(body["vector"], ref)) <= 1e-9

    # revoke B: fully recovered, vector identical to the import-time one
    r = client.delete("/videos/v1/restrictions/B")
    body = r.json()
    assert body["visible_count"] == 8
    assert max(abs(a - b) for a, b in zip(body["vector"], initial)) <= 1e-9

    # unknown order / unknown video
    assert client.delete("/videos/v1/restrictions/B").status_code == 404
    assert client.get("/videos/nope/retrieval-vector").status_code == 404


def test_api_withdrawn_marker(client):
    client.post("/videos/v2/shots", json={"shots": _shots(3, seed=2)})
    r = client.post("/videos/v2/restrictions", json={"order_id": "ALL", "start": 0, "end": 3})
    body = r.json()
    assert body["status"] == "withdrawn" and body["vector"] is None and body["visible_count"] == 0
    r = client.delete("/videos/v2/restrictions/ALL")
    body = r.json()
    assert body["status"] == "ok" and body["vector"] is not None


def test_api_validation(client):
    client.post("/videos/v3/shots", json={"shots": _shots(4, seed=3)})
    bad_dim = {"feature": [0.0] * 15, "score": 0.0}
    assert client.post("/videos/v4/shots", json={"shots": [bad_dim]}).status_code == 422
    bad_val = {"feature": [1.5] * DIM, "score": 0.0}
    assert client.post("/videos/v4/shots", json={"shots": [bad_val]}).status_code == 422
    bad_score = {"feature": [0.0] * DIM, "score": 1000.1}
    assert client.post("/videos/v4/shots", json={"shots": [bad_score]}).status_code == 422
    assert client.post("/videos/v3/restrictions",
                       json={"order_id": "X", "start": 3, "end": 3}).status_code == 400
    assert client.post("/videos/v3/restrictions",
                       json={"order_id": "X", "start": 0, "end": 5}).status_code == 400
    assert client.post("/videos/v3/shots", json={"shots": []}).status_code == 422


# ------------------------------------------------------------- performance

def test_update_is_log_n():
    """Register/revoke must stay O(log N): per-op cost at N=100k should be
    roughly the same as at N=10k (factor << the 10x linear growth)."""
    timings = {}
    for n in (10_000, 100_000):
        rng = random.Random(5)
        features = [[rng.uniform(-1, 1) for _ in range(DIM)] for _ in range(n)]
        scores = [rng.uniform(-1000, 1000) for _ in range(n)]
        tree = VisibleSoftmaxTree(features, scores)
        ops = 300
        ranges = []
        for _ in range(ops):
            s = rng.randrange(0, n)
            e = min(n, s + rng.randrange(1, 2000))
            ranges.append((s, e))
        t0 = time.perf_counter()
        for s, e in ranges:
            tree.add_cover(s, e, +1)
        for s, e in ranges:
            tree.add_cover(s, e, -1)
        dt = time.perf_counter() - t0
        timings[n] = dt / (2 * ops)
        print(f"\nN={n}: {ops} register + {ops} revoke in {dt:.3f}s "
              f"-> {dt / (2 * ops) * 1e6:.1f} us/op")
    assert timings[100_000] < 5e-3  # generous absolute bound: 5 ms per op
    assert timings[100_000] < 6 * timings[10_000]  # far below 10x linear growth
