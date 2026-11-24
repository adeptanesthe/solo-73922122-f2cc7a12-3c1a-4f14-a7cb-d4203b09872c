"""Self-tests: shared catalog across video ids, per-video restriction
isolation, hiding/recovery, idempotency/conflict, re-import semantics,
concurrent imports, storage accounting, <=1e-9 agreement with a stable full
recomputation, and O(log N) update timing."""

import math
import random
import threading
import time
from array import array

import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.main import ImportShotsRequest, Shot, app
from app.segment_tree import DIM, RestrictionIndex, ShotIndex


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


def _flat(features):
    flat = array("d", bytes(8 * DIM * len(features)))
    for i, f in enumerate(features):
        flat[i * DIM:(i + 1) * DIM] = array("d", f)
    return flat


def _index(features, scores):
    return ShotIndex(_flat(features), array("d", scores))


# ----------------------------------------------------- index vs brute force

@pytest.mark.parametrize("seed", range(40))
def test_restriction_index_matches_stable_full_recompute(seed):
    rng = random.Random(seed)
    n = rng.choice([1, 2, 3, 5, 17, 64, 100, 333, 1000])
    features = [[rng.uniform(-1, 1) for _ in range(DIM)] for _ in range(n)]
    scores = [rng.uniform(-1000, 1000) for _ in range(n)]
    idx = RestrictionIndex(_index(features, scores))
    hidden = set()
    assert_close(idx.retrieval_vector(), brute_force_vector(features, scores, hidden))

    active = []  # order ids of live orders
    for k in range(30):
        if active and rng.random() < 0.4:
            oid = active.pop(rng.randrange(len(active)))
            s, e = idx.orders[oid]
            idx.revoke(oid)
        else:
            s = rng.randrange(0, n)
            e = rng.randrange(s + 1, n + 1)
            oid = f"o{k}"
            idx.register(oid, s, e)
            active.append(oid)
        hidden = set()
        for oid in active:
            s, e = idx.orders[oid]
            hidden.update(range(s, e))
        assert idx.visible_count() == n - len(hidden)
        assert_close(idx.retrieval_vector(), brute_force_vector(features, scores, hidden))


def test_index_extreme_scores_and_features():
    rng = random.Random(7)
    n = 500
    features = [[rng.choice([-1.0, 1.0]) for _ in range(DIM)] for _ in range(n)]
    scores = [rng.choice([-1000.0, 1000.0]) for _ in range(n)]
    idx = RestrictionIndex(_index(features, scores))
    active = []
    for k in range(20):
        s = rng.randrange(0, n)
        e = rng.randrange(s + 1, n + 1)
        idx.register(f"o{k}", s, e)
        active.append((s, e))
        hidden = set()
        for s2, e2 in active:
            hidden.update(range(s2, e2))
        assert_close(idx.retrieval_vector(), brute_force_vector(features, scores, hidden))
    for k in range(len(active)):
        idx.revoke(f"o{k}")
    assert idx.visible_count() == n
    assert_close(idx.retrieval_vector(), brute_force_vector(features, scores, set()))
    # overlay is fully released once no orders remain
    assert idx.covers == {} and idx.overrides == {}


def test_index_all_hidden_returns_none():
    idx = RestrictionIndex(_index([[0.5] * DIM, [-0.5] * DIM], [3.0, -3.0]))
    idx.register("ALL", 0, 2)
    assert idx.visible_count() == 0
    assert idx.retrieval_vector() is None
    idx.revoke("ALL")
    assert idx.visible_count() == 2
    assert idx.retrieval_vector() is not None


def test_index_large_n_accuracy():
    rng = random.Random(99)
    n = 100_000
    features = [[rng.uniform(-1, 1) for _ in range(DIM)] for _ in range(n)]
    scores = [rng.uniform(-1000, 1000) for _ in range(n)]
    idx = RestrictionIndex(_index(features, scores))
    hidden = set()
    for k in range(300):
        s = rng.randrange(0, n)
        e = min(n, s + rng.randrange(1, 5000))
        idx.register(f"o{k}", s, e)
        hidden.update(range(s, e))
    assert_close(idx.retrieval_vector(), brute_force_vector(features, scores, hidden))


# ------------------------------------------------------------------- API

@pytest.fixture(autouse=True)
def _clean_state():
    with main._LOCK:
        main._VIDEOS.clear()
        main._CATALOGS.clear()
        main._BUILDING.clear()
    yield
    with main._LOCK:
        main._VIDEOS.clear()
        main._CATALOGS.clear()
        main._BUILDING.clear()


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


# ------------------------------------------------- shared catalog semantics

def test_same_content_shares_one_catalog(client):
    shots = _shots(1000, seed=11)
    bodies = [client.post(f"/videos/chan-{i}/shots", json={"shots": shots}).json()
              for i in range(3)]
    # all three point at the same catalog with identical data
    assert len({b["catalog_id"] for b in bodies}) == 1
    assert {b["catalog_id"] for b in bodies} == {bodies[0]["catalog_id"]}
    assert [b["shared_videos"] for b in bodies] == [1, 2, 3]
    assert bodies[0]["vector"] == bodies[2]["vector"]

    st = client.get("/storage").json()
    assert st["catalog_count"] == 1
    assert st["video_count"] == 3
    assert len(st["catalogs"][0]["video_ids"]) == 3
    # exactly one copy of the index plus three tiny overlays
    assert st["retained_bytes_total"] == st["catalog_bytes_total"] + st["overlay_bytes_total"]
    assert st["naive_bytes_total"] >= 3 * st["catalog_bytes_total"]
    assert st["overlay_bytes_total"] < st["catalog_bytes_total"] // 10


def test_same_content_independent_orders(client):
    shots = _shots(20, seed=12)
    for vid in ("a", "b"):
        client.post(f"/videos/{vid}/shots", json={"shots": shots})
    ra = client.post("/videos/a/restrictions",
                     json={"order_id": "lic", "start": 2, "end": 10}).json()
    rb = client.post("/videos/b/restrictions",
                     json={"order_id": "lic", "start": 12, "end": 20}).json()
    assert ra["visible_count"] == 12 and rb["visible_count"] == 12
    assert ra["catalog_id"] == rb["catalog_id"]
    # revoking on b leaves a untouched; same order id is scoped per video
    client.delete("/videos/b/restrictions/lic")
    assert client.get("/videos/a/retrieval-vector").json()["visible_count"] == 12
    assert client.get("/videos/b/retrieval-vector").json()["visible_count"] == 20
    assert client.delete("/videos/a/restrictions/lic").json()["visible_count"] == 20


def test_different_content_gets_own_catalog(client):
    client.post("/videos/x/shots", json={"shots": _shots(5, seed=1)})
    client.post("/videos/y/shots", json={"shots": _shots(5, seed=2)})
    st = client.get("/storage").json()
    assert st["catalog_count"] == 2 and st["video_count"] == 2


def test_last_video_leaving_releases_catalog(client):
    shots_a = _shots(5, seed=21)
    shots_b = _shots(5, seed=22)
    client.post("/videos/a/shots", json={"shots": shots_a})
    client.post("/videos/b/shots", json={"shots": shots_a})
    client.post("/videos/b/restrictions", json={"order_id": "o", "start": 1, "end": 3})
    assert client.get("/storage").json()["catalog_count"] == 1

    # b switches content: it leaves the shared catalog, which survives for a
    client.post("/videos/b/shots", json={"shots": shots_b})
    st = client.get("/storage").json()
    assert st["catalog_count"] == 2
    assert client.get("/videos/a/retrieval-vector").json()["visible_count"] == 5

    # a re-imports different content too: the shared catalog is freed
    client.post("/videos/a/shots", json={"shots": shots_b})
    st = client.get("/storage").json()
    assert st["catalog_count"] == 1
    assert st["video_count"] == 2
    assert all(v in ("a", "b") for v in st["catalogs"][0]["video_ids"])


def test_reimport_replaces_shots_and_clears_orders(client):
    client.post("/videos/r/shots", json={"shots": _shots(6, seed=31)})
    client.post("/videos/r/restrictions", json={"order_id": "o", "start": 0, "end": 2})
    assert client.get("/videos/r/retrieval-vector").json()["visible_count"] == 4

    shots = _shots(9, seed=32)
    body = client.post("/videos/r/shots", json={"shots": shots}).json()
    assert body["shot_count"] == 9 and body["visible_count"] == 9
    assert body["status"] == "ok"
    ref = brute_force_vector([s["feature"] for s in shots],
                             [s["score"] for s in shots], set())
    assert max(abs(a - b) for a, b in zip(body["vector"], ref)) <= 1e-9
    # old orders are gone (404), so indices now refer to the new shot list
    assert client.delete("/videos/r/restrictions/o").status_code == 404
    assert client.post("/videos/r/restrictions",
                       json={"order_id": "o", "start": 0, "end": 3}).status_code == 200


def test_reimport_same_content_also_clears_orders(client):
    shots = _shots(6, seed=33)
    body = client.post("/videos/rs/shots", json={"shots": shots}).json()
    cid = body["catalog_id"]
    client.post("/videos/rs/restrictions", json={"order_id": "o", "start": 0, "end": 2})
    body = client.post("/videos/rs/shots", json={"shots": shots}).json()
    assert body["catalog_id"] == cid  # still the very same shared index
    assert body["visible_count"] == 6 and body["status"] == "ok"
    # no second catalog was built, but this video's orders were reset
    assert client.get("/storage").json()["catalog_count"] == 1
    assert client.delete("/videos/rs/restrictions/o").status_code == 404


def test_failed_reimport_keeps_old_index_and_orders(client):
    client.post("/videos/f/shots", json={"shots": _shots(6, seed=41)})
    client.post("/videos/f/restrictions", json={"order_id": "keep", "start": 1, "end": 3})
    before = client.get("/videos/f/retrieval-vector").json()

    bad = {"feature": [0.0] * 15, "score": 0.0}
    assert client.post("/videos/f/shots", json={"shots": [bad]}).status_code == 422

    after = client.get("/videos/f/retrieval-vector").json()
    assert after["shot_count"] == 6 and after["visible_count"] == 4
    assert after["vector"] == before["vector"]
    assert client.delete("/videos/f/restrictions/keep").status_code == 200


def test_failed_reimport_build_keeps_old_index(client, monkeypatch):
    client.post("/videos/g/shots", json={"shots": _shots(6, seed=42)})
    client.post("/videos/g/restrictions", json={"order_id": "keep", "start": 1, "end": 3})
    before = client.get("/videos/g/retrieval-vector").json()

    def boom(*a, **kw):
        raise MemoryError("simulated build failure")

    monkeypatch.setattr(main, "ShotIndex", boom)
    with TestClient(app, raise_server_exceptions=False) as raw:
        assert raw.post("/videos/g/shots",
                        json={"shots": _shots(7, seed=43)}).status_code == 500

    after = client.get("/videos/g/retrieval-vector").json()
    assert after["shot_count"] == 6 and after["visible_count"] == 4
    assert after["vector"] == before["vector"]
    assert client.delete("/videos/g/restrictions/keep").status_code == 200


def test_queries_and_orders_continue_during_reimport_build(monkeypatch):
    """The tree build runs outside the lock: while a re-import is building,
    the old index stays queryable and orders can still be changed."""
    shots_old = _shots(10, seed=51)
    shots_new = _shots(12, seed=52)
    main.import_shots("h", ImportShotsRequest(shots=[Shot(**s) for s in shots_old]))
    main.register_restriction("h", main.RestrictionRequest(
        order_id="old", start=0, end=2))

    original = main.ShotIndex
    entered = threading.Event()
    release = threading.Event()

    def slow_index(features, scores):
        if len(scores) == len(shots_new):
            entered.set()
            assert release.wait(10)
        return original(features, scores)

    monkeypatch.setattr(main, "ShotIndex", slow_index)
    payload = ImportShotsRequest(shots=[Shot(**s) for s in shots_new])
    thread = threading.Thread(target=main.import_shots, args=("h", payload))
    thread.start()
    assert entered.wait(10)

    # while the build is in flight, the old shots and orders are still live
    t0 = time.perf_counter()
    mid = main.get_retrieval_vector("h")
    assert mid.shot_count == 10 and mid.visible_count == 8
    main.register_restriction("h", main.RestrictionRequest(order_id="mid", start=8, end=10))
    mid = main.get_retrieval_vector("h")
    assert mid.shot_count == 10 and mid.visible_count == 6
    assert time.perf_counter() - t0 < 1.0

    release.set()
    thread.join(10)
    after = main.get_retrieval_vector("h")
    assert after.shot_count == 12 and after.visible_count == 12  # orders cleared
    with pytest.raises(Exception):
        main.revoke_restriction("h", "old")


def test_concurrent_identical_imports_share_one_catalog():
    shots = [Shot(**s) for s in _shots(200, seed=61)]
    payload = ImportShotsRequest(shots=shots)
    errors = []

    def worker(i):
        try:
            main.import_shots(f"c{i}", payload)
        except BaseException as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    assert not errors
    st = main.storage()
    assert st.catalog_count == 1 and st.video_count == 8
    vecs = {tuple(main.get_retrieval_vector(f"c{i}").vector) for i in range(8)}
    assert len(vecs) == 1


def test_storage_bounded_by_n_plus_v_plus_r_log_n(client):
    shots = _shots(1000, seed=71)
    for i in range(3):
        client.post(f"/videos/s{i}/shots", json={"shots": shots})
    for i in range(3):
        for j in range(10):
            client.post(f"/videos/s{i}/restrictions",
                        json={"order_id": f"o{j}", "start": j * 10, "end": j * 10 + 25})
    st = client.get("/storage").json()
    # one shared index no matter how many video ids use it
    assert st["catalog_count"] == 1 and st["video_count"] == 3
    assert st["catalog_bytes_total"] == st["catalogs"][0]["index_bytes"]
    # per-video overlays stay small even with 30 live orders each
    assert st["saved_bytes"] > 0
    assert st["retained_bytes_total"] < st["naive_bytes_total"]


# ------------------------------------------------------------- performance

def test_update_is_log_n():
    """Register/revoke must stay O(log N): per-op cost at N=100k should be
    roughly the same as at N=10k (factor << the 10x linear growth)."""
    timings = {}
    for n in (10_000, 100_000):
        rng = random.Random(5)
        features = [[rng.uniform(-1, 1) for _ in range(DIM)] for _ in range(n)]
        scores = [rng.uniform(-1000, 1000) for _ in range(n)]
        idx = RestrictionIndex(_index(features, scores))
        ops = 300
        ranges = []
        for _ in range(ops):
            s = rng.randrange(0, n)
            e = min(n, s + rng.randrange(1, 2000))
            ranges.append((s, e))
        t0 = time.perf_counter()
        for k, (s, e) in enumerate(ranges):
            idx.register(f"o{k}", s, e)
        for k in range(ops):
            idx.revoke(f"o{k}")
        dt = time.perf_counter() - t0
        timings[n] = dt / (2 * ops)
        print(f"\nN={n}: {ops} register + {ops} revoke in {dt:.3f}s "
              f"-> {dt / (2 * ops) * 1e6:.1f} us/op")
    assert timings[100_000] < 5e-3  # generous absolute bound: 5 ms per op
    assert timings[100_000] < 6 * timings[10_000]  # far below 10x linear growth
