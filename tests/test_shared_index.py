"""Tests for the shared-index module: concurrent import coalescing, real
memory release on the last leave, and atomic visibility under re-import."""

import gc
import random
import threading
import weakref

import app.main as main
from app.main import ImportShotsRequest, Shot
from app.segment_tree import DIM


def _shots(n, seed):
    rng = random.Random(seed)
    return [{"feature": [rng.uniform(-1, 1) for _ in range(DIM)],
             "score": rng.uniform(-1000, 1000)} for _ in range(n)]


def _register(vid, shots):
    main.import_shots(vid, ImportShotsRequest(shots=[Shot(**s) for s in shots]))


def _clean():
    with main._LOCK:
        main._VIDEOS.clear()
        main._CATALOGS.clear()
        main._BUILDING.clear()


def test_concurrent_identical_imports_build_only_one_index(monkeypatch):
    """8 threads importing the same content: one build, everyone shares it."""
    _clean()
    try:
        shots = _shots(500, 5)
        payload = ImportShotsRequest(shots=[Shot(**s) for s in shots])
        original = main.ShotIndex
        builds = []
        build_lock = threading.Lock()

        def counting_index(features, scores):
            with build_lock:
                builds.append(1)
            return original(features, scores)

        monkeypatch.setattr(main, "ShotIndex", counting_index)
        errors = []

        def worker(i):
            try:
                main.import_shots(f"v{i}", payload)
            except BaseException as exc:  # pragma: no cover
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(60)
        assert not errors
        assert len(builds) == 1, f"expected a single build, got {len(builds)}"
        st = main.storage()
        assert st.catalog_count == 1 and st.video_count == 8
        vecs = {tuple(main.get_retrieval_vector(f"v{i}").vector) for i in range(8)}
        assert len(vecs) == 1
    finally:
        _clean()


def test_waiters_do_not_stay_stuck_if_builder_fails(monkeypatch):
    """If the builder's tree construction raises, waiters still import."""
    _clean()
    try:
        shots = _shots(50, 6)
        payload = ImportShotsRequest(shots=[Shot(**s) for s in shots])
        original = main.ShotIndex
        calls = {"n": 0}
        entered = threading.Event()
        release = threading.Event()

        def flaky_index(features, scores):
            calls["n"] += 1
            if calls["n"] == 1:
                entered.set()
                assert release.wait(10)
                raise RuntimeError("first build fails")
            return original(features, scores)

        monkeypatch.setattr(main, "ShotIndex", flaky_index)
        results = {}

        def first():
            try:
                main.import_shots("bad", payload)
            except RuntimeError as exc:
                results["first"] = exc

        def second():
            main.import_shots("good", payload)
            results["second"] = "ok"

        t1 = threading.Thread(target=first)
        t2 = threading.Thread(target=second)
        t1.start()
        assert entered.wait(10)
        t2.start()
        release.set()
        t1.join(20)
        t2.join(20)
        assert isinstance(results.get("first"), RuntimeError)
        assert results.get("second") == "ok"
        assert main.get_retrieval_vector("good").shot_count == 50
        assert main.storage().catalog_count == 1
        assert main._BUILDING == {}
    finally:
        _clean()


def test_last_video_leaving_really_frees_index_memory():
    _clean()
    try:
        a = _shots(1000, 1)
        b = _shots(1000, 2)
        _register("a", a)
        _register("b", a)
        _register("c", b)

        ref_a = weakref.ref(main._VIDEOS["a"].base)
        ref_b = weakref.ref(main._VIDEOS["c"].base)
        assert len({s.catalog_id for s in main._VIDEOS.values()}) == 2

        # a leaves its catalog, which b still uses -> stays resident
        _register("a", b)
        gc.collect()
        assert ref_a() is not None
        assert main._VIDEOS["b"].base is ref_a()
        assert main.storage().catalog_count == 2

        # b leaves too -> no video references the first index any more
        _register("b", b)
        gc.collect()
        assert ref_a() is None, "old index was not released"
        assert ref_b() is main._VIDEOS["a"].base
        st = main.storage()
        assert st.catalog_count == 1 and st.video_count == 3
    finally:
        _clean()


def test_reimport_under_load_is_atomic():
    """A reader polling vectors while a video is re-imported must only ever
    observe a complete old or complete new shot list, never a mix."""
    _clean()
    try:
        old = _shots(2000, 11)
        new = _shots(3000, 12)
        _register("hot", old)
        main.register_restriction("hot", main.RestrictionRequest(
            order_id="x", start=0, end=100))
        stop = False
        bad = []

        def reader():
            while not stop:
                r = main.get_retrieval_vector("hot")
                # old list + its order, old list just re-imported (order
                # cleared), or the new list
                if (r.shot_count, r.visible_count) not in (
                    (2000, 1900), (2000, 2000), (3000, 3000)
                ):
                    bad.append((r.shot_count, r.visible_count))

        t = threading.Thread(target=reader)
        t.start()
        for _ in range(5):
            _register("hot", new)
            _register("hot", old)
            main.register_restriction("hot", main.RestrictionRequest(
                order_id="x", start=0, end=100))
        stop = True
        t.join(10)
        assert not bad, bad[:5]
    finally:
        _clean()
