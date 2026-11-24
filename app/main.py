"""Video retrieval-vector service with a content-addressed shared shot index.

The same film is released on several channels, each under its own video id,
with an identical shot list (order, 16-dim features, attention scores).  The
expensive per-film data -- leaf payload and the full segment tree of softmax
aggregates -- is stored once per distinct content and referenced by every
video id that imported it.  Each video id only owns a sparse restriction
overlay holding its own orders, so importing the same content under V ids
keeps O(N) index plus O(V + R log N) order state resident, not O(V*N).

Import up to 100k time-ordered shots per video, hide shot ranges by order id,
and always get back the retrieval vector computed over the currently playable
shots:

    softmax(attention) weighted sum of features.

Ranges are 0-based, left-closed right-open [start, end).  Orders may
overlap; a shot stays hidden until every order covering it is revoked.
Re-registering the same order id with the same range is a no-op; the same
id with a different range is rejected with 409.  When every shot is hidden
the response carries status="withdrawn" (vector=null) as the
index-withdrawal marker.

Imports are content-addressed: an import whose payload byte-for-byte matches
a catalog already in memory attaches to it instead of building a second tree.
Concurrent imports of the same new content are coalesced onto one build, so
only a single index copy is ever resident.  Building happens outside the lock,
so an in-flight re-import does not block or disturb queries/order changes on
the previous index; the swap is atomic and only after a successful build.
When the last video id leaves a catalog the catalog is dropped and its memory
released.
"""

import hashlib
import sys
import threading
from array import array

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from .segment_tree import DIM, RestrictionIndex, ShotIndex

MAX_SHOTS = 100_000
SCORE_MIN, SCORE_MAX = -1000.0, 1000.0
BUILD_WAIT_S = 120.0  # max wait for a concurrent build of the same content

app = FastAPI(title="shot-retrieval-vector", version="2.0.0")


# ---------------------------------------------------------------- schemas

class Shot(BaseModel):
    feature: list[float] = Field(min_length=DIM, max_length=DIM)
    score: float = Field(ge=SCORE_MIN, le=SCORE_MAX)

    @field_validator("feature")
    @classmethod
    def _feature_range(cls, v):
        for x in v:
            if not (-1.0 <= x <= 1.0):
                raise ValueError("feature dimensions must be within [-1, 1]")
        return v


class ImportShotsRequest(BaseModel):
    shots: list[Shot] = Field(min_length=1, max_length=MAX_SHOTS)


class RestrictionRequest(BaseModel):
    order_id: str = Field(min_length=1, max_length=128)
    start: int = Field(ge=0)
    end: int = Field(ge=0)


class VectorResponse(BaseModel):
    video_id: str
    shot_count: int
    visible_count: int
    status: str  # "ok" | "withdrawn"
    vector: list[float] | None
    catalog_id: str
    shared_videos: int


class CatalogInfo(BaseModel):
    catalog_id: str
    shot_count: int
    index_bytes: int
    video_ids: list[str]


class StorageResponse(BaseModel):
    catalog_count: int
    video_count: int
    catalog_bytes_total: int
    overlay_bytes_total: int
    retained_bytes_total: int
    naive_bytes_total: int
    saved_bytes: int
    catalogs: list[CatalogInfo]


# ----------------------------------------------------------------- state

class _Catalog:
    __slots__ = ("base", "video_ids")

    def __init__(self, base):
        self.base = base
        self.video_ids = set()


class _VideoState:
    __slots__ = ("catalog_id", "base", "restrictions")

    def __init__(self, catalog_id, base):
        self.catalog_id = catalog_id
        self.base = base
        self.restrictions = RestrictionIndex(base)


_VIDEOS: dict[str, _VideoState] = {}
_CATALOGS: dict[str, _Catalog] = {}
_BUILDING: dict[str, threading.Event] = {}  # fingerprint -> in-flight build
_LOCK = threading.Lock()  # guards the registries; held only for O(1)/O(log N) work


def _get_state(video_id: str) -> _VideoState:
    st = _VIDEOS.get(video_id)
    if st is None:
        raise HTTPException(status_code=404, detail=f"unknown video_id: {video_id!r}")
    return st


def _response(video_id: str, st: _VideoState) -> VectorResponse:
    vec = st.restrictions.retrieval_vector()
    return VectorResponse(
        video_id=video_id,
        shot_count=st.base.n,
        visible_count=st.restrictions.visible_count(),
        status="ok" if vec is not None else "withdrawn",
        vector=vec,
        catalog_id=st.catalog_id,
        shared_videos=len(_CATALOGS[st.catalog_id].video_ids),
    )


def _payload_arrays(req: ImportShotsRequest):
    """Flatten validated shots into compact arrays + a content digest."""
    n = len(req.shots)
    features = array("d", bytes(8 * DIM * n))
    scores = array("d", bytes(8 * n))
    for i, shot in enumerate(req.shots):
        features[i * DIM:(i + 1) * DIM] = array("d", shot.feature)
        scores[i] = shot.score
    digest = hashlib.sha256()
    digest.update(scores.tobytes())
    digest.update(features.tobytes())
    return features, scores, digest.hexdigest()[:16]


def _attach(catalog_id: str, base, video_id: str):
    """Bind `video_id` to a catalog, replacing any previous binding.

    Caller holds _LOCK.  The old catalog is released in the same critical
    section, so concurrent re-imports of one video cannot leak a catalog.
    A re-import always starts a fresh restriction overlay, even when the
    content (and therefore the catalog) is unchanged.
    """
    old = _VIDEOS.get(video_id)
    if old is not None and old.catalog_id != catalog_id:
        _detach(old.catalog_id, video_id)
    cat = _CATALOGS.get(catalog_id)
    if cat is None:
        cat = _CATALOGS[catalog_id] = _Catalog(base)
    cat.video_ids.add(video_id)
    _VIDEOS[video_id] = _VideoState(catalog_id, cat.base)


def _detach(catalog_id: str, video_id: str):
    """Unbind `video_id`; free the catalog once its last video leaves."""
    cat = _CATALOGS.get(catalog_id)
    if cat is None:
        return
    cat.video_ids.discard(video_id)
    if not cat.video_ids:
        del _CATALOGS[catalog_id]


# -------------------------------------------------------------- endpoints

@app.get("/health")
def health():
    return {"status": "up"}


@app.post("/videos/{video_id}/shots", response_model=VectorResponse)
def import_shots(video_id: str, req: ImportShotsRequest):
    """(Re)import a video's shots.  Identical content reuses one shared
    catalog across video ids.  Re-importing replaces all shots and drops
    every existing restriction order of that video; on failure the old shots
    and orders stay in effect."""
    features, scores, catalog_id = _payload_arrays(req)  # validation is done
    own_event = None  # set only by the thread that registered the build slot

    # Fast path: an identical catalog already exists -> no tree build at all.
    with _LOCK:
        if catalog_id in _CATALOGS:
            _attach(catalog_id, _CATALOGS[catalog_id].base, video_id)
            return _response(video_id, _VIDEOS[video_id])
        # Concurrent imports of the same content: the first one builds, the
        # rest wait for it and then share the result instead of each building
        # their own O(N) tree (only one index is ever resident).
        pending = _BUILDING.get(catalog_id)
        if pending is None:
            own_event = _BUILDING[catalog_id] = threading.Event()

    if own_event is None:
        # Bounded wait: a lost builder thread must not wedge importers.
        pending.wait(BUILD_WAIT_S)
        with _LOCK:
            cat = _CATALOGS.get(catalog_id)
            if cat is not None:  # builder succeeded
                _attach(catalog_id, cat.base, video_id)
                return _response(video_id, _VIDEOS[video_id])
        # Builder failed (or never finished): build this import's own index.

    try:
        base = ShotIndex(features, scores)  # O(N), outside the lock
    except BaseException:
        if own_event is not None:
            with _LOCK:
                if _BUILDING.get(catalog_id) is own_event:
                    del _BUILDING[catalog_id]
            own_event.set()
        raise

    with _LOCK:
        cat = _CATALOGS.get(catalog_id)
        if cat is None:  # nobody inserted the same content meanwhile
            _attach(catalog_id, base, video_id)
        else:  # a concurrent import won the race: share its tree, drop ours
            _attach(catalog_id, cat.base, video_id)
        resp = _response(video_id, _VIDEOS[video_id])
        if own_event is not None and _BUILDING.get(catalog_id) is own_event:
            del _BUILDING[catalog_id]
    if own_event is not None:
        own_event.set()  # wake waiters only after the new state is visible
    return resp


@app.post("/videos/{video_id}/restrictions", response_model=VectorResponse)
def register_restriction(video_id: str, req: RestrictionRequest):
    """Hide shots [start, end) under `order_id`.  Idempotent per
    (order_id, start, end); same id with a different range -> 409."""
    with _LOCK:
        st = _get_state(video_id)
        if req.start >= req.end or req.end > st.base.n:
            raise HTTPException(
                status_code=400,
                detail=f"invalid range [{req.start}, {req.end}) for {st.base.n} shots",
            )
        existing = st.restrictions.orders.get(req.order_id)
        if existing is not None:
            if existing == (req.start, req.end):
                return _response(video_id, st)  # duplicate registration: no state change
            raise HTTPException(
                status_code=409,
                detail=(
                    f"order {req.order_id!r} already registered with range "
                    f"[{existing[0]}, {existing[1]}); revoke it first"
                ),
            )
        st.restrictions.register(req.order_id, req.start, req.end)
        return _response(video_id, st)


@app.delete("/videos/{video_id}/restrictions/{order_id}", response_model=VectorResponse)
def revoke_restriction(video_id: str, order_id: str):
    """Revoke one order.  Shots still covered by other orders stay hidden."""
    with _LOCK:
        st = _get_state(video_id)
        if order_id not in st.restrictions.orders:
            raise HTTPException(status_code=404, detail=f"unknown order_id: {order_id!r}")
        st.restrictions.revoke(order_id)
        return _response(video_id, st)


@app.get("/videos/{video_id}/retrieval-vector", response_model=VectorResponse)
def get_retrieval_vector(video_id: str):
    with _LOCK:
        return _response(video_id, _get_state(video_id))


@app.get("/storage", response_model=StorageResponse)
def storage():
    """Resident index storage: shared catalogs vs. the per-video-copy layout."""
    with _LOCK:
        catalogs = [
            CatalogInfo(
                catalog_id=cid,
                shot_count=cat.base.n,
                index_bytes=cat.base.byte_size(),
                video_ids=sorted(cat.video_ids),
            )
            for cid, cat in _CATALOGS.items()
        ]
        catalog_bytes = sum(c.index_bytes for c in catalogs)
        overlay_bytes = 0
        naive_bytes = 0
        for st in _VIDEOS.values():
            own = (
                st.restrictions.byte_size()
                + sys.getsizeof(st)
                + sys.getsizeof(st.restrictions)
            )
            overlay_bytes += own
            # pre-sharing layout: every video id owned a full index copy
            naive_bytes += st.base.byte_size() + own
    retained = catalog_bytes + overlay_bytes
    return StorageResponse(
        catalog_count=len(catalogs),
        video_count=len(_VIDEOS),
        catalog_bytes_total=catalog_bytes,
        overlay_bytes_total=overlay_bytes,
        retained_bytes_total=retained,
        naive_bytes_total=naive_bytes,
        saved_bytes=naive_bytes - retained,
        catalogs=catalogs,
    )
