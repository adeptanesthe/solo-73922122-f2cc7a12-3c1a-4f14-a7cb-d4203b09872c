"""Video retrieval-vector service.

Import up to 100k time-ordered shots (16-dim feature in [-1,1], attention
score in [-1000,1000]) per video, hide shot ranges by order id, and always
get back the retrieval vector computed over the currently playable shots:

    softmax(attention) weighted sum of features.

Ranges are 0-based, left-closed right-open [start, end).  Orders may
overlap; a shot stays hidden until every order covering it is revoked.
Re-registering the same order id with the same range is a no-op; the same
id with a different range is rejected with 409.  When every shot is hidden
the response carries status="withdrawn" (vector=null) as the
index-withdrawal marker.
"""

import threading

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from .segment_tree import DIM, VisibleSoftmaxTree

MAX_SHOTS = 100_000
SCORE_MIN, SCORE_MAX = -1000.0, 1000.0

app = FastAPI(title="shot-retrieval-vector", version="1.0.0")


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


# ----------------------------------------------------------------- state

class _VideoState:
    __slots__ = ("tree", "orders")

    def __init__(self, tree):
        self.tree = tree
        self.orders = {}  # order_id -> (start, end)


_VIDEOS: dict[str, _VideoState] = {}
_LOCK = threading.Lock()  # mutations are O(log N); held only for them


def _get_state(video_id: str) -> _VideoState:
    st = _VIDEOS.get(video_id)
    if st is None:
        raise HTTPException(status_code=404, detail=f"unknown video_id: {video_id!r}")
    return st


def _response(video_id: str, st: _VideoState) -> VectorResponse:
    vec = st.tree.retrieval_vector()
    return VectorResponse(
        video_id=video_id,
        shot_count=st.tree.n,
        visible_count=st.tree.visible_count(),
        status="ok" if vec is not None else "withdrawn",
        vector=vec,
    )


# -------------------------------------------------------------- endpoints

@app.get("/health")
def health():
    return {"status": "up"}


@app.post("/videos/{video_id}/shots", response_model=VectorResponse)
def import_shots(video_id: str, req: ImportShotsRequest):
    """(Re)import a video's shots.  Re-importing replaces all shots and
    drops every existing restriction order of that video."""
    features = [s.feature for s in req.shots]
    scores = [s.score for s in req.shots]
    tree = VisibleSoftmaxTree(features, scores)  # O(N), built outside the lock
    with _LOCK:
        _VIDEOS[video_id] = _VideoState(tree)
        return _response(video_id, _VIDEOS[video_id])


@app.post("/videos/{video_id}/restrictions", response_model=VectorResponse)
def register_restriction(video_id: str, req: RestrictionRequest):
    """Hide shots [start, end) under `order_id`.  Idempotent per
    (order_id, start, end); same id with a different range -> 409."""
    with _LOCK:
        st = _get_state(video_id)
        if req.start >= req.end or req.end > st.tree.n:
            raise HTTPException(
                status_code=400,
                detail=f"invalid range [{req.start}, {req.end}) for {st.tree.n} shots",
            )
        existing = st.orders.get(req.order_id)
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
        st.tree.add_cover(req.start, req.end, +1)
        st.orders[req.order_id] = (req.start, req.end)
        return _response(video_id, st)


@app.delete("/videos/{video_id}/restrictions/{order_id}", response_model=VectorResponse)
def revoke_restriction(video_id: str, order_id: str):
    """Revoke one order.  Shots still covered by other orders stay hidden."""
    with _LOCK:
        st = _get_state(video_id)
        rng = st.orders.pop(order_id, None)
        if rng is None:
            raise HTTPException(status_code=404, detail=f"unknown order_id: {order_id!r}")
        st.tree.add_cover(rng[0], rng[1], -1)
        return _response(video_id, st)


@app.get("/videos/{video_id}/retrieval-vector", response_model=VectorResponse)
def get_retrieval_vector(video_id: str):
    with _LOCK:
        return _response(video_id, _get_state(video_id))
