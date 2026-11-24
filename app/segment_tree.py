"""Shared shot index + per-video restriction overlays.

The same film is often released through several channels, each with its own
video id.  Its shots (order, 16-dim features, attention scores) are identical,
so the expensive part -- the leaf payloads and the full segment tree of
softmax aggregates -- is stored **once** in a :class:`ShotIndex`, shared by
every video id that imported the same content (``app.main`` fingerprints the
payload and reference-counts the catalog).

Each video id keeps only its own restriction state in a :class:`RestrictionIndex`:

* ``covers[node]``    -- how many live orders cover this canonical node
* ``overrides[node]`` -- aggregate for nodes whose visible set differs from
  the shared base index (``_EMPTY`` means the whole node is hidden)
* ``orders[order_id]`` -- the registered range

Both dicts are sparse and only hold nodes touched by live orders, so for R
live orders over N shots the per-video overlay is O(R log N).

For the visible set V the retrieval vector is

    w_i = exp(score_i - M) / sum_j exp(score_j - M),  M = max_j score_j
    v   = sum_i w_i * feature_i            (16 dims)

Base nodes and overlay nodes store, for the visible shots in their interval:

    cnt  - number of visible shots
    mx   - max attention score among them (-inf if none)
    sm   - sum of exp(score - mx)
    vec  - sum of exp(score - mx) * feature   (16 floats, flat array)

Sibling aggregates are merged by re-scaling to the common max:

    m  = max(mx_l, mx_r)
    sm = sm_l * exp(mx_l - m) + sm_r * exp(mx_r - m)
    vec = vec_l * exp(mx_l - m) + vec_r * exp(mx_r - m)

which is the textbook numerically-stable online softmax decomposition.

Hiding is a lazy *reference count* per canonical node: overlapping orders each
add +1 on the O(log N) nodes covering their range; revoking adds -1.  A node
with cover > 0 contributes nothing.  Children aggregates are always kept
valid, so no push-down is needed and both register and revoke cost
O(log N) node visits with a 16-dim merge each (constant factor).
"""

import math
import sys
from array import array

DIM = 16
NEG_INF = float("-inf")
_ZERO = array("d", bytes(8 * DIM))  # 16 zero doubles
_EMPTY = object()  # override marker: whole node hidden


def _deep_size(obj):
    """Approximate resident bytes of a small Python object graph.

    ``array`` objects already report their whole buffer from ``getsizeof``;
    dict/list/tuple/set are walked so per-entry overhead is included.
    """
    if isinstance(obj, dict):
        return sys.getsizeof(obj) + sum(
            _deep_size(k) + _deep_size(v) for k, v in obj.items()
        )
    if isinstance(obj, (list, tuple, set, frozenset)):
        return sys.getsizeof(obj) + sum(_deep_size(x) for x in obj)
    return sys.getsizeof(obj)


class ShotIndex:
    """Immutable, shareable segment tree over all shots of one film.

    Every shot is visible in this index; per-video hiding lives in a sparse
    :class:`RestrictionIndex` overlay.
    """

    __slots__ = ("n", "size", "scores", "features", "cnt", "maxs", "sums", "vec",
                 "__weakref__")

    def __init__(self, features, scores):
        """``features`` is a flat array('d') of n*16 values, ``scores`` an
        array('d') of n values (same input format as the import payload)."""
        n = len(scores)
        if n == 0:
            raise ValueError("at least one shot is required")
        if len(features) != n * DIM:
            raise ValueError("features must be a flat n*16 array")
        self.n = n
        size = 1
        while size < n:
            size <<= 1
        self.size = size
        total = 2 * size
        self.scores = scores
        self.features = features
        # Compact node storage (array, not list-of-Python-floats).
        self.cnt = array("i", bytes(4 * total))
        self.maxs = array("d", [NEG_INF]) * total
        self.sums = array("d", bytes(8 * total))
        self.vec = array("d", bytes(8 * DIM * total))
        base = size
        for i in range(n):
            node = base + i
            self.cnt[node] = 1
            self.maxs[node] = scores[i]
            self.sums[node] = 1.0  # exp(score - score)
            self.vec[node * DIM:(node + 1) * DIM] = features[i * DIM:(i + 1) * DIM]
        for node in range(size - 1, 0, -1):
            self._pull(node)

    # -- internal ---------------------------------------------------------

    def _pull(self, node):
        l = node * 2
        r = l + 1
        cnt = self.cnt
        maxs = self.maxs
        sums = self.sums
        vec = self.vec
        lc = cnt[l]
        rc = cnt[r]
        b = node * DIM
        if lc == 0:
            cnt[node] = rc
            maxs[node] = maxs[r]
            sums[node] = sums[r]
            rb = r * DIM
            vec[b:b + DIM] = vec[rb:rb + DIM]
            return
        if rc == 0:
            cnt[node] = lc
            maxs[node] = maxs[l]
            sums[node] = sums[l]
            lb = l * DIM
            vec[b:b + DIM] = vec[lb:lb + DIM]
            return
        ml = maxs[l]
        mr = maxs[r]
        m = ml if ml > mr else mr
        wl = math.exp(ml - m)
        wr = math.exp(mr - m)
        cnt[node] = lc + rc
        maxs[node] = m
        sums[node] = sums[l] * wl + sums[r] * wr
        lb = l * DIM
        rb = r * DIM
        for k in range(DIM):
            vec[b + k] = vec[lb + k] * wl + vec[rb + k] * wr

    # -- public -----------------------------------------------------------

    def byte_size(self):
        """Resident bytes of this catalog's leaf payload + base tree."""
        return sum(
            sys.getsizeof(x)
            for x in (self.scores, self.features, self.cnt,
                      self.maxs, self.sums, self.vec)
        )


class RestrictionIndex:
    """Sparse per-video overlay on a shared :class:`ShotIndex`."""

    __slots__ = ("base", "covers", "overrides", "orders")

    def __init__(self, base):
        self.base = base
        self.covers = {}    # node -> live-order cover count
        self.overrides = {}  # node -> array('d', 19) aggregate, or _EMPTY
        self.orders = {}    # order_id -> (start, end)

    # -- overlay node access ---------------------------------------------

    def _recompute(self, node):
        """Refresh ``node``'s aggregate from its cover count and children.

        An override is removed whenever the node's visible aggregate is
        exactly the base aggregate (no cover here and no overridden child),
        keeping overlay storage proportional to live orders.
        """
        if self.covers.get(node, 0) > 0:
            self.overrides[node] = _EMPTY
            return
        if node >= self.base.size:  # leaf, not covered: identical to base
            self.overrides.pop(node, None)
            return
        l = node * 2
        r = l + 1
        lo = self.overrides.get(l)
        ro = self.overrides.get(r)
        if lo is None and ro is None:
            self.overrides.pop(node, None)
            return

        base = self.base
        if lo is _EMPTY:
            lc, lm, ls, lv, lb = 0, NEG_INF, 0.0, None, 0
        elif lo is None:
            lc, lm, ls, lv, lb = (
                base.cnt[l], base.maxs[l], base.sums[l], base.vec, l * DIM,
            )
        else:
            lc, lm, ls, lv, lb = int(lo[0]), lo[1], lo[2], lo, 3
        if ro is _EMPTY:
            rc, rm, rs, rv, rb = 0, NEG_INF, 0.0, None, 0
        elif ro is None:
            rc, rm, rs, rv, rb = (
                base.cnt[r], base.maxs[r], base.sums[r], base.vec, r * DIM,
            )
        else:
            rc, rm, rs, rv, rb = int(ro[0]), ro[1], ro[2], ro, 3

        if lc == 0 and rc == 0:
            if base.cnt[node] == 0:  # padding node: base already empty
                self.overrides.pop(node, None)
            else:
                self.overrides[node] = _EMPTY
            return
        if lc == 0:
            self.overrides[node] = array("d", [rc, rm, rs, *rv[rb:rb + DIM]])
            return
        if rc == 0:
            self.overrides[node] = array("d", [lc, lm, ls, *lv[lb:lb + DIM]])
            return
        m = lm if lm > rm else rm
        wl = math.exp(lm - m)
        wr = math.exp(rm - m)
        vals = [lc + rc, m, ls * wl + rs * wr]
        for k in range(DIM):
            vals.append(lv[lb + k] * wl + rv[rb + k] * wr)
        self.overrides[node] = array("d", vals)

    def _update(self, node, nl, nr, ql, qr, delta):
        if qr <= nl or nr <= ql:
            return
        if ql <= nl and nr <= qr:
            c = self.covers.get(node, 0) + delta
            if c > 0:
                self.covers[node] = c
            else:
                self.covers.pop(node, None)
            self._recompute(node)
            return
        mid = (nl + nr) >> 1
        self._update(node * 2, nl, mid, ql, qr, delta)
        self._update(node * 2 + 1, mid, nr, ql, qr, delta)
        self._recompute(node)

    # -- public -----------------------------------------------------------

    def register(self, order_id, start, end):
        """Hide shots [start, end) under ``order_id``.  O(log N)."""
        if not (0 <= start < end <= self.base.n):
            raise ValueError("invalid range")
        # Callers enforce idempotency/conflict before calling this.
        self.orders[order_id] = (start, end)
        self._update(1, 0, self.base.size, start, end, +1)

    def revoke(self, order_id):
        """Remove one order; shots still covered by others stay hidden."""
        start, end = self.orders.pop(order_id)
        self._update(1, 0, self.base.size, start, end, -1)

    def visible_count(self):
        ov = self.overrides.get(1)
        if ov is None:
            return self.base.cnt[1]
        if ov is _EMPTY:
            return 0
        return int(ov[0])

    def retrieval_vector(self):
        """Softmax-weighted feature sum over visible shots, or None if all hidden."""
        ov = self.overrides.get(1)
        if ov is None:
            s = self.base.sums[1]
            vec = self.base.vec
            return [vec[DIM + k] / s for k in range(DIM)]
        if ov is _EMPTY:
            return None
        s = ov[2]
        return [ov[3 + k] / s for k in range(DIM)]

    def byte_size(self):
        """Resident bytes of this video's sparse overlay + order table."""
        return (
            _deep_size(self.covers)
            + _deep_size(self.overrides)
            + _deep_size(self.orders)
        )
