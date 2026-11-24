"""Segment tree maintaining a softmax-weighted feature sum over the *visible*
shots of one video.

For the visible set V the retrieval vector is

    w_i = exp(score_i - M) / sum_j exp(score_j - M),  M = max_j score_j
    v   = sum_i w_i * feature_i            (16 dims)

Each tree node stores, for the visible shots in its interval:

    cnt  - number of visible shots
    mx   - max attention score among them (-inf if none)
    sm   - sum of exp(score - mx)
    vec  - sum of exp(score - mx) * feature   (16 floats, flat array)

Two sibling nodes are merged by re-scaling to the common max:

    m  = max(mx_l, mx_r)
    sm = sm_l * exp(mx_l - m) + sm_r * exp(mx_r - m)
    vec = vec_l * exp(mx_l - m) + vec_r * exp(mx_r - m)

which is the textbook numerically-stable online softmax decomposition.

Hiding is a lazy *reference count* per node: overlapping orders each add +1
on the canonical O(log N) nodes covering their range; revoking adds -1.
A node with cover > 0 contributes nothing.  Children aggregates are always
kept valid, so no push-down is needed and both register and revoke cost
O(log N) node visits with a 16-dim merge each (constant factor).
"""

import math
from array import array

DIM = 16
NEG_INF = float("-inf")
_ZERO = array("d", bytes(8 * DIM))  # 16 zero doubles


class VisibleSoftmaxTree:
    __slots__ = ("n", "size", "cover", "cnt", "maxs", "sums", "vec",
                 "scores", "features")

    def __init__(self, features, scores):
        n = len(scores)
        if n == 0:
            raise ValueError("at least one shot is required")
        if any(len(f) != DIM for f in features):
            raise ValueError("every feature must have 16 dims")
        self.n = n
        size = 1
        while size < n:
            size <<= 1
        self.size = size
        total = 2 * size
        self.cover = [0] * total
        self.cnt = [0] * total
        self.maxs = [NEG_INF] * total
        self.sums = [0.0] * total
        self.vec = array("d", bytes(8 * DIM * total))  # zero-initialised
        # leaf payloads are kept separately so a leaf can be restored after
        # its cover counter returns to 0
        self.scores = array("d", scores)
        flat = array("d", bytes(8 * DIM * n))
        for i, f in enumerate(features):
            flat[i * DIM:(i + 1) * DIM] = array("d", f)
        self.features = flat
        base = size
        for i in range(n):
            node = base + i
            self.cnt[node] = 1
            self.maxs[node] = scores[i]
            self.sums[node] = 1.0  # exp(score - score)
            self.vec[node * DIM:(node + 1) * DIM] = flat[i * DIM:(i + 1) * DIM]
        for node in range(size - 1, 0, -1):
            self._pull(node)

    # -- internal ---------------------------------------------------------

    def _pull(self, node):
        """Recompute node aggregates from its cover counter and children."""
        b = node * DIM
        if node >= self.size:  # leaf
            i = node - self.size
            if self.cover[node] > 0 or i >= self.n:  # hidden or padding
                self.cnt[node] = 0
                self.maxs[node] = NEG_INF
                self.sums[node] = 0.0
                self.vec[b:b + DIM] = _ZERO
            else:
                self.cnt[node] = 1
                self.maxs[node] = self.scores[i]
                self.sums[node] = 1.0
                self.vec[b:b + DIM] = self.features[i * DIM:(i + 1) * DIM]
            return
        if self.cover[node] > 0:
            self.cnt[node] = 0
            self.maxs[node] = NEG_INF
            self.sums[node] = 0.0
            self.vec[b:b + DIM] = _ZERO
            return
        l = node * 2
        r = l + 1
        if self.cnt[l] == 0:
            self.cnt[node] = self.cnt[r]
            self.maxs[node] = self.maxs[r]
            self.sums[node] = self.sums[r]
            rb = r * DIM
            self.vec[b:b + DIM] = self.vec[rb:rb + DIM]
        elif self.cnt[r] == 0:
            self.cnt[node] = self.cnt[l]
            self.maxs[node] = self.maxs[l]
            self.sums[node] = self.sums[l]
            lb = l * DIM
            self.vec[b:b + DIM] = self.vec[lb:lb + DIM]
        else:
            ml = self.maxs[l]
            mr = self.maxs[r]
            m = ml if ml > mr else mr
            wl = math.exp(ml - m)
            wr = math.exp(mr - m)
            self.cnt[node] = self.cnt[l] + self.cnt[r]
            self.maxs[node] = m
            self.sums[node] = self.sums[l] * wl + self.sums[r] * wr
            lb = l * DIM
            rb = r * DIM
            vec = self.vec
            for k in range(DIM):
                vec[b + k] = vec[lb + k] * wl + vec[rb + k] * wr

    def _update(self, node, nl, nr, ql, qr, delta):
        if qr <= nl or nr <= ql:
            return
        if ql <= nl and nr <= qr:
            self.cover[node] += delta
            self._pull(node)
            return
        mid = (nl + nr) >> 1
        self._update(node * 2, nl, mid, ql, qr, delta)
        self._update(node * 2 + 1, mid, nr, ql, qr, delta)
        self._pull(node)

    # -- public -----------------------------------------------------------

    def add_cover(self, start, end, delta):
        """Add +1 (hide) or -1 (unhide) to every shot in [start, end).

        O(log N) node visits.
        """
        if not (0 <= start < end <= self.n):
            raise ValueError("invalid range")
        self._update(1, 0, self.size, start, end, delta)

    def visible_count(self):
        return self.cnt[1]

    def retrieval_vector(self):
        """Softmax-weighted feature sum over visible shots, or None if all hidden."""
        if self.cnt[1] == 0:
            return None
        s = self.sums[1]
        vec = self.vec
        return [vec[DIM + k] / s for k in range(DIM)]  # node 1 -> offset DIM
