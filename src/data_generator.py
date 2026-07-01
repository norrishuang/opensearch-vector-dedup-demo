"""Simulated video-embedding vector generator.

Produces streams of L2-normalized 768-D FP32 vectors with a controllable
fraction of near-duplicates, so we can measure dedup accuracy against a
known ground truth.

Design goals:
- Never materialize all N vectors in memory (supports 1亿+ scale) — vectors
  are yielded batch by batch.
- Each vector carries a ``group_id``: vectors sharing a group_id are
  near-duplicates of one another (cosine >= threshold). Distinct group_ids
  are (with overwhelming probability) far apart in 768-D space.
- The number of *unique groups* is the ground-truth "unique vector" count.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (mat / norms).astype(np.float32)


@dataclass
class Batch:
    vectors: np.ndarray      # shape (n, dim), float32, L2-normalized
    ids: np.ndarray          # shape (n,), int64 global row index
    group_ids: np.ndarray    # shape (n,), int64 ground-truth group


class VectorGenerator:
    """Streaming generator of normalized vectors with injected near-dups.

    Parameters
    ----------
    total : int
        Total number of vectors to emit across all batches.
    dim : int
        Vector dimensionality (768).
    dup_ratio : float
        Fraction of emitted vectors that are near-duplicates of an
        earlier vector (0.0 - <1.0). E.g. 0.30 => ~30% duplicates,
        ~70% unique groups.
    near_dup_sim : float
        Target cosine similarity for an injected near-duplicate against
        its group's base vector (e.g. 0.97).
    seed : int
        RNG seed for reproducibility.
    """

    def __init__(self, total: int, dim: int = 768, dup_ratio: float = 0.30,
                 near_dup_sim: float = 0.99, seed: int = 42):
        if not 0.0 <= dup_ratio < 1.0:
            raise ValueError("dup_ratio must be in [0, 1)")
        self.total = total
        self.dim = dim
        self.dup_ratio = dup_ratio
        self.near_dup_sim = near_dup_sim
        self.rng = np.random.default_rng(seed)

        self._emitted = 0
        self._next_group = 0
        # Keep a rolling pool of recent base vectors so a duplicate can point
        # back to a real earlier vector. Bounded to cap memory.
        self._pool_vectors: list[np.ndarray] = []
        self._pool_groups: list[int] = []
        self._pool_cap = 200_000

    @property
    def unique_groups(self) -> int:
        """Ground-truth number of distinct groups emitted so far."""
        return self._next_group

    def _perturb(self, base: np.ndarray) -> np.ndarray:
        """Return a vector with cosine ~= near_dup_sim to ``base``."""
        noise = self.rng.standard_normal(self.dim).astype(np.float32)
        # remove component along base so noise is orthogonal-ish
        noise -= np.dot(noise, base) * base
        n = np.linalg.norm(noise)
        if n > 0:
            noise /= n
        s = self.near_dup_sim
        v = s * base + np.sqrt(max(0.0, 1.0 - s * s)) * noise
        return v.astype(np.float32)

    def _add_to_pool(self, vec: np.ndarray, group: int) -> None:
        if len(self._pool_vectors) >= self._pool_cap:
            # drop a random old entry to keep memory bounded
            idx = self.rng.integers(0, len(self._pool_vectors))
            self._pool_vectors[idx] = vec
            self._pool_groups[idx] = group
        else:
            self._pool_vectors.append(vec)
            self._pool_groups.append(group)

    def batches(self, batch_size: int):
        """Yield ``Batch`` objects until ``total`` vectors are emitted."""
        while self._emitted < self.total:
            n = int(min(batch_size, self.total - self._emitted))
            vecs = np.empty((n, self.dim), dtype=np.float32)
            groups = np.empty(n, dtype=np.int64)

            # decide which rows are duplicates. First-ever vector must be unique.
            is_dup = self.rng.random(n) < self.dup_ratio
            for i in range(n):
                make_dup = bool(is_dup[i]) and len(self._pool_vectors) > 0
                if make_dup:
                    # Always perturb the group's *original base* vector (never
                    # another near-dup), so every pair within a group stays
                    # >= near_dup_sim^2 similar and ground truth is consistent.
                    j = self.rng.integers(0, len(self._pool_vectors))
                    base = self._pool_vectors[j]
                    grp = self._pool_groups[j]
                    v = self._perturb(base)
                    vecs[i] = v
                    groups[i] = grp
                else:
                    v = self.rng.standard_normal(self.dim).astype(np.float32)
                    v /= np.linalg.norm(v)
                    grp = self._next_group
                    self._next_group += 1
                    vecs[i] = v
                    groups[i] = grp
                    self._add_to_pool(v, grp)

            ids = np.arange(self._emitted, self._emitted + n, dtype=np.int64)
            self._emitted += n
            yield Batch(vectors=l2_normalize(vecs), ids=ids, group_ids=groups)
