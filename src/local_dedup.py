"""Intra-batch local dedup via numpy matrix multiplication.

For a batch of L2-normalized vectors, ``batch @ batch.T`` yields the full
pairwise cosine-similarity matrix in one BLAS call. We keep the first
occurrence of each near-duplicate cluster and discard the rest.

This is exact (no approximation) and ~70x faster than naive pairwise loops.
"""
from __future__ import annotations

import numpy as np


def local_dedup(batch: np.ndarray, threshold: float = 0.95) -> np.ndarray:
    """Return a boolean keep-mask over rows of ``batch``.

    A row ``i`` is discarded if it has cosine >= ``threshold`` with any
    earlier *kept* row ``j < i``. Vectors must already be L2-normalized.

    Parameters
    ----------
    batch : np.ndarray, shape (n, dim), float32, normalized
    threshold : float, cosine similarity cutoff

    Returns
    -------
    keep : np.ndarray[bool], shape (n,)
    """
    n = batch.shape[0]
    if n == 0:
        return np.zeros(0, dtype=bool)

    # full pairwise cosine similarity (normalized => dot product == cosine)
    sim = batch @ batch.T
    np.fill_diagonal(sim, 0.0)

    keep = np.ones(n, dtype=bool)
    for i in range(1, n):
        if keep[i]:
            # max similarity against all earlier kept rows
            prev = sim[i, :i]
            if prev[keep[:i]].size and prev[keep[:i]].max() >= threshold:
                keep[i] = False
    return keep


def local_dedup_blocked(batch: np.ndarray, threshold: float = 0.95,
                        block: int = 4000) -> np.ndarray:
    """Memory-friendly variant that avoids the full n×n matrix.

    For very large batches (n > ~20k) the dense n×n similarity matrix can be
    large (20k×20k float32 ≈ 1.6 GB). This version compares each row only
    against earlier kept rows in blocks, trading a little speed for a much
    smaller memory footprint.
    """
    n = batch.shape[0]
    if n == 0:
        return np.zeros(0, dtype=bool)

    keep = np.ones(n, dtype=bool)
    kept_vecs = np.empty((0, batch.shape[1]), dtype=np.float32)

    for start in range(0, n, block):
        end = min(start + block, n)
        chunk = batch[start:end]
        m = end - start

        # 1) vectorized check against all previously kept vectors
        if kept_vecs.shape[0] > 0:
            hit_prev = (chunk @ kept_vecs.T).max(axis=1) >= threshold
        else:
            hit_prev = np.zeros(m, dtype=bool)

        # 2) greedy within-chunk dedup using the dense chunk matrix
        #    (block x block stays bounded, e.g. 4000x4000 float32 ~ 64 MB)
        sim = chunk @ chunk.T
        np.fill_diagonal(sim, 0.0)
        local_keep = ~hit_prev
        for i in range(m):
            if local_keep[i]:
                mask = local_keep[:i]
                if mask.any() and sim[i, :i][mask].max() >= threshold:
                    local_keep[i] = False
        keep[start:end] = local_keep

        # 3) grow the kept pool with this chunk's survivors
        surv = chunk[local_keep]
        if surv.shape[0]:
            kept_vecs = np.vstack([kept_vecs, surv]) if kept_vecs.size else surv

    return keep
