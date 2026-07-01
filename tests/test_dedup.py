"""Unit tests for local dedup and the dry-run pipeline.

Run with:  python -m pytest tests/  -q
or simply: python tests/test_dedup.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.data_generator import VectorGenerator, l2_normalize
from src.local_dedup import local_dedup, local_dedup_blocked
from src.dedup_runner import run


def test_local_dedup_removes_exact_copies():
    rng = np.random.default_rng(0)
    base = l2_normalize(rng.standard_normal((5, 768)).astype(np.float32))
    # duplicate rows 0 and 2
    batch = np.vstack([base, base[0:1], base[2:3]])
    keep = local_dedup(batch, threshold=0.95)
    assert keep[:5].all()            # first occurrences kept
    assert not keep[5]               # copy of row 0 discarded
    assert not keep[6]               # copy of row 2 discarded
    assert keep.sum() == 5


def test_local_dedup_blocked_matches_dense():
    rng = np.random.default_rng(1)
    base = l2_normalize(rng.standard_normal((30, 768)).astype(np.float32))
    batch = np.vstack([base, base[3:6]])  # 3 dups
    dense = local_dedup(batch, 0.95)
    blocked = local_dedup_blocked(batch, 0.95, block=8)
    assert (dense == blocked).all()
    assert dense.sum() == 30


def test_generator_unique_group_count():
    gen = VectorGenerator(total=5000, dim=768, dup_ratio=0.3, seed=7)
    seen = set()
    for b in gen.batches(1000):
        seen.update(int(g) for g in b.group_ids)
    # ~70% unique groups expected; allow generous tolerance
    assert 0.55 * 5000 < len(seen) < 0.85 * 5000
    assert len(seen) == gen.unique_groups


def test_dry_run_zero_leaks():
    cfg = Config()
    cfg.dry_run = True
    cfg.total_vectors = 20000
    cfg.batch_size = 2000
    cfg.duplicate_ratio = 0.3
    stats = run(cfg, progress_every=1000)
    s = stats.summary()
    # dry-run uses an exact oracle => zero leaked duplicates and
    # indexed count == ground-truth unique groups
    assert s["leaked_duplicates"] == 0
    assert s["indexed"] == s["ground_truth_unique"]
    assert s["accuracy_pct"] == 100.0


if __name__ == "__main__":
    test_local_dedup_removes_exact_copies()
    test_local_dedup_blocked_matches_dense()
    test_generator_unique_group_count()
    test_dry_run_zero_leaks()
    print("All tests passed.")
