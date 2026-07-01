"""Sequential-batch dedup runner.

Implements the loop from the solution design:

    for each batch:
        1. LOCAL DEDUP   (numpy matrix multiply, exact intra-batch)
        2. _msearch      (radial search survivors against the index)
        3. _bulk         (index only the non-matching survivors)
        4. wait refresh  (so the next batch sees these vectors)

Tracks throughput and — using the generator's ground-truth group_ids —
measures real dedup accuracy (leaked duplicates + missed uniques).

A ``--dry-run`` mode simulates the index with a local ground-truth oracle so
the full pipeline (including accuracy accounting) can be validated before an
OpenSearch cluster exists.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .config import Config
from .data_generator import VectorGenerator
from .local_dedup import local_dedup, local_dedup_blocked


@dataclass
class Stats:
    processed: int = 0            # total vectors pulled from the source
    local_discarded: int = 0      # removed by intra-batch local dedup
    os_discarded: int = 0         # removed by _msearch (matched existing)
    indexed: int = 0             # written to the index (survivors)
    batches: int = 0
    elapsed_s: float = 0.0
    # ground-truth accounting
    groups_seen: set = field(default_factory=set)
    leaked_dups: int = 0          # a group indexed more than once
    _indexed_groups: set = field(default_factory=set)

    @property
    def vps(self) -> float:
        return self.processed / self.elapsed_s if self.elapsed_s else 0.0

    def summary(self) -> dict:
        unique_truth = len(self.groups_seen)
        return {
            "processed": self.processed,
            "indexed": self.indexed,
            "local_discarded": self.local_discarded,
            "os_discarded": self.os_discarded,
            "batches": self.batches,
            "elapsed_s": round(self.elapsed_s, 2),
            "throughput_vps": round(self.vps, 1),
            "ground_truth_unique": unique_truth,
            "leaked_duplicates": self.leaked_dups,
            "accuracy_pct": round(
                100.0 * (1 - self.leaked_dups / max(1, self.processed - unique_truth)), 3
            ) if self.processed > unique_truth else 100.0,
        }


class _OracleIndex:
    """In-memory stand-in for OpenSearch used by --dry-run.

    Uses the ground-truth group_ids to answer "does a match already exist?"
    exactly, so pipeline logic and accounting can be tested without a cluster.
    """

    def __init__(self):
        self._indexed_groups: set[int] = set()

    def find_matches_by_group(self, group_ids: np.ndarray) -> np.ndarray:
        return np.array([g in self._indexed_groups for g in group_ids], dtype=bool)

    def add_groups(self, group_ids) -> None:
        self._indexed_groups.update(int(g) for g in group_ids)


def _pick_local_dedup(n: int):
    # dense matrix is fine up to ~10k; use blocked variant beyond that
    return local_dedup if n <= 10_000 else local_dedup_blocked


def run(cfg: Config, progress_every: int = 10) -> Stats:
    gen = VectorGenerator(
        total=cfg.total_vectors, dim=cfg.dimension,
        dup_ratio=cfg.duplicate_ratio, near_dup_sim=cfg.near_dup_sim,
        seed=cfg.seed,
    )
    stats = Stats()

    index = None
    oracle = None
    if cfg.dry_run:
        oracle = _OracleIndex()
        print("[dry-run] no OpenSearch; using ground-truth oracle index")
    else:
        # imported lazily so numpy-only dry-runs need no opensearch-py
        from .os_client import DedupIndex
        index = DedupIndex(cfg)
        print(f"[live] connecting to {cfg.host}:{cfg.port}, recreating index "
              f"'{cfg.index_name}'")
        index.recreate_index()
        try:
            index.tune_circuit_breaker("75%")
        except Exception as e:  # non-fatal on serverless / limited perms
            print(f"[warn] could not set circuit breaker: {e}")

    t0 = time.time()
    for batch in gen.batches(cfg.batch_size):
        stats.batches += 1
        stats.processed += len(batch.vectors)
        stats.groups_seen.update(int(g) for g in batch.group_ids)

        # 1) local dedup (exact, intra-batch)
        dedup_fn = _pick_local_dedup(len(batch.vectors))
        keep = dedup_fn(batch.vectors, cfg.cosine_threshold)
        stats.local_discarded += int((~keep).sum())
        surv_vecs = batch.vectors[keep]
        surv_ids = batch.ids[keep]
        surv_groups = batch.group_ids[keep]

        # 2) radial search survivors against the index
        if cfg.dry_run:
            matched = oracle.find_matches_by_group(surv_groups)
        else:
            matched = index.find_matches(surv_vecs)
        stats.os_discarded += int(matched.sum())

        final_vecs = surv_vecs[~matched]
        final_ids = surv_ids[~matched]
        final_groups = surv_groups[~matched]

        # ground-truth: count leaks (a group indexed more than once)
        for g in final_groups:
            gi = int(g)
            if gi in stats._indexed_groups:
                stats.leaked_dups += 1
            else:
                stats._indexed_groups.add(gi)

        # 3) index survivors
        if cfg.dry_run:
            oracle.add_groups(final_groups)
            stats.indexed += len(final_vecs)
        else:
            stats.indexed += index.index_survivors(final_vecs, final_ids)
            # 4) wait for refresh so next batch sees these vectors
            index.refresh()
            if cfg.refresh_wait_s:
                time.sleep(cfg.refresh_wait_s)

        stats.elapsed_s = time.time() - t0
        if stats.batches % progress_every == 0:
            print(f"  batch {stats.batches:>6} | processed {stats.processed:>12,} "
                  f"| indexed {stats.indexed:>12,} | {stats.vps:,.0f} vec/s")

    stats.elapsed_s = time.time() - t0
    return stats
