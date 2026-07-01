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
    # per-phase cumulative seconds (to locate the real bottleneck)
    t_local: float = 0.0          # local numpy dedup
    t_search: float = 0.0         # radial search (is-duplicate check)
    t_index: float = 0.0          # bulk write / insert
    t_refresh: float = 0.0        # refresh + refresh wait
    # ground-truth accounting
    groups_seen: set = field(default_factory=set)
    leaked_dups: int = 0          # a group indexed more than once
    _indexed_groups: set = field(default_factory=set)

    @property
    def vps(self) -> float:
        return self.processed / self.elapsed_s if self.elapsed_s else 0.0

    def summary(self) -> dict:
        unique_truth = len(self.groups_seen)
        total_phase = self.t_local + self.t_search + self.t_index + self.t_refresh
        pct = lambda t: round(100.0 * t / total_phase, 1) if total_phase else 0.0
        return {
            "processed": self.processed,
            "indexed": self.indexed,
            "local_discarded": self.local_discarded,
            "os_discarded": self.os_discarded,
            "batches": self.batches,
            "elapsed_s": round(self.elapsed_s, 2),
            "throughput_vps": round(self.vps, 1),
            "phase_seconds": {
                "local_dedup": round(self.t_local, 2),
                "search": round(self.t_search, 2),
                "index_write": round(self.t_index, 2),
                "refresh": round(self.t_refresh, 2),
            },
            "phase_pct": {
                "local_dedup": pct(self.t_local),
                "search": pct(self.t_search),
                "index_write": pct(self.t_index),
                "refresh": pct(self.t_refresh),
            },
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
        print("[dry-run] no backend; using ground-truth oracle index")
    else:
        index = _build_backend(cfg)
        index.setup()
        # verify the vector index is actually used (diagnose seq-scan slowness)
        if hasattr(index, "explain_search"):
            try:
                plan = index.explain_search()
                uses_idx = "hnsw" in plan.lower() and "index scan" in plan.lower()
                print(f"[diag] search plan {'USES HNSW index' if uses_idx else 'does NOT use index (!)'}:")
                for line in plan.splitlines():
                    print(f"       {line}")
            except Exception as e:
                print(f"[warn] EXPLAIN failed: {e}")

    t0 = time.time()
    for batch in gen.batches(cfg.batch_size):
        stats.batches += 1
        stats.processed += len(batch.vectors)
        stats.groups_seen.update(int(g) for g in batch.group_ids)

        # 1) local dedup (exact, intra-batch)
        ts = time.time()
        dedup_fn = _pick_local_dedup(len(batch.vectors))
        keep = dedup_fn(batch.vectors, cfg.cosine_threshold)
        stats.t_local += time.time() - ts
        stats.local_discarded += int((~keep).sum())
        surv_vecs = batch.vectors[keep]
        surv_ids = batch.ids[keep]
        surv_groups = batch.group_ids[keep]

        # 2) radial search survivors against the index
        ts = time.time()
        if cfg.dry_run:
            matched = oracle.find_matches_by_group(surv_groups)
        else:
            matched = index.find_matches(surv_vecs)
        stats.t_search += time.time() - ts
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
            ts = time.time()
            stats.indexed += index.index_survivors(final_vecs, final_ids)
            stats.t_index += time.time() - ts
            # 4) make writes visible to the next batch
            ts = time.time()
            index.refresh()
            if index.needs_refresh_wait() and cfg.refresh_wait_s:
                time.sleep(cfg.refresh_wait_s)
            stats.t_refresh += time.time() - ts

        stats.elapsed_s = time.time() - t0
        if stats.batches % progress_every == 0:
            print(f"  batch {stats.batches:>6} | processed {stats.processed:>12,} "
                  f"| indexed {stats.indexed:>12,} | {stats.vps:,.0f} vec/s")

    stats.elapsed_s = time.time() - t0
    if index is not None and hasattr(index, "close"):
        index.close()
    return stats


def _build_backend(cfg: Config):
    """Instantiate the configured vector backend.

    Imports are lazy so a run only needs the driver for the chosen engine
    (e.g. pgvector runs don't require opensearch-py, and vice versa).
    """
    backend = cfg.backend.lower()
    if backend in ("opensearch", "os"):
        from .os_client import DedupIndex
        print(f"[live] OpenSearch {cfg.host}:{cfg.port}, index '{cfg.index_name}'")
        return DedupIndex(cfg)
    if backend in ("pgvector", "postgres", "postgresql", "pg", "rds"):
        from .pg_client import PgVectorIndex
        print(f"[live] PostgreSQL {cfg.pg_host}:{cfg.pg_port}/{cfg.pg_db}, "
              f"table '{cfg.pg_table}'")
        return PgVectorIndex(cfg)
    raise ValueError(f"unknown backend '{cfg.backend}' (use 'opensearch' or 'pgvector')")
