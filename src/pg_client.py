"""PostgreSQL + pgvector backend (e.g. Amazon RDS for PostgreSQL).

Mirrors the OpenSearch backend but uses pgvector's HNSW index. Cosine
similarity maps to pgvector's cosine *distance* operator ``<=>``:

    cosine_distance = 1 - cosine_similarity
    cosine_sim >= 0.95   <=>   cosine_distance <= 0.05

So a survivor is a duplicate if its nearest neighbor's cosine distance is
<= (1 - cosine_threshold).

Requires the ``vector`` extension (RDS PostgreSQL supports it via
``CREATE EXTENSION vector``) and the ``psycopg2`` + ``pgvector`` packages.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .config import Config
from .backend_base import VectorBackend


def _vec_literal(v: np.ndarray) -> str:
    """Format a numpy vector as a pgvector text literal: '[1,2,3]'."""
    return "[" + ",".join(f"{x:.7g}" for x in v.tolist()) + "]"


class PgVectorIndex(VectorBackend):
    """pgvector-backed dedup index over a single table."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        # Lazy imports so OpenSearch-only / dry-run users don't need psycopg2.
        import psycopg2
        from psycopg2.pool import ThreadedConnectionPool

        self._psycopg2 = psycopg2
        self.threshold_distance = 1.0 - cfg.cosine_threshold

        maxconn = max(cfg.msearch_workers, cfg.bulk_workers) + 1
        self.pool = ThreadedConnectionPool(
            minconn=1, maxconn=maxconn,
            host=cfg.pg_host, port=cfg.pg_port, dbname=cfg.pg_db,
            user=cfg.pg_user, password=cfg.pg_password, sslmode=cfg.pg_sslmode,
        )

    # ---- connection helper ----
    def _conn(self):
        c = self.pool.getconn()
        c.autocommit = True  # commits are immediately visible to next batch
        return c

    # ---- VectorBackend interface ----
    def setup(self) -> None:
        cfg = self.cfg
        conn = self._conn()
        try:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                cur.execute(f"DROP TABLE IF EXISTS {cfg.pg_table};")
                cur.execute(
                    f"CREATE TABLE {cfg.pg_table} ("
                    f"  id BIGSERIAL PRIMARY KEY,"
                    f"  video_id TEXT,"
                    f"  embedding vector({cfg.dimension})"
                    f");"
                )
                # --- HNSW build tuning (memory-sensitive) ---
                # If the graph exceeds maintenance_work_mem, pgvector spills to
                # disk and the build slows by orders of magnitude. Raise it and
                # (optionally) enable parallel index-build workers per session.
                if cfg.pg_maintenance_work_mem:
                    try:
                        cur.execute(
                            f"SET maintenance_work_mem = '{cfg.pg_maintenance_work_mem}';")
                        print(f"[pg] maintenance_work_mem = "
                              f"{cfg.pg_maintenance_work_mem}")
                    except Exception as e:
                        print(f"[warn] could not set maintenance_work_mem: {e}")
                if cfg.pg_max_parallel_maintenance_workers > 0:
                    try:
                        cur.execute(
                            f"SET max_parallel_maintenance_workers = "
                            f"{cfg.pg_max_parallel_maintenance_workers};")
                        print(f"[pg] max_parallel_maintenance_workers = "
                              f"{cfg.pg_max_parallel_maintenance_workers}")
                    except Exception as e:
                        print(f"[warn] could not set parallel maint workers: {e}")

                # HNSW index for cosine distance (built empty; maintained on insert)
                cur.execute(
                    f"CREATE INDEX ON {cfg.pg_table} "
                    f"USING hnsw (embedding vector_cosine_ops) "
                    f"WITH (m = {cfg.pg_hnsw_m}, "
                    f"ef_construction = {cfg.pg_hnsw_ef_construction});"
                )
            print(f"[pg] table '{cfg.pg_table}' + HNSW cosine index ready")
        finally:
            self.pool.putconn(conn)

    def needs_refresh_wait(self) -> bool:
        return False  # PostgreSQL is read-your-writes after commit

    # ---- radial search (is-duplicate check) ----
    def _match_chunk_single(self, chunk: np.ndarray) -> list[bool]:
        """Per-vector kNN queries. Each ``ORDER BY embedding <=> $1 LIMIT 1``
        uses a bound parameter, so pgvector CAN use the HNSW index. Round trips
        are amortized on one pooled connection; DB-side parallelism comes from
        running many chunks (connections) concurrently in the thread pool.
        """
        conn = self._conn()
        try:
            out = [False] * len(chunk)
            sql = (
                f"SELECT embedding <=> %s::vector AS dist "
                f"FROM {self.cfg.pg_table} "
                f"ORDER BY embedding <=> %s::vector LIMIT 1"
            )
            with conn.cursor() as cur:
                cur.execute(f"SET hnsw.ef_search = {self.cfg.pg_hnsw_ef_search};")
                for i, v in enumerate(chunk):
                    lit = _vec_literal(v)
                    cur.execute(sql, (lit, lit))
                    row = cur.fetchone()
                    out[i] = (row is not None and row[0] is not None
                              and row[0] <= self.threshold_distance)
            return out
        finally:
            self.pool.putconn(conn)

    def _match_chunk_lateral(self, chunk: np.ndarray) -> list[bool]:
        """Batch many vectors in one round trip via LATERAL. Fewer round trips,
        but the correlated ORDER BY may not use the HNSW index (can seq-scan).
        Kept for A/B comparison; not the default.
        """
        conn = self._conn()
        try:
            with conn.cursor() as cur:
                cur.execute(f"SET hnsw.ef_search = {self.cfg.pg_hnsw_ef_search};")
                literals = [_vec_literal(v) for v in chunk]
                sql = (
                    "SELECT q.ord, n.dist "
                    "FROM unnest(%s::text[]) WITH ORDINALITY AS q(vt, ord) "
                    "LEFT JOIN LATERAL ("
                    f"  SELECT t.embedding <=> q.vt::vector AS dist "
                    f"  FROM {self.cfg.pg_table} t "
                    f"  ORDER BY t.embedding <=> q.vt::vector "
                    "  LIMIT 1"
                    ") n ON true "
                    "ORDER BY q.ord;"
                )
                cur.execute(sql, (literals,))
                rows = cur.fetchall()
            out = [False] * len(chunk)
            for ord_, dist in rows:
                out[ord_ - 1] = (dist is not None) and (dist <= self.threshold_distance)
            return out
        finally:
            self.pool.putconn(conn)

    def _match_chunk(self, chunk: np.ndarray) -> list[bool]:
        if self.cfg.pg_query_mode == "lateral":
            return self._match_chunk_lateral(chunk)
        return self._match_chunk_single(chunk)

    def find_matches(self, vectors: np.ndarray) -> np.ndarray:
        if len(vectors) == 0:
            return np.zeros(0, dtype=bool)
        cs = self.cfg.msearch_chunk
        chunks = [vectors[i:i + cs] for i in range(0, len(vectors), cs)]
        with ThreadPoolExecutor(max_workers=self.cfg.msearch_workers) as ex:
            results = list(ex.map(self._match_chunk, chunks))
        return np.array([b for r in results for b in r], dtype=bool)

    def explain_search(self) -> str:
        """Return the EXPLAIN plan for one kNN dedup query, to verify the HNSW
        index is actually used (look for 'Index Scan using ...hnsw')."""
        import numpy as _np
        v = _np.zeros(self.cfg.dimension, dtype=_np.float32)
        v[0] = 1.0
        lit = _vec_literal(v)
        conn = self._conn()
        try:
            with conn.cursor() as cur:
                cur.execute(f"SET hnsw.ef_search = {self.cfg.pg_hnsw_ef_search};")
                cur.execute(
                    f"EXPLAIN SELECT embedding <=> %s::vector AS dist "
                    f"FROM {self.cfg.pg_table} "
                    f"ORDER BY embedding <=> %s::vector LIMIT 1",
                    (lit, lit),
                )
                return "\n".join(r[0] for r in cur.fetchall())
        finally:
            self.pool.putconn(conn)

    # ---- bulk insert (survivors) ----
    def _insert_chunk(self, chunk_vecs: np.ndarray, chunk_ids) -> int:
        from psycopg2.extras import execute_values

        conn = self._conn()
        try:
            rows = [(str(vid), _vec_literal(v)) for v, vid in zip(chunk_vecs, chunk_ids)]
            with conn.cursor() as cur:
                execute_values(
                    cur,
                    f"INSERT INTO {self.cfg.pg_table} (video_id, embedding) VALUES %s",
                    rows,
                    template="(%s, %s::vector)",
                    page_size=1000,
                )
            return len(rows)
        finally:
            self.pool.putconn(conn)

    def index_survivors(self, vectors: np.ndarray, ids) -> int:
        if len(vectors) == 0:
            return 0
        cs = self.cfg.bulk_chunk
        chunks = [(vectors[i:i + cs], ids[i:i + cs])
                  for i in range(0, len(vectors), cs)]
        total = 0
        with ThreadPoolExecutor(max_workers=self.cfg.bulk_workers) as ex:
            for ok in ex.map(lambda c: self._insert_chunk(c[0], c[1]), chunks):
                total += ok
        return total

    def count(self) -> int:
        conn = self._conn()
        try:
            with conn.cursor() as cur:
                cur.execute(f"SELECT count(*) FROM {self.cfg.pg_table};")
                return int(cur.fetchone()[0])
        finally:
            self.pool.putconn(conn)

    def close(self) -> None:
        try:
            self.pool.closeall()
        except Exception:
            pass
