"""Configuration for the OpenSearch vector dedup test demo.

All tunables live here. Values can also be overridden by CLI flags or
environment variables (see ``run_dedup.py``).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class Config:
    # ---- Backend selection ----
    # "opensearch" (Faiss HNSW kNN) or "pgvector" (RDS PostgreSQL + pgvector)
    backend: str = os.getenv("BACKEND", "opensearch")

    # ---- OpenSearch connection ----
    host: str = os.getenv("OS_HOST", "localhost")
    port: int = int(os.getenv("OS_PORT", "443"))
    use_ssl: bool = os.getenv("OS_USE_SSL", "true").lower() == "true"
    verify_certs: bool = os.getenv("OS_VERIFY_CERTS", "true").lower() == "true"
    username: str | None = os.getenv("OS_USERNAME")
    password: str | None = os.getenv("OS_PASSWORD")
    # AWS SigV4 auth (for managed Amazon OpenSearch Service). If enabled,
    # username/password are ignored and boto3 credentials are used.
    use_aws_auth: bool = os.getenv("OS_USE_AWS_AUTH", "false").lower() == "true"
    aws_region: str = os.getenv("OS_AWS_REGION", "us-west-2")
    aws_service: str = os.getenv("OS_AWS_SERVICE", "es")  # "es" or "aoss"

    # ---- Index / vector params (mirror the solution design) ----
    index_name: str = os.getenv("OS_INDEX", "video_vectors")
    dimension: int = 768
    space_type: str = "innerproduct"          # cosine on normalized vectors
    hnsw_m: int = 16
    hnsw_ef_construction: int = 128
    # Dedup only needs top-1 above threshold, not high-recall ranking, so a
    # small ef_search cuts per-query CPU a lot. Raise via OS_EF_SEARCH if
    # accuracy stats show missed near-duplicates.
    hnsw_ef_search: int = int(os.getenv("OS_EF_SEARCH", "32"))
    number_of_shards: int = int(os.getenv("OS_SHARDS", "8"))
    number_of_replicas: int = int(os.getenv("OS_REPLICAS", "0"))
    # "-1" disables periodic auto-refresh: visibility is driven solely by the
    # explicit index.refresh() at the end of each batch. This avoids the ~1
    # refresh/sec firing during search+write (each auto-refresh cuts a new
    # segment -> segment bloat -> search slows down). Set OS_REFRESH_INTERVAL
    # back to e.g. "1s" for A/B comparison.
    refresh_interval: str = os.getenv("OS_REFRESH_INTERVAL", "-1")

    # ---- Dedup thresholds ----
    cosine_threshold: float = 0.95            # >= is a duplicate
    min_score: float = 1.95                   # innerproduct score = 1 + cosine

    # ---- Batch / chunk sizes ----
    batch_size: int = int(os.getenv("BATCH_SIZE", "20000"))
    msearch_chunk: int = int(os.getenv("MSEARCH_CHUNK", "1000"))
    bulk_chunk: int = int(os.getenv("BULK_CHUNK", "5000"))
    msearch_workers: int = int(os.getenv("MSEARCH_WORKERS", "20"))
    bulk_workers: int = int(os.getenv("BULK_WORKERS", "4"))
    # index.refresh() is synchronous (returns after the refresh completes), so
    # an extra sleep is normally unnecessary. Default 0; raise if needed.
    refresh_wait_s: float = float(os.getenv("REFRESH_WAIT_S", "0"))

    # ---- PostgreSQL / pgvector connection (RDS) ----
    pg_host: str = os.getenv("PG_HOST", "localhost")
    pg_port: int = int(os.getenv("PG_PORT", "5432"))
    pg_db: str = os.getenv("PG_DB", "vectordb")
    pg_user: str = os.getenv("PG_USER", "postgres")
    pg_password: str = os.getenv("PG_PASSWORD", "")
    pg_sslmode: str = os.getenv("PG_SSLMODE", "require")  # RDS defaults to SSL
    pg_table: str = os.getenv("PG_TABLE", "video_vectors")
    # pgvector HNSW build params (analogous to OpenSearch m / ef_construction)
    pg_hnsw_m: int = int(os.getenv("PG_HNSW_M", "16"))
    pg_hnsw_ef_construction: int = int(os.getenv("PG_HNSW_EF_CONSTRUCTION", "128"))
    # Dedup only needs "is the nearest neighbor closer than threshold?", not a
    # high-precision Top-K, so ef_search can be much lower than for ranking.
    pg_hnsw_ef_search: int = int(os.getenv("PG_HNSW_EF_SEARCH", "128"))
    # Radial-search query mode:
    #   "single"  -> one indexed kNN query per vector (bound param, uses HNSW)
    #   "lateral" -> batch many vectors in one round trip via LATERAL join
    #                (fewer round trips, but the correlated ORDER BY may NOT
    #                 use the HNSW index -> can degrade to a seq scan)
    pg_query_mode: str = os.getenv("PG_QUERY_MODE", "single")
    # HNSW index build is memory-sensitive: if the graph doesn't fit in
    # maintenance_work_mem it spills to disk and build slows dramatically.
    # Raised per-session before creating the index. Empty => leave RDS default.
    pg_maintenance_work_mem: str = os.getenv("PG_MAINTENANCE_WORK_MEM", "2GB")
    # Parallel workers for the index build (0 => leave server default).
    pg_max_parallel_maintenance_workers: int = int(
        os.getenv("PG_MAX_PARALLEL_MAINT_WORKERS", "0"))

    # ---- Test data generation ----
    total_vectors: int = int(os.getenv("TOTAL_VECTORS", "100000000"))  # 1亿
    duplicate_ratio: float = float(os.getenv("DUP_RATIO", "0.30"))     # 30% dups
    # Injected cosine of a near-dup to its group base. 0.99 keeps *pairwise*
    # within-group similarity (~sim^2) comfortably above the 0.95 threshold,
    # so ground truth ("same group") is consistent with cosine detection.
    # Lower this toward ~0.96 to stress near-threshold HNSW recall.
    near_dup_sim: float = float(os.getenv("NEAR_DUP_SIM", "0.99"))
    seed: int = int(os.getenv("SEED", "42"))

    # ---- Run behavior ----
    dry_run: bool = False        # if True, skip OpenSearch and only do local dedup
    request_timeout: int = 120
    max_retries: int = 3

    def index_settings(self) -> dict:
        return {
            "settings": {
                "index": {
                    "number_of_shards": self.number_of_shards,
                    "number_of_replicas": self.number_of_replicas,
                    "refresh_interval": self.refresh_interval,
                    "knn": True,
                    "knn.algo_param.ef_search": self.hnsw_ef_search,
                }
            },
            "mappings": {
                "properties": {
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": self.dimension,
                        "data_type": "float",
                        "method": {
                            "name": "hnsw",
                            "engine": "faiss",
                            "space_type": self.space_type,
                            "parameters": {
                                "m": self.hnsw_m,
                                "ef_construction": self.hnsw_ef_construction,
                            },
                        },
                    },
                    "video_id": {"type": "keyword"},
                }
            },
        }
