"""Thin OpenSearch client for the dedup demo.

Wraps index creation, parallel radial ``_msearch``, parallel ``_bulk`` writes,
and refresh. Supports basic-auth (self-managed / fine-grained access control)
and AWS SigV4 (managed Amazon OpenSearch Service / Serverless).
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from opensearchpy import OpenSearch, RequestsHttpConnection, helpers

try:
    import orjson
    _fast_dumps = lambda obj: orjson.dumps(obj).decode("utf-8")
except ImportError:
    _fast_dumps = json.dumps

from .config import Config
from .backend_base import VectorBackend


def build_client(cfg: Config) -> OpenSearch:
    http_auth = None
    if cfg.use_aws_auth:
        # Lazy import so boto3 is only needed when actually using AWS auth.
        from requests_aws4auth import AWS4Auth
        import boto3

        creds = boto3.Session().get_credentials()
        http_auth = AWS4Auth(
            creds.access_key, creds.secret_key, cfg.aws_region,
            cfg.aws_service, session_token=creds.token,
        )
    elif cfg.username and cfg.password:
        http_auth = (cfg.username, cfg.password)

    return OpenSearch(
        hosts=[{"host": cfg.host, "port": cfg.port}],
        http_auth=http_auth,
        use_ssl=cfg.use_ssl,
        verify_certs=cfg.verify_certs,
        ssl_show_warn=False,
        connection_class=RequestsHttpConnection,
        timeout=cfg.request_timeout,
        max_retries=cfg.max_retries,
        retry_on_timeout=True,
    )


class DedupIndex(VectorBackend):
    """Operations against a single kNN index for the dedup loop."""

    def __init__(self, cfg: Config, client: OpenSearch | None = None):
        self.cfg = cfg
        self.client = client or build_client(cfg)

    # ---- VectorBackend interface ----
    def setup(self) -> None:
        self.recreate_index()
        try:
            self.tune_circuit_breaker("75%")
        except Exception as e:  # non-fatal on serverless / limited perms
            print(f"[warn] could not set circuit breaker: {e}")

    def needs_refresh_wait(self) -> bool:
        return True  # OpenSearch is near-real-time

    # ---- index lifecycle ----
    def recreate_index(self) -> None:
        idx = self.cfg.index_name
        if self.client.indices.exists(index=idx):
            self.client.indices.delete(index=idx)
        self.client.indices.create(index=idx, body=self.cfg.index_settings())

    def count(self) -> int:
        self.client.indices.refresh(index=self.cfg.index_name)
        return int(self.client.count(index=self.cfg.index_name)["count"])

    def tune_circuit_breaker(self, limit: str = "75%") -> None:
        self.client.cluster.put_settings(
            body={"persistent": {"knn.memory.circuit_breaker.limit": limit}}
        )

    # ---- radial search (is-duplicate check) ----
    def _msearch_chunk(self, chunk: np.ndarray) -> list[bool]:
        """Return per-row ``has_match`` booleans for one chunk."""
        # Pre-built header line (same for all vectors in the index)
        header = _fast_dumps({"index": self.cfg.index_name})
        min_score = self.cfg.min_score
        lines = []
        for v in chunk:
            lines.append(header)
            lines.append(_fast_dumps({
                "size": 1,
                "query": {"knn": {"embedding": {
                    "vector": v.tolist(),
                    "min_score": min_score,
                }}},
            }))
        body = "\n".join(lines) + "\n"
        resp = self.client.msearch(body=body)
        out = []
        for r in resp["responses"]:
            total = r.get("hits", {}).get("total", {})
            val = total.get("value", 0) if isinstance(total, dict) else total
            out.append(val > 0)
        return out

    def find_matches(self, vectors: np.ndarray) -> np.ndarray:
        """Return a boolean mask: True where a duplicate already exists in the index."""
        if len(vectors) == 0:
            return np.zeros(0, dtype=bool)
        cs = self.cfg.msearch_chunk
        chunks = [vectors[i:i + cs] for i in range(0, len(vectors), cs)]
        with ThreadPoolExecutor(max_workers=self.cfg.msearch_workers) as ex:
            results = list(ex.map(self._msearch_chunk, chunks))
        return np.array([b for chunk_res in results for b in chunk_res], dtype=bool)

    # ---- bulk index (survivors) ----
    def _bulk_chunk(self, chunk_vecs: np.ndarray, chunk_ids) -> int:
        actions = (
            {
                "_index": self.cfg.index_name,
                "_source": {"embedding": v.tolist(), "video_id": str(vid)},
            }
            for v, vid in zip(chunk_vecs, chunk_ids)
        )
        ok, _ = helpers.bulk(self.client, actions, chunk_size=len(chunk_vecs),
                             request_timeout=self.cfg.request_timeout)
        return ok

    def index_survivors(self, vectors: np.ndarray, ids) -> int:
        if len(vectors) == 0:
            return 0
        cs = self.cfg.bulk_chunk
        chunks = [(vectors[i:i + cs], ids[i:i + cs])
                  for i in range(0, len(vectors), cs)]
        total = 0
        with ThreadPoolExecutor(max_workers=self.cfg.bulk_workers) as ex:
            for ok in ex.map(lambda c: self._bulk_chunk(c[0], c[1]), chunks):
                total += ok
        return total

    def refresh(self) -> None:
        self.client.indices.refresh(index=self.cfg.index_name)
