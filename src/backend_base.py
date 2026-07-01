"""Common interface for vector backends (OpenSearch / PostgreSQL-pgvector).

The dedup runner talks only to this interface, so adding a new engine means
implementing these methods. Both backends share the same sequential-batch
contract:

    setup()                       -> create a fresh index/table + HNSW index
    find_matches(vectors) -> mask -> which survivors already exist (dup)
    index_survivors(vecs, ids)    -> insert non-matching survivors
    refresh()                     -> make writes visible to the next batch
    needs_refresh_wait()          -> whether the loop should sleep after refresh
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class VectorBackend(ABC):
    """Abstract vector store used by the dedup loop."""

    @abstractmethod
    def setup(self) -> None:
        """Create a fresh index/table and its HNSW vector index."""

    @abstractmethod
    def find_matches(self, vectors: np.ndarray) -> np.ndarray:
        """Return a bool mask: True where a near-duplicate already exists."""

    @abstractmethod
    def index_survivors(self, vectors: np.ndarray, ids) -> int:
        """Insert the given vectors; return the number actually written."""

    def refresh(self) -> None:
        """Make just-written vectors searchable. Default: no-op."""

    def needs_refresh_wait(self) -> bool:
        """Whether the loop must sleep after refresh for near-real-time stores.

        OpenSearch is near-real-time (needs a ~1s refresh wait). PostgreSQL is
        transactionally consistent (visible right after commit) -> no wait.
        """
        return False

    def count(self) -> int:
        """Total documents/rows currently indexed. -1 if unsupported."""
        return -1
