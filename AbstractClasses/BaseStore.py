"""
AbstractClasses/BaseStore.py
Abstract interface that every vector store must implement.
"""

from abc import ABC, abstractmethod
from langchain_core.documents import Document


class BaseStore(ABC):
    """Indexes chunks and retrieves them by similarity + optional filters."""

    @abstractmethod
    def index(self, chunks: list[Document]) -> None:
        """Add chunks to the vector store."""
        ...

    @abstractmethod
    def get_indexed_ids(self) -> set[str]:
        """
        Return the set of arxiv_ids already stored in the vector DB.
        Used by the ingestion pipeline to skip duplicates.
        """
        ...

    @abstractmethod
    def query(self, query: str, k: int = 5, filters: dict | None = None) -> list[Document]:
        """
        Retrieve the top-k most relevant chunks.

        filters: optional dict for metadata filtering, e.g.
            {"models": "CNN"} or {"metric_value_min": 90.0}
        """
        ...