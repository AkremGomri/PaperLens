"""
AbstractClasses/BaseExtractor.py
Abstract interface that every extractor must implement.
"""

from abc import ABC, abstractmethod
from langchain_core.documents import Document


class BaseExtractor(ABC):
    """Extracts metadata from documents."""

    @abstractmethod
    def extract(self, doc: Document) -> dict:
        """Extract metadata from a document."""
        ...