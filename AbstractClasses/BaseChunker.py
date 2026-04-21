from abc import ABC

class BaseChunker(ABC):
    def chunk(self, paper_text: str) -> list[str]: ...