from dataclasses import dataclass
from typing import Protocol

from rag.chunkers.base import Chunk


@dataclass
class SearchResult:
    chunk: Chunk
    score: float


class VectorStoreProtocol(Protocol):
    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None: ...
    def search(self, query_vector: list[float], k: int) -> list[SearchResult]: ...
    def delete(self, ids: list[str]) -> None: ...
    def count(self) -> int: ...
