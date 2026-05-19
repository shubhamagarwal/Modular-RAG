from typing import Protocol

from rag.stores.base import SearchResult


class RerankerProtocol(Protocol):
    def rerank(self, query: str, results: list[SearchResult], top_k: int) -> list[SearchResult]: ...
