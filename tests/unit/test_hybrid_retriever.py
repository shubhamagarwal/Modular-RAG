from unittest.mock import MagicMock

from rag.chunkers.base import Chunk
from rag.retrievers.bm25_retriever import BM25Retriever
from rag.retrievers.hybrid_retriever import HybridRetriever
from rag.stores.base import SearchResult


def _make_chunk(content: str, idx: int = 0) -> Chunk:
    return Chunk(
        content=content,
        file_path="f.py",
        language="python",
        repo_root="/tmp",
        chunk_index=idx,
        start_line=idx * 10 + 1,
        end_line=idx * 10 + 5,
    )


def _make_result(content: str, score: float = 0.8, idx: int = 0) -> SearchResult:
    return SearchResult(chunk=_make_chunk(content, idx), score=score)


def test_bm25_fused_reorders():
    """BM25 should surface the keyword-matching chunk even when semantic ranks it lower."""
    chunks = [
        _make_chunk("unrelated content here", idx=0),
        _make_chunk("def authenticate_user(token): ...", idx=1),
    ]
    store = MagicMock()
    store.search.return_value = [
        SearchResult(chunk=chunks[0], score=0.9),  # semantic top
        SearchResult(chunk=chunks[1], score=0.7),
    ]
    bm25 = BM25Retriever(chunks)
    retriever = HybridRetriever(store, top_k=10, bm25=bm25)
    results = retriever.search([0.1, 0.2], raw_query="authenticate user token")
    assert "authenticate" in results[0].chunk.content


def test_no_query_returns_semantic_order():
    store = MagicMock()
    expected = [_make_result("a"), _make_result("b")]
    store.search.return_value = expected
    retriever = HybridRetriever(store)
    results = retriever.search([0.1], raw_query="")
    assert results == expected


def test_no_bm25_falls_back_to_semantic():
    store = MagicMock()
    expected = [_make_result("a"), _make_result("b")]
    store.search.return_value = expected
    retriever = HybridRetriever(store, bm25=None)
    results = retriever.search([0.1], raw_query="some query")
    assert results == expected
