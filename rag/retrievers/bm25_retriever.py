from __future__ import annotations

import re

from rank_bm25 import BM25Plus

from rag.chunkers.base import Chunk
from rag.logger import get_logger
from rag.stores.base import SearchResult

log = get_logger("retrievers.bm25")


def _tokenize(text: str) -> list[str]:
    # Split on non-alphanumeric chars AND on underscore/camelCase boundaries
    # so "authenticate_user" → ["authenticate", "user"]
    words = re.findall(r"[a-zA-Z0-9]+", text)
    tokens = []
    for word in words:
        # split camelCase: "myFunction" → ["my", "Function"]
        parts = re.sub(r"([a-z])([A-Z])", r"\1 \2", word).split()
        # split snake_case already handled by the findall above
        tokens.extend(p.lower() for p in parts)
    return tokens


class BM25Retriever:
    """
    In-memory BM25 index over a fixed corpus of chunks.
    Build once after ingestion, reuse for all queries.
    """

    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks
        log.info("Building BM25Plus index over %d chunks", len(chunks))
        tokenized = [_tokenize(c.content) for c in chunks]
        self._index = BM25Plus(tokenized)
        log.info("BM25 index ready")

    def search(self, query: str, k: int = 10) -> list[SearchResult]:
        tokens = _tokenize(query)
        log.info("BM25 search ┌─────────────────────────────────")
        log.info("BM25 search │ Query      : %s", query)
        log.info("BM25 search │ Tokens     : %s", tokens)
        log.info("BM25 search │ Corpus size: %d chunks", len(self._chunks))

        scores: list[float] = self._index.get_scores(tokens).tolist()

        if not any(scores):
            log.info("BM25 search │ Result     : no matching chunks")
            log.info("BM25 search └─────────────────────────────────")
            return []

        max_score = max(scores)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        results = [
            SearchResult(chunk=self._chunks[i], score=score)
            for i, score in ranked[:k]
            if score >= max_score * 0.1
        ]

        log.info("BM25 search │ Threshold  : %.4f (10%% of max=%.4f)", max_score * 0.1, max_score)
        log.info("BM25 search │ Results    : %d chunks passed threshold", len(results))
        log.info("BM25 search ├─────────────────────────────────")
        for i, r in enumerate(results):
            preview = r.chunk.content[:80].replace("\n", " ").strip()
            log.info(
                "BM25 search │ [%d] score=%-8.4f  %s  lines %d-%d",
                i + 1, r.score, r.chunk.file_path, r.chunk.start_line, r.chunk.end_line,
            )
            log.info("BM25 search │     preview: %s%s", preview, "…" if len(r.chunk.content) > 80 else "")
        log.info("BM25 search └─────────────────────────────────")
        return results
