from __future__ import annotations

from rag.logger import get_logger
from rag.stores.base import SearchResult

log = get_logger("rerankers.cross_encoder")

_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    def __init__(self) -> None:
        self._model = None  # lazy-loaded to avoid slow import at startup

    def _get_model(self):
        if self._model is None:
            log.info("Loading cross-encoder model: %s (first run — downloading if needed)", _MODEL_NAME)
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(_MODEL_NAME)
            log.info("Cross-encoder model loaded")
        return self._model

    def rerank(self, query: str, results: list[SearchResult], top_k: int = 4) -> list[SearchResult]:
        if not results:
            log.info("Reranker — no candidates to rerank")
            return []

        log.info("Reranking %d candidates → keeping top %d", len(results), top_k)
        model = self._get_model()
        pairs = [(query, r.chunk.content) for r in results]
        scores: list[float] = model.predict(pairs).tolist()

        scored = sorted(zip(scores, results), key=lambda x: x[0], reverse=True)
        reranked = [r for _, r in scored[:top_k]]

        log.info("Reranking complete — top %d chunks:", len(reranked))
        for i, (score, r) in enumerate(scored[:top_k]):
            log.info("  [%d] score=%.4f  %s lines %d-%d", i + 1, score, r.chunk.file_path, r.chunk.start_line, r.chunk.end_line)
        return reranked
