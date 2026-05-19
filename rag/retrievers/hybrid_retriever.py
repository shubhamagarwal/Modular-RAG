from __future__ import annotations

from rag.logger import get_logger
from rag.stores.base import SearchResult, VectorStoreProtocol

log = get_logger("retrievers.hybrid")


def _rrf(
    rankings: list[list[str]],
    weights: list[float] | None = None,
    k: int = 60,
) -> dict[str, float]:
    """Weighted Reciprocal Rank Fusion — returns chunk_id → fused_score mapping."""
    if weights is None:
        weights = [1.0] * len(rankings)
    scores: dict[str, float] = {}
    for ranked_list, weight in zip(rankings, weights):
        for rank, chunk_id in enumerate(ranked_list):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + weight / (rank + k)
    return scores


class HybridRetriever:
    """
    Fuses semantic vector search with BM25 keyword search via Reciprocal Rank Fusion.
    Falls back to semantic-only if no BM25 retriever is provided.
    """

    def __init__(
        self,
        store: VectorStoreProtocol,
        top_k: int = 10,
        bm25=None,  # BM25Retriever | None
    ) -> None:
        self._store = store
        self._top_k = top_k
        self._bm25 = bm25

    def search(self, query_vector: list[float], raw_query: str = "") -> list[SearchResult]:
        log.info("HybridRetriever ┌──────────────────────────────────────")
        log.info("HybridRetriever │ Mode : %s", "semantic + BM25" if (self._bm25 and raw_query) else "semantic only")
        semantic_results = self._store.search(query_vector, k=self._top_k)

        if not self._bm25 or not raw_query:
            log.info("HybridRetriever │ Returning %d semantic-only results", len(semantic_results))
            log.info("HybridRetriever └──────────────────────────────────────")
            return semantic_results

        bm25_results = self._bm25.search(raw_query, k=self._top_k)

        # Build lookup: chunk_id → SearchResult
        result_map: dict[str, SearchResult] = {}
        for r in semantic_results + bm25_results:
            result_map[r.chunk.chunk_id] = r

        semantic_ids = [r.chunk.chunk_id for r in semantic_results]
        bm25_ids     = [r.chunk.chunk_id for r in bm25_results]

        # BM25 weight > semantic for code: exact identifier matches matter more
        rrf_scores = _rrf([semantic_ids, bm25_ids], weights=[1.0, 1.5])
        fused_ids  = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)
        fused      = [result_map[cid] for cid in fused_ids if cid in result_map][: self._top_k]

        log.info("HybridRetriever │ Semantic candidates : %d", len(semantic_results))
        log.info("HybridRetriever │ BM25 candidates     : %d", len(bm25_results))
        log.info("HybridRetriever │ Unique after union  : %d", len(result_map))
        log.info("HybridRetriever │ RRF weights         : semantic=1.0  BM25=1.5")
        log.info("HybridRetriever ├── Fused ranking (top %d) ──────────────", len(fused))
        for i, cid in enumerate(fused_ids[: self._top_k]):
            r = result_map[cid]
            sem_rank = next((j + 1 for j, x in enumerate(semantic_results) if x.chunk.chunk_id == cid), "-")
            bm25_rank = next((j + 1 for j, x in enumerate(bm25_results)   if x.chunk.chunk_id == cid), "-")
            preview = r.chunk.content[:70].replace("\n", " ").strip()
            log.info(
                "HybridRetriever │ [%d] rrf=%.5f  sem_rank=%-3s  bm25_rank=%-3s  %s:%d-%d",
                i + 1, rrf_scores[cid], sem_rank, bm25_rank,
                r.chunk.file_path, r.chunk.start_line, r.chunk.end_line,
            )
            log.info("HybridRetriever │     %s%s", preview, "…" if len(r.chunk.content) > 70 else "")
        log.info("HybridRetriever └──────────────────────────────────────")
        return fused
