from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator

from rag.chunkers.base import Chunk
from rag.chunkers.code_chunker import CodeChunker
from rag.config import settings
from rag.context.context_builder import ContextBuilder
from rag.embedders.openai_embedder import OpenAIEmbedder
from rag.llm.openai_llm import OpenAILLM, _SYSTEM_PROMPT
from rag.loaders.directory_loader import DirectoryLoader
from rag.rerankers.cross_encoder_reranker import CrossEncoderReranker
from rag.retrievers.bm25_retriever import BM25Retriever
from rag.retrievers.hybrid_retriever import HybridRetriever
from rag.stores.chroma_store import ChromaStore


@dataclass
class IngestResult:
    files_processed: int
    chunks_added: int
    duration_s: float


@dataclass
class Citation:
    file_path: str
    start_line: int
    end_line: int


@dataclass
class AnswerChunk:
    text: str
    citations: list[Citation]


class RAGPipeline:
    def __init__(self) -> None:
        self._loader = DirectoryLoader()
        self._chunker = CodeChunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        self._embedder = OpenAIEmbedder()
        self._store = ChromaStore()
        self._reranker = CrossEncoderReranker()
        self._context = ContextBuilder(max_tokens=settings.context_max_tokens)
        self._llm = OpenAILLM()

        # BM25 index is built after ingest; starts as None
        self._bm25: BM25Retriever | None = None
        self._retriever = HybridRetriever(
            self._store,
            top_k=settings.retrieval_top_k,
            bm25=self._bm25,
        )

    async def ingest(self, path: str | Path) -> IngestResult:
        start = time.perf_counter()

        documents = self._loader.load(path)
        chunks = self._chunker.chunk(documents)

        if chunks:
            texts = [c.content for c in chunks]
            embeddings = await self._embedder.embed(texts)
            self._store.add(chunks, embeddings)
            self._rebuild_bm25(chunks)

        return IngestResult(
            files_processed=len(documents),
            chunks_added=len(chunks),
            duration_s=round(time.perf_counter() - start, 2),
        )

    def _rebuild_bm25(self, chunks: list[Chunk]) -> None:
        self._bm25 = BM25Retriever(chunks)
        self._retriever = HybridRetriever(
            self._store,
            top_k=settings.retrieval_top_k,
            bm25=self._bm25,
        )

    async def query(self, question: str) -> AsyncIterator[AnswerChunk]:
        query_embeddings = await self._embedder.embed([question])
        query_vector = query_embeddings[0]

        candidates = self._retriever.search(query_vector, raw_query=question)
        ranked = self._reranker.rerank(question, candidates, top_k=settings.rerank_top_k)

        context = self._context.build(ranked)
        citations = [
            Citation(
                file_path=r.chunk.file_path,
                start_line=r.chunk.start_line,
                end_line=r.chunk.end_line,
            )
            for r in ranked
        ]

        async def _stream() -> AsyncIterator[AnswerChunk]:
            async for token in self._llm.generate(_SYSTEM_PROMPT, question, context):
                yield AnswerChunk(text=token, citations=citations)

        return _stream()
