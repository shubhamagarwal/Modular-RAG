import chromadb

from rag.chunkers.base import Chunk
from rag.config import settings
from rag.logger import get_logger

from .base import SearchResult

log = get_logger("stores.chroma")


class ChromaStore:
    def __init__(self) -> None:
        log.info("Initialising ChromaDB at '%s' (collection: '%s')", settings.chroma_persist_dir, settings.chroma_collection)
        self._client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=settings.chroma_collection,
            metadata={"hnsw:space": "cosine"},
        )
        log.info("ChromaDB ready — %d chunks already in collection", self._collection.count())

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        log.info("Upserting %d chunks into ChromaDB", len(chunks))
        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=[c.content for c in chunks],
            metadatas=[
                {
                    "file_path": c.file_path,
                    "language": c.language,
                    "repo_root": c.repo_root,
                    "chunk_index": c.chunk_index,
                    "start_line": c.start_line,
                    "end_line": c.end_line,
                }
                for c in chunks
            ],
        )

    def search(self, query_vector: list[float], k: int = 10) -> list[SearchResult]:
        log.info("Semantic search — top %d from %d chunks", k, self._collection.count())
        results = self._collection.query(
            query_embeddings=[query_vector],
            n_results=min(k, self._collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        output: list[SearchResult] = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            chunk = Chunk(
                content=doc,
                file_path=meta["file_path"],
                language=meta["language"],
                repo_root=meta["repo_root"],
                chunk_index=meta["chunk_index"],
                start_line=meta["start_line"],
                end_line=meta["end_line"],
            )
            output.append(SearchResult(chunk=chunk, score=1.0 - dist))
        if output:
            log.info("Semantic search returned %d results (top score=%.4f)", len(output), output[0].score)
        return output

    def delete(self, ids: list[str]) -> None:
        self._collection.delete(ids=ids)

    def count(self) -> int:
        return self._collection.count()
