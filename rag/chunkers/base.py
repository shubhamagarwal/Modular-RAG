from dataclasses import dataclass
from typing import Protocol

from rag.loaders.base import Document


@dataclass
class Chunk:
    content: str
    file_path: str
    language: str
    repo_root: str
    chunk_index: int
    start_line: int
    end_line: int
    chunk_id: str = ""

    def __post_init__(self) -> None:
        if not self.chunk_id:
            import hashlib
            key = f"{self.file_path}:{self.chunk_index}"
            self.chunk_id = hashlib.sha256(key.encode()).hexdigest()[:16]


class ChunkerProtocol(Protocol):
    def chunk(self, documents: list[Document]) -> list[Chunk]: ...
