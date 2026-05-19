from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass
class Document:
    content: str
    file_path: str
    language: str
    repo_root: str
    doc_id: str = field(default="")

    def __post_init__(self) -> None:
        if not self.doc_id:
            import hashlib
            self.doc_id = hashlib.sha256(self.file_path.encode()).hexdigest()[:16]


class LoaderProtocol(Protocol):
    def load(self, path: Path) -> list[Document]: ...
