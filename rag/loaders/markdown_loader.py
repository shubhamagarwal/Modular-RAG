from pathlib import Path

from .base import Document


class MarkdownLoader:
    def load(self, path: Path, repo_root: str = "") -> list[Document]:
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return []
        return [
            Document(
                content=content,
                file_path=str(path),
                language="markdown",
                repo_root=repo_root or str(path.parent),
            )
        ]
