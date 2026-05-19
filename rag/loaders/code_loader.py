from pathlib import Path

from .base import Document

EXTENSION_LANGUAGE: dict[str, str] = {
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript",
    ".jsx": "javascript",
    ".go": "go",
    ".java": "java",
    ".rs": "rust",
    ".rb": "ruby",
    ".cs": "csharp",
    ".cpp": "cpp",
    ".c": "c",
    ".h": "c",
    ".hpp": "cpp",
    ".kt": "kotlin",
    ".swift": "swift",
    ".sh": "bash",
}


class CodeLoader:
    def load(self, path: Path, repo_root: str = "") -> list[Document]:
        language = EXTENSION_LANGUAGE.get(path.suffix, "unknown")
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return []
        return [
            Document(
                content=content,
                file_path=str(path),
                language=language,
                repo_root=repo_root or str(path.parent),
            )
        ]
