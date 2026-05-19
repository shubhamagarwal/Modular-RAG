from pathlib import Path

import pathspec

from rag.logger import get_logger

from .base import Document
from .code_loader import EXTENSION_LANGUAGE, CodeLoader
from .markdown_loader import MarkdownLoader

log = get_logger("loaders.directory")

_MARKDOWN_EXTS = {".md", ".mdx"}
_TEXT_EXTS = {".txt", ".env.example"}
_SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv", "dist", "build", ".chroma"}


def _load_gitignore(root: Path) -> pathspec.PathSpec:
    gitignore = root / ".gitignore"
    if gitignore.exists():
        return pathspec.PathSpec.from_lines("gitignore", gitignore.read_text().splitlines())
    return pathspec.PathSpec.from_lines("gitignore", [])


class DirectoryLoader:
    def __init__(self) -> None:
        self._code = CodeLoader()
        self._markdown = MarkdownLoader()

    def load(self, root: str | Path) -> list[Document]:
        root = Path(root).resolve()
        log.info("Scanning directory: %s", root)
        spec = _load_gitignore(root)
        documents: list[Document] = []
        skipped = 0

        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if any(part in _SKIP_DIRS for part in path.parts):
                skipped += 1
                continue
            rel = path.relative_to(root)
            if spec.match_file(str(rel)):
                log.debug("Skipped (gitignore): %s", rel)
                skipped += 1
                continue

            ext = path.suffix.lower()
            if ext in EXTENSION_LANGUAGE:
                docs = self._code.load(path, repo_root=str(root))
                log.debug("Loaded code file: %s (%s)", rel, docs[0].language if docs else "?")
                documents.extend(docs)
            elif ext in _MARKDOWN_EXTS:
                docs = self._markdown.load(path, repo_root=str(root))
                log.debug("Loaded markdown file: %s", rel)
                documents.extend(docs)
            else:
                skipped += 1

        log.info("Directory scan complete — %d files loaded, %d skipped", len(documents), skipped)
        return documents
