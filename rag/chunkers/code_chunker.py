import tiktoken

from rag.loaders.base import Document
from rag.logger import get_logger

from .base import Chunk

log = get_logger("chunkers.code")

_ENCODER = tiktoken.get_encoding("cl100k_base")


def _token_len(text: str) -> int:
    return len(_ENCODER.encode(text))


def _split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Recursive character splitter that respects newlines then falls back to chars."""
    if _token_len(text) <= chunk_size:
        return [text]

    separators = ["\n\n", "\n", " ", ""]
    for sep in separators:
        if sep not in text:
            continue
        parts = text.split(sep)
        chunks: list[str] = []
        current = ""
        for part in parts:
            candidate = (current + sep + part).lstrip(sep) if current else part
            if _token_len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                # carry overlap
                overlap_text = sep.join(current.split(sep)[-3:]) if sep else current[-chunk_overlap * 4:]
                current = (overlap_text + sep + part).lstrip(sep) if overlap_text else part
        if current:
            chunks.append(current)
        return chunks

    # last resort: hard-split on characters
    tokens = _ENCODER.encode(text)
    result = []
    i = 0
    while i < len(tokens):
        window = tokens[i: i + chunk_size]
        result.append(_ENCODER.decode(window))
        i += chunk_size - chunk_overlap
    return result


class CodeChunker:
    def __init__(self, chunk_size: int = 1200, chunk_overlap: int = 200) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, documents: list[Document]) -> list[Chunk]:
        log.info(
            "Chunking %d documents (chunk_size=%d, overlap=%d)",
            len(documents), self.chunk_size, self.chunk_overlap,
        )
        chunks: list[Chunk] = []
        for doc in documents:
            pieces = _split_text(doc.content, self.chunk_size, self.chunk_overlap)
            line_cursor = 1
            for idx, piece in enumerate(pieces):
                line_count = piece.count("\n") + 1
                chunks.append(
                    Chunk(
                        content=piece,
                        file_path=doc.file_path,
                        language=doc.language,
                        repo_root=doc.repo_root,
                        chunk_index=idx,
                        start_line=line_cursor,
                        end_line=line_cursor + line_count - 1,
                    )
                )
                line_cursor += line_count
            log.debug("  %s → %d chunks", doc.file_path, len(pieces))
        log.info("Chunking complete — %d total chunks from %d documents", len(chunks), len(documents))
        return chunks
