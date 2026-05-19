import tiktoken

from rag.logger import get_logger
from rag.stores.base import SearchResult

log = get_logger("context.builder")

_ENCODER = tiktoken.get_encoding("cl100k_base")


def _token_len(text: str) -> int:
    return len(_ENCODER.encode(text))


class ContextBuilder:
    def __init__(self, max_tokens: int = 6000) -> None:
        self._max_tokens = max_tokens

    def build(self, results: list[SearchResult]) -> str:
        log.info("Building context from %d chunks (budget=%d tokens)", len(results), self._max_tokens)
        seen: set[str] = set()
        blocks: list[str] = []
        total = 0

        for result in results:
            chunk = result.chunk
            dedup_key = f"{chunk.file_path}:{chunk.start_line}"
            if dedup_key in seen:
                log.debug("Skipping duplicate chunk: %s line %d", chunk.file_path, chunk.start_line)
                continue
            seen.add(dedup_key)

            header = f"### {chunk.file_path}  (lines {chunk.start_line}–{chunk.end_line})"
            fenced = f"```{chunk.language}\n{chunk.content}\n```"
            block = f"{header}\n{fenced}"

            block_tokens = _token_len(block)
            if total + block_tokens > self._max_tokens:
                log.info("Token budget reached at %d/%d — stopping after %d blocks", total, self._max_tokens, len(blocks))
                break

            blocks.append(block)
            total += block_tokens
            log.debug("  Added: %s lines %d-%d (%d tokens)", chunk.file_path, chunk.start_line, chunk.end_line, block_tokens)

        log.info("Context built — %d blocks, %d total tokens", len(blocks), total)
        return "\n\n".join(blocks)
