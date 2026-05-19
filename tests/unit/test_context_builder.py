from rag.chunkers.base import Chunk
from rag.context.context_builder import ContextBuilder
from rag.stores.base import SearchResult


def _result(file: str, content: str, start: int = 1, end: int = 10) -> SearchResult:
    chunk = Chunk(
        content=content,
        file_path=file,
        language="python",
        repo_root="/tmp",
        chunk_index=0,
        start_line=start,
        end_line=end,
    )
    return SearchResult(chunk=chunk, score=0.9)


def test_builds_fenced_blocks():
    builder = ContextBuilder(max_tokens=2000)
    results = [_result("src/foo.py", "def foo(): pass")]
    ctx = builder.build(results)
    assert "src/foo.py" in ctx
    assert "```python" in ctx
    assert "def foo(): pass" in ctx


def test_deduplicates_same_location():
    builder = ContextBuilder(max_tokens=2000)
    r1 = _result("src/foo.py", "def foo(): pass", start=1, end=5)
    r2 = _result("src/foo.py", "def foo(): pass", start=1, end=5)
    ctx = builder.build([r1, r2])
    assert ctx.count("src/foo.py") == 1


def test_respects_token_limit():
    builder = ContextBuilder(max_tokens=50)
    results = [_result(f"file{i}.py", "x " * 200) for i in range(5)]
    ctx = builder.build(results)
    # With a 50-token budget only 0 or 1 blocks fit
    assert ctx.count("```python") <= 1
