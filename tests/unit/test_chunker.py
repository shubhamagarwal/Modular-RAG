from rag.chunkers.code_chunker import CodeChunker
from rag.loaders.base import Document


def _make_doc(content: str, lang: str = "python") -> Document:
    return Document(content=content, file_path="test.py", language=lang, repo_root="/tmp")


def test_small_document_single_chunk():
    doc = _make_doc("def foo():\n    return 1\n")
    chunks = CodeChunker().chunk([doc])
    assert len(chunks) == 1
    assert "def foo" in chunks[0].content


def test_chunk_metadata():
    doc = _make_doc("line1\nline2\nline3\n")
    chunks = CodeChunker(chunk_size=10, chunk_overlap=2).chunk([doc])
    assert all(c.file_path == "test.py" for c in chunks)
    assert all(c.language == "python" for c in chunks)
    assert chunks[0].start_line == 1


def test_chunk_ids_unique():
    doc = _make_doc("a\n" * 500)
    chunks = CodeChunker(chunk_size=50, chunk_overlap=10).chunk([doc])
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
