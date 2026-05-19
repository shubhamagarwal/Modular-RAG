import tempfile
from pathlib import Path

from rag.loaders.code_loader import CodeLoader
from rag.loaders.directory_loader import DirectoryLoader


def test_code_loader_python():
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write("def hello(): pass\n")
        path = Path(f.name)

    docs = CodeLoader().load(path)
    assert len(docs) == 1
    assert docs[0].language == "python"
    assert "def hello" in docs[0].content


def test_directory_loader_skips_git(tmp_path: Path):
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("git stuff")
    (tmp_path / "main.py").write_text("print('hello')")

    docs = DirectoryLoader().load(tmp_path)
    paths = [d.file_path for d in docs]
    assert all(".git" not in p for p in paths)
    assert any("main.py" in p for p in paths)


def test_directory_loader_respects_gitignore(tmp_path: Path):
    (tmp_path / ".gitignore").write_text("secret.py\n")
    (tmp_path / "secret.py").write_text("PASSWORD = 'hunter2'")
    (tmp_path / "app.py").write_text("print('app')")

    docs = DirectoryLoader().load(tmp_path)
    paths = [d.file_path for d in docs]
    assert not any("secret.py" in p for p in paths)
    assert any("app.py" in p for p in paths)
