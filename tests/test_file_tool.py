from pathlib import Path

import pytest

from app.tools.file_tool import FileTool


@pytest.fixture()
def upload_dir(temp_env):
    return temp_env["upload_dir"]


def test_file_tool_reads_csv(upload_dir):
    csv_path = upload_dir / "data.csv"
    csv_path.write_text("name,score\nAlice,90\nBob,85\n")

    tool = FileTool()
    result = tool.run(filename="data.csv")
    assert result.success is True
    assert result.output["columns"] == ["name", "score"]
    assert result.output["row_count"] == 2


def test_file_tool_blocks_path_traversal(upload_dir):
    tool = FileTool()
    result = tool.run(filename="../../etc/passwd")
    assert result.success is False
    assert "escapes" in result.error.lower() or "not found" in result.error.lower()


def test_file_tool_missing_file(upload_dir):
    tool = FileTool()
    result = tool.run(filename="does_not_exist.txt")
    assert result.success is False


def test_file_tool_rejects_unsupported_extension(upload_dir):
    bad_file = upload_dir / "script.exe"
    bad_file.write_bytes(b"not really an exe")
    tool = FileTool()
    result = tool.run(filename="script.exe")
    assert result.success is False
    assert "unsupported" in result.error.lower()
