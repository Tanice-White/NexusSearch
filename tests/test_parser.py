from pathlib import Path

from core.parser import NexusParser


def test_parser_treats_python_file_as_code(tmp_path: Path) -> None:
    source = tmp_path / "demo.py"
    source.write_text(
        "class Demo:\n"
        "    def run(self):\n"
        "        return 'ok'\n",
        encoding="utf-8",
    )

    parser = NexusParser()
    document = parser.parse(source)

    assert document.file_type == "code"
    assert document.language == "python"
    assert document.sections
