from pathlib import Path

from core.parser import NexusParser


def test_parser_adds_context_around_markdown_table(tmp_path: Path) -> None:
    source = tmp_path / "report.md"
    source.write_text(
        "# Revenue\n"
        "The following table summarizes quarterly revenue.\n\n"
        "| Quarter | Revenue |\n"
        "| --- | --- |\n"
        "| Q1 | 100 |\n"
        "| Q2 | 120 |\n\n"
        "The table shows continued growth in the first half.\n",
        encoding="utf-8",
    )

    parser = NexusParser()
    document = parser.parse(source)

    table_sections = [section for section in document.sections if section.kind == "table"]

    assert table_sections
    assert "Table context before" in table_sections[0].text
    assert "Table context after" in table_sections[0].text
    assert table_sections[0].metadata["contains_table"] is True
