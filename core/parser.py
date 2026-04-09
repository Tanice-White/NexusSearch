from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Iterable

from core.schemas import ParsedDocument, ParsedSection

CODE_SUFFIX_TO_LANGUAGE = {
    ".py": "python",
    ".java": "java",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".kt": "kotlin",
}

TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".html", ".htm", ".json", ".yml", ".yaml"}
BINARY_DOC_SUFFIXES = {".pdf", ".docx"}


class NexusParser:
    """Parse heterogeneous local files into structured documents."""

    SUPPORTED_SUFFIXES = TEXT_SUFFIXES | BINARY_DOC_SUFFIXES | set(CODE_SUFFIX_TO_LANGUAGE)

    def __init__(self) -> None:
        self.converter = None
        self._tree_sitter_available = False
        try:
            from docling.document_converter import DocumentConverter

            self.converter = DocumentConverter()
        except Exception:
            self.converter = None

        try:
            from tree_sitter_languages import get_parser

            self._get_tree_sitter_parser = get_parser
            self._tree_sitter_available = True
        except Exception:
            self._get_tree_sitter_parser = None

    def is_supported(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in self.SUPPORTED_SUFFIXES

    def iter_files(self, source: Path) -> Iterable[Path]:
        if source.is_file():
            if self.is_supported(source):
                yield source
            return

        for path in source.rglob("*"):
            if path.is_file() and self.is_supported(path):
                yield path

    def parse(self, file_path: Path) -> ParsedDocument:
        suffix = file_path.suffix.lower()
        metadata = self._file_metadata(file_path)
        if suffix in CODE_SUFFIX_TO_LANGUAGE:
            return self._parse_code(file_path, CODE_SUFFIX_TO_LANGUAGE[suffix], metadata)

        if suffix in TEXT_SUFFIXES:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
            return ParsedDocument(
                source_path=str(file_path),
                content=text,
                file_type="document",
                parser_name="plain",
                sections=self._parse_markdown_sections(text),
                metadata=metadata,
            )

        if self.converter is not None:
            result = self.converter.convert(str(file_path))
            markdown = result.document.export_to_markdown()
            return ParsedDocument(
                source_path=str(file_path),
                content=markdown,
                file_type="document",
                parser_name="docling",
                sections=self._parse_markdown_sections(markdown),
                metadata=metadata,
            )

        return ParsedDocument(
            source_path=str(file_path),
            content=file_path.read_text(encoding="utf-8", errors="ignore"),
            file_type="document",
            parser_name="fallback",
            metadata=metadata,
        )

    def _parse_code(
        self,
        file_path: Path,
        language: str,
        metadata: dict[str, object],
    ) -> ParsedDocument:
        source = file_path.read_text(encoding="utf-8", errors="ignore")
        if self._tree_sitter_available:
            sections = self._parse_code_with_tree_sitter(source, language)
            if sections:
                return ParsedDocument(
                    source_path=str(file_path),
                    content=source,
                    file_type="code",
                    language=language,
                    parser_name="tree-sitter",
                    sections=sections,
                    metadata=metadata,
                )

        return ParsedDocument(
            source_path=str(file_path),
            content=source,
            file_type="code",
            language=language,
            parser_name="regex-fallback",
            sections=self._parse_code_with_regex(source, language),
            metadata=metadata,
        )

    def _parse_markdown_sections(self, markdown: str) -> list[ParsedSection]:
        lines = markdown.splitlines()
        sections: list[ParsedSection] = []
        headers: list[str] = []
        paragraph_buffer: list[str] = []
        idx = 0

        def flush_paragraph() -> None:
            text = "\n".join(paragraph_buffer).strip()
            if not text:
                paragraph_buffer.clear()
                return
            title = " / ".join(headers) if headers else "ROOT"
            sections.append(ParsedSection(title=title, text=text, kind="section"))
            paragraph_buffer.clear()

        while idx < len(lines):
            line = lines[idx]
            heading_match = re.match(r"^(#{1,6})\s+(.*)$", line.strip())
            if heading_match:
                flush_paragraph()
                level = len(heading_match.group(1))
                title = heading_match.group(2).strip()
                headers[:] = headers[: level - 1]
                headers.append(title)
                idx += 1
                continue

            if self._is_table_line(line):
                flush_paragraph()
                table_lines = [line]
                idx += 1
                while idx < len(lines) and self._is_table_line(lines[idx]):
                    table_lines.append(lines[idx])
                    idx += 1
                title = " / ".join(headers) if headers else "ROOT"
                table_text = self._compose_table_context(
                    lines=lines,
                    table_start=idx - len(table_lines),
                    table_end=idx,
                    table_lines=table_lines,
                )
                sections.append(
                    ParsedSection(
                        title=f"{title} / TABLE",
                        text=table_text,
                        kind="table",
                        metadata={"contains_table": True},
                    )
                )
                continue

            paragraph_buffer.append(line)
            idx += 1

        flush_paragraph()
        if not sections and markdown.strip():
            sections.append(ParsedSection(title="ROOT", text=markdown.strip(), kind="section"))
        return sections

    def _parse_code_with_tree_sitter(
        self,
        source: str,
        language: str,
    ) -> list[ParsedSection]:
        if self._get_tree_sitter_parser is None:
            return []

        try:
            parser = self._get_tree_sitter_parser(language)
        except Exception:
            return []

        try:
            tree = parser.parse(source.encode("utf-8"))
        except Exception:
            return []

        nodes = self._interesting_node_types(language)
        sections: list[ParsedSection] = []
        for node in self._walk(tree.root_node):
            if node.type not in nodes:
                continue
            snippet = source.encode("utf-8")[node.start_byte : node.end_byte].decode(
                "utf-8", errors="ignore"
            )
            title = self._extract_symbol_title(snippet, node.type)
            if not snippet.strip():
                continue
            sections.append(
                ParsedSection(
                    title=title,
                    text=snippet,
                    kind="code-symbol",
                    metadata={
                        "node_type": node.type,
                        "start_line": node.start_point[0] + 1,
                        "end_line": node.end_point[0] + 1,
                    },
                )
            )

        return sections

    def _parse_code_with_regex(self, source: str, language: str) -> list[ParsedSection]:
        pattern = self._regex_for_language(language)
        if pattern is None:
            return [ParsedSection(title="module", text=source, kind="code-module")]

        matches = list(pattern.finditer(source))
        if not matches:
            return [ParsedSection(title="module", text=source, kind="code-module")]

        sections: list[ParsedSection] = []
        for idx, match in enumerate(matches):
            start = match.start()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(source)
            snippet = source[start:end].strip()
            title = match.group("name") if "name" in pattern.groupindex else f"symbol_{idx}"
            sections.append(
                ParsedSection(
                    title=title,
                    text=snippet,
                    kind="code-symbol",
                    metadata={"parser": "regex", "language": language},
                )
            )
        return sections

    @staticmethod
    def _interesting_node_types(language: str) -> set[str]:
        default = {
            "class_definition",
            "function_definition",
            "method_definition",
        }
        if language == "java":
            return default | {"class_declaration", "method_declaration", "interface_declaration"}
        if language == "python":
            return default
        return default

    @staticmethod
    def _extract_symbol_title(snippet: str, node_type: str) -> str:
        first_line = snippet.strip().splitlines()[0] if snippet.strip() else node_type
        compact = re.sub(r"\s+", " ", first_line)
        return compact[:120]

    @staticmethod
    def _walk(node):  # noqa: ANN001, ANN205
        yield node
        for child in node.children:
            yield from NexusParser._walk(child)

    @staticmethod
    def _regex_for_language(language: str):
        if language == "python":
            return re.compile(
                r"^(?:async\s+def|def|class)\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)",
                re.MULTILINE,
            )
        if language == "java":
            return re.compile(
                r"^\s*(?:public|protected|private|static|final|\s)+"
                r"(?:class|interface|enum|[\w<>\[\]]+\s+)+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\(",
                re.MULTILINE,
            )
        if language in {"javascript", "typescript", "tsx"}:
            return re.compile(
                r"^\s*(?:export\s+)?(?:async\s+)?(?:function|class)\s+"
                r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)",
                re.MULTILINE,
            )
        return None

    @staticmethod
    def _is_table_line(line: str) -> bool:
        stripped = line.strip()
        if not stripped.startswith("|"):
            return False
        return stripped.count("|") >= 2

    @staticmethod
    def _compose_table_context(
        lines: list[str],
        table_start: int,
        table_end: int,
        table_lines: list[str],
    ) -> str:
        before = NexusParser._nearest_context_paragraph(lines, table_start - 1, -1)
        after = NexusParser._nearest_context_paragraph(lines, table_end, 1)
        parts = []
        if before:
            parts.append(f"Table context before:\n{before}")
        parts.append("Table content:\n" + "\n".join(table_lines))
        if after:
            parts.append(f"Table context after:\n{after}")
        return "\n\n".join(parts)

    @staticmethod
    def _nearest_context_paragraph(lines: list[str], start: int, step: int) -> str:
        collected: list[str] = []
        idx = start
        while 0 <= idx < len(lines):
            candidate = lines[idx].strip()
            if not candidate:
                if collected:
                    break
                idx += step
                continue
            if candidate.startswith("#") or NexusParser._is_table_line(candidate):
                if collected:
                    break
                idx += step
                continue
            if step < 0:
                collected.insert(0, candidate)
            else:
                collected.append(candidate)
            idx += step
        return " ".join(collected).strip()

    @staticmethod
    def _file_metadata(file_path: Path) -> dict[str, object]:
        stat = file_path.stat()
        modified = datetime.fromtimestamp(stat.st_mtime)
        created = datetime.fromtimestamp(stat.st_ctime)
        return {
            "source_name": file_path.name,
            "source_suffix": file_path.suffix.lower(),
            "source_stem": file_path.stem,
            "file_size_bytes": stat.st_size,
            "created_at": created.isoformat(timespec="seconds"),
            "updated_at": modified.isoformat(timespec="seconds"),
            "updated_year": modified.year,
        }
