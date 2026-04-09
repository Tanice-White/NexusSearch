from __future__ import annotations

import re
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
        if suffix in CODE_SUFFIX_TO_LANGUAGE:
            return self._parse_code(file_path, CODE_SUFFIX_TO_LANGUAGE[suffix])

        if suffix in TEXT_SUFFIXES:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
            return ParsedDocument(
                source_path=str(file_path),
                content=text,
                file_type="document",
                parser_name="plain",
            )

        if self.converter is not None:
            result = self.converter.convert(str(file_path))
            markdown = result.document.export_to_markdown()
            return ParsedDocument(
                source_path=str(file_path),
                content=markdown,
                file_type="document",
                parser_name="docling",
            )

        return ParsedDocument(
            source_path=str(file_path),
            content=file_path.read_text(encoding="utf-8", errors="ignore"),
            file_type="document",
            parser_name="fallback",
        )

    def _parse_code(self, file_path: Path, language: str) -> ParsedDocument:
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
                )

        return ParsedDocument(
            source_path=str(file_path),
            content=source,
            file_type="code",
            language=language,
            parser_name="regex-fallback",
            sections=self._parse_code_with_regex(source, language),
        )

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
