from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from typing import Callable

import numpy as np

from core.schemas import DocumentChunk, ParsedDocument, ParsedSection

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？.!?])\s+|\n+")
TOKEN_RE = re.compile(r"[\u4e00-\u9fff]+|[a-zA-Z0-9_]+")


@dataclass(slots=True)
class HeaderAwareChunker:
    chunk_size: int = 900
    chunk_overlap: int = 120

    def chunk_document(self, document: ParsedDocument) -> list[DocumentChunk]:
        sections = document.sections or self._sections_from_markdown(document.content)
        output: list[DocumentChunk] = []

        for section_index, section in enumerate(sections):
            section_chunks = self._split_by_size(section.text.strip())
            for chunk_index, chunk_text in enumerate(section_chunks):
                normalized = " ".join(chunk_text.split())
                if not normalized:
                    continue
                output.append(
                    self._build_chunk(
                        document=document,
                        section=section,
                        section_index=section_index,
                        chunk_index=chunk_index,
                        text=normalized,
                    )
                )
        return output

    def _build_chunk(
        self,
        document: ParsedDocument,
        section: ParsedSection,
        section_index: int,
        chunk_index: int,
        text: str,
    ) -> DocumentChunk:
        chunk_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"{document.source_path}|{section.title}|{section_index}|{chunk_index}",
            )
        )
        metadata = dict(document.metadata)
        metadata.update(section.metadata)
        return DocumentChunk(
            id=chunk_id,
            text=text,
            source_path=document.source_path,
            header_path=section.title,
            section_index=section_index,
            chunk_index=chunk_index,
            token_count=self._estimate_tokens(text),
            file_type=document.file_type,
            language=document.language,
            chunk_kind=section.kind,
            metadata=metadata,
        )

    def _sections_from_markdown(self, markdown: str) -> list[ParsedSection]:
        lines = markdown.splitlines()
        sections: list[ParsedSection] = []
        headers: list[str] = []
        buffer: list[str] = []

        def flush() -> None:
            if not buffer:
                return
            title = " / ".join(headers) if headers else "ROOT"
            sections.append(ParsedSection(title=title, text="\n".join(buffer).strip()))
            buffer.clear()

        for line in lines:
            match = HEADING_RE.match(line.strip())
            if match:
                flush()
                level = len(match.group(1))
                title = match.group(2).strip()
                headers[:] = headers[: level - 1]
                headers.append(title)
            else:
                buffer.append(line)

        flush()
        if not sections and markdown.strip():
            sections.append(ParsedSection(title="ROOT", text=markdown.strip()))
        return sections

    def _split_by_size(self, text: str) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text]

        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        chunks: list[str] = []
        current = ""

        for paragraph in paragraphs:
            units = (
                SENTENCE_SPLIT_RE.split(paragraph)
                if len(paragraph) > self.chunk_size
                else [paragraph]
            )
            for unit in units:
                unit = unit.strip()
                if not unit:
                    continue
                candidate = f"{current}\n{unit}".strip() if current else unit
                if len(candidate) <= self.chunk_size:
                    current = candidate
                    continue

                if current:
                    chunks.append(current)
                    overlap = current[-self.chunk_overlap :] if self.chunk_overlap > 0 else ""
                    current = f"{overlap}\n{unit}".strip()
                else:
                    chunks.extend(self._hard_split(unit))
                    current = ""

        if current:
            chunks.append(current)
        return chunks

    def _hard_split(self, text: str) -> list[str]:
        stride = max(self.chunk_size - self.chunk_overlap, 1)
        return [text[i : i + self.chunk_size] for i in range(0, len(text), stride)]

    def _estimate_tokens(self, text: str) -> int:
        return len(TOKEN_RE.findall(text))


@dataclass(slots=True)
class SemanticChunker:
    embed_texts: Callable[[list[str]], list[list[float]]]
    chunk_size: int = 900
    chunk_overlap: int = 120
    similarity_threshold: float = 0.72
    min_sentences: int = 2

    def chunk_document(self, document: ParsedDocument) -> list[DocumentChunk]:
        if document.file_type == "code":
            return HeaderAwareChunker(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            ).chunk_document(document)

        sections = document.sections or [ParsedSection(title="ROOT", text=document.content)]
        output: list[DocumentChunk] = []
        for section_index, section in enumerate(sections):
            chunks = self._semantic_split(section.text)
            for chunk_index, chunk_text in enumerate(chunks):
                normalized = " ".join(chunk_text.split())
                if not normalized:
                    continue
                chunk_id = str(
                    uuid.uuid5(
                        uuid.NAMESPACE_URL,
                        (
                            f"{document.source_path}|{section.title}|"
                            f"{section_index}|semantic|{chunk_index}"
                        ),
                    )
                )
                metadata = dict(document.metadata)
                metadata.update(section.metadata)
                output.append(
                    DocumentChunk(
                        id=chunk_id,
                        text=normalized,
                        source_path=document.source_path,
                        header_path=section.title,
                        section_index=section_index,
                        chunk_index=chunk_index,
                        token_count=len(TOKEN_RE.findall(normalized)),
                        file_type=document.file_type,
                        language=document.language,
                        chunk_kind=f"{section.kind}:semantic",
                        metadata=metadata,
                    )
                )
        return output

    def _semantic_split(self, text: str) -> list[str]:
        sentences = [segment.strip() for segment in SENTENCE_SPLIT_RE.split(text) if segment.strip()]
        if len(sentences) <= self.min_sentences:
            return HeaderAwareChunker(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )._split_by_size(text)

        try:
            vectors = np.asarray(self.embed_texts(sentences), dtype=np.float32)
            similarities = self._adjacent_similarities(vectors)
        except Exception:
            return HeaderAwareChunker(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )._split_by_size(text)

        chunks: list[str] = []
        current: list[str] = []
        for idx, sentence in enumerate(sentences):
            current.append(sentence)
            should_split = False
            if idx < len(similarities):
                enough_sentences = len(current) >= self.min_sentences
                current_len = len(" ".join(current))
                should_split = enough_sentences and (
                    similarities[idx] < self.similarity_threshold
                    or current_len >= self.chunk_size
                )
            if should_split:
                chunks.append(" ".join(current))
                current = self._carry_overlap(current)

        if current:
            chunks.append(" ".join(current))
        return [chunk for chunk in chunks if chunk.strip()]

    def _carry_overlap(self, sentences: list[str]) -> list[str]:
        if self.chunk_overlap <= 0:
            return []
        kept: list[str] = []
        total = 0
        for sentence in reversed(sentences):
            total += len(sentence)
            kept.append(sentence)
            if total >= self.chunk_overlap:
                break
        return list(reversed(kept))

    @staticmethod
    def _adjacent_similarities(vectors: np.ndarray) -> list[float]:
        normalized = vectors / np.clip(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-8, None)
        similarities: list[float] = []
        for idx in range(len(normalized) - 1):
            similarities.append(float(np.dot(normalized[idx], normalized[idx + 1])))
        return similarities
