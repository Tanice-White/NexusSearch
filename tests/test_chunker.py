from core.chunker import HeaderAwareChunker, SemanticChunker
from core.schemas import ParsedDocument, ParsedSection


def test_header_aware_chunker_keeps_section_metadata() -> None:
    chunker = HeaderAwareChunker(chunk_size=80, chunk_overlap=20)
    document = ParsedDocument(
        source_path="demo.md",
        content="",
        file_type="document",
        sections=[
            ParsedSection(
                title="Intro / Method",
                text="NexusSearch combines dense retrieval and sparse retrieval.",
                kind="section",
                metadata={"parser": "docling"},
            )
        ],
    )

    chunks = chunker.chunk_document(document)

    assert chunks
    assert chunks[0].source_path == "demo.md"
    assert chunks[0].header_path == "Intro / Method"
    assert chunks[0].metadata["parser"] == "docling"


def test_semantic_chunker_preserves_code_as_code_chunks() -> None:
    chunker = SemanticChunker(
        embed_texts=lambda texts: [[1.0, 0.0] for _ in texts],
        chunk_size=120,
        chunk_overlap=20,
        similarity_threshold=0.8,
    )
    document = ParsedDocument(
        source_path="Example.java",
        content="",
        file_type="code",
        language="java",
        sections=[
            ParsedSection(
                title="ExampleService.run",
                text="public void run() { System.out.println(\"hi\"); }",
                kind="code-symbol",
            )
        ],
    )

    chunks = chunker.chunk_document(document)

    assert len(chunks) == 1
    assert chunks[0].file_type == "code"
    assert chunks[0].chunk_kind == "code-symbol"
