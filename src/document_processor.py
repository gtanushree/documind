"""
PDF ingestion: extract text per page with pypdf, then split into overlapping
chunks with LangChain's RecursiveCharacterTextSplitter. Each chunk keeps
metadata (source filename, page number, chunk id) so answers can later be
traced back to an exact page.
"""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from .config import settings


@dataclass
class ProcessedFile:
    name: str
    chunks: list[Document]
    num_pages: int


def _extract_pages(file_bytes: bytes, filename: str) -> list[Document]:
    reader = PdfReader(BytesIO(file_bytes))
    pages: list[Document] = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if not text.strip():
            continue  # skip blank / image-only pages
        pages.append(
            Document(
                page_content=text,
                metadata={"source": filename, "page": i + 1},
            )
        )
    return pages, len(reader.pages)


def load_and_chunk_pdf(file_bytes: bytes, filename: str) -> ProcessedFile:
    """Extract text from a PDF (as raw bytes) and split it into retrieval-sized chunks."""
    pages, total_pages = _extract_pages(file_bytes, filename)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(pages)

    for i, chunk in enumerate(chunks):
        # Stable, content-addressable ID -> re-uploading the same file overwrites
        # its old vectors instead of duplicating them in the index.
        chunk.metadata["chunk_id"] = f"{filename}::{i}"

    return ProcessedFile(name=filename, chunks=chunks, num_pages=total_pages)
