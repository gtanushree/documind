from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Ensure src/ is importable regardless of the working directory uvicorn
# was launched from.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from typing import AsyncIterator

from fastapi import FastAPI, File, HTTPException, Path as FPath, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field

from src import registry
from src.config import settings
from src.document_processor import load_and_chunk_pdf
from src.qa_chain import build_chain, to_lc_history
from src.vector_store import (
    delete_namespace_vectors,
    index_documents,
    namespace_vector_count,
)

# ─────────────────────────────── app setup ───────────────────────────────────

app = FastAPI(
    title="DocuMind API",
    description="AI-powered PDF document Q&A : LangChain · Pinecone · Groq",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for production
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-process chain cache keyed by namespace (lives as long as the process).
_chain_cache: dict = {}


def _get_chain(namespace: str):
    if namespace not in _chain_cache:
        _chain_cache[namespace] = build_chain(namespace)
    return _chain_cache[namespace]


def _invalidate_chain(namespace: str) -> None:
    _chain_cache.pop(namespace, None)


def _require_keys() -> None:
    if not settings.is_configured():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API keys not configured. Set GROQ_API_KEY and PINECONE_API_KEY in .env.",
        )


def _require_namespace(name: str) -> None:
    if name not in registry.list_namespaces():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Collection '{name}' not found. Create it first with POST /collections.",
        )


# ─────────────────────────────── schemas ─────────────────────────────────────


class CollectionCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=80, examples=["research-papers"])


class CollectionInfo(BaseModel):
    name: str
    created_at: str | None
    files: list[dict]
    vector_count: int


class ChatTurn(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    history: list[ChatTurn] = Field(default_factory=list)


class Source(BaseModel):
    source: str
    page: int | str
    snippet: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]


class UploadResponse(BaseModel):
    files: list[str]
    total_chunks: int
    message: str


class HealthResponse(BaseModel):
    status: str
    configured: bool
    embedding_model: str
    llm_model: str
    pinecone_index: str


# ─────────────────────────────── health ──────────────────────────────────────


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health():
    return HealthResponse(
        status="ok",
        configured=settings.is_configured(),
        embedding_model=settings.embedding_model,
        llm_model=settings.llm_model,
        pinecone_index=settings.pinecone_index_name,
    )


# ─────────────────────────── collections ─────────────────────────────────────


@app.post(
    "/collections",
    status_code=status.HTTP_201_CREATED,
    tags=["Collections"],
    summary="Create a new collection",
)
def create_collection(body: CollectionCreate):
    _require_keys()
    clean = body.name.strip().lower().replace(" ", "-")
    if clean in registry.list_namespaces():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Collection '{clean}' already exists.",
        )
    registry.create_namespace(clean)
    return {"name": clean, "message": "Collection created."}


@app.get("/collections", tags=["Collections"], summary="List all collections")
def list_collections():
    _require_keys()
    return {"collections": registry.list_namespaces()}


@app.get(
    "/collections/{name}",
    response_model=CollectionInfo,
    tags=["Collections"],
    summary="Get collection info",
)
def get_collection(name: str = FPath(...)):
    _require_keys()
    _require_namespace(name)
    info = registry.get_namespace_info(name)
    return CollectionInfo(
        name=name,
        created_at=info.get("created_at"),
        files=info.get("files", []),
        vector_count=namespace_vector_count(name),
    )


@app.delete(
    "/collections/{name}",
    status_code=status.HTTP_200_OK,
    tags=["Collections"],
    summary="Delete a collection and all its vectors",
)
def delete_collection(name: str = FPath(...)):
    _require_keys()
    _require_namespace(name)
    delete_namespace_vectors(name)
    registry.delete_namespace(name)
    _invalidate_chain(name)
    return {"message": f"Collection '{name}' deleted."}


# ─────────────────────────── documents ───────────────────────────────────────


@app.post(
    "/collections/{name}/upload",
    response_model=UploadResponse,
    tags=["Documents"],
    summary="Upload and index PDF files into a collection",
)
async def upload_documents(
    name: str = FPath(...),
    files: list[UploadFile] = File(..., description="One or more PDF files"),
):
    _require_keys()
    _require_namespace(name)

    non_pdf = [f.filename for f in files if not (f.filename or "").lower().endswith(".pdf")]
    if non_pdf:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Only PDF files are accepted. Rejected: {non_pdf}",
        )

    all_chunks = []
    file_meta = []

    for f in files:
        raw = await f.read()
        processed = load_and_chunk_pdf(raw, f.filename or "unknown.pdf")
        all_chunks.extend(processed.chunks)
        file_meta.append(
            {"name": processed.name, "chunks": len(processed.chunks), "pages": processed.num_pages}
        )

    if not all_chunks:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No text could be extracted from the uploaded PDFs. Ensure they are not image-only/scanned.",
        )

    index_documents(all_chunks, name)
    registry.register_files(name, file_meta)
    _invalidate_chain(name)  # force retriever rebuild on next query

    return UploadResponse(
        files=[m["name"] for m in file_meta],
        total_chunks=len(all_chunks),
        message=f"Successfully indexed {len(files)} file(s) into '{name}'.",
    )


@app.get(
    "/collections/{name}/files",
    tags=["Documents"],
    summary="List files indexed in a collection",
)
def list_files(name: str = FPath(...)):
    _require_keys()
    _require_namespace(name)
    info = registry.get_namespace_info(name)
    return {"collection": name, "files": info.get("files", [])}


# ────────────────────────────── chat ─────────────────────────────────────────


@app.post(
    "/collections/{name}/chat",
    response_model=ChatResponse,
    tags=["Chat"],
    summary="Single-turn Q&A with optional conversation history",
)
def chat(name: str = FPath(...), body: ChatRequest = ...):
    """
    Send a question and get a grounded answer with source citations.

    Pass previous turns in `history` (alternating user / assistant) to enable
    follow-up resolution — the chain will condense context-dependent questions
    (e.g. "what about its overhead?") into standalone queries before retrieval.

    Example request body:
    ```json
    {
      "question": "What loss function is used?",
      "history": [
        {"role": "user",      "content": "Summarise the paper."},
        {"role": "assistant", "content": "The paper proposes ..."}
      ]
    }
    ```
    """
    _require_keys()
    _require_namespace(name)
    info = registry.get_namespace_info(name)
    if not info.get("files"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Collection '{name}' has no indexed documents. Upload PDFs first.",
        )

    chain = _get_chain(name)
    history_dicts = [{"role": t.role, "content": t.content} for t in body.history]
    result = chain.invoke(
        {"input": body.question, "chat_history": to_lc_history(history_dicts)}
    )

    # Deduplicate and format source citations
    sources: list[Source] = []
    seen: set[tuple] = set()
    for doc in result.get("context", []):
        key = (doc.metadata.get("source"), doc.metadata.get("page"))
        if key in seen:
            continue
        seen.add(key)
        snippet = doc.page_content[:280].strip()
        if len(doc.page_content) > 280:
            snippet += "…"
        sources.append(
            Source(
                source=doc.metadata.get("source", "unknown"),
                page=doc.metadata.get("page", "?"),
                snippet=snippet,
            )
        )

    return ChatResponse(answer=result["answer"], sources=sources)


@app.post(
    "/collections/{name}/stream",
    tags=["Chat"],
    summary="Streaming Q&A — server-sent events (text/event-stream)",
    response_class=StreamingResponse,
    responses={
        200: {
            "description": "SSE stream of answer tokens. Final event is a JSON `[SOURCES]` block.",
            "content": {"text/event-stream": {}},
        }
    },
)
def chat_stream(name: str = FPath(...), body: ChatRequest = ...):
    """
    Same as `/chat` but streams the answer token-by-token as Server-Sent Events.

    Each streamed event is one of:
    - `data: <token>\\n\\n`            — a text chunk
    - `data: [DONE]\\n\\n`             — end of answer tokens
    - `data: [SOURCES] <json>\\n\\n`   — final JSON array of source citations

    Consume with `EventSource` in JS or `httpx`/`requests` with `stream=True` in Python.
    """
    _require_keys()
    _require_namespace(name)
    info = registry.get_namespace_info(name)
    if not info.get("files"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Collection '{name}' has no indexed documents.",
        )

    from src.qa_chain import (
        ANSWER_PROMPT,
        CONDENSE_PROMPT,
        _format_docs,
    )
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnableBranch, RunnablePassthrough
    from langchain_google_genai import ChatGoogleGenerativeAI
    from src.vector_store import get_vector_store
    import json

    history_dicts = [{"role": t.role, "content": t.content} for t in body.history]
    lc_history = to_lc_history(history_dicts)

    def event_stream():
        retriever = get_vector_store(name).as_retriever(
            search_kwargs={"k": settings.retrieval_k}
        )
        llm = ChatGoogleGenerativeAI(
            model=settings.llm_model,
            google_api_key=settings.google_api_key,
            temperature=0.1,
        )

        # 1. Resolve standalone query (same logic as build_chain)
        condense_chain = CONDENSE_PROMPT | llm | StrOutputParser()
        if lc_history:
            standalone = condense_chain.invoke(
                {"input": body.question, "chat_history": lc_history}
            )
        else:
            standalone = body.question

        # 2. Retrieve context
        docs = retriever.invoke(standalone)
        context_str = _format_docs(docs)

        # 3. Stream the answer token-by-token
        prompt = ANSWER_PROMPT.invoke(
            {"input": body.question, "chat_history": lc_history, "context": context_str}
        )
        for chunk in llm.stream(prompt):
            token = chunk.content
            if token:
                yield f"data: {token}\n\n"

        yield "data: [DONE]\n\n"

        # 4. Emit deduplicated source citations as a final SSE event
        sources = []
        seen: set[tuple] = set()
        for doc in docs:
            key = (doc.metadata.get("source"), doc.metadata.get("page"))
            if key in seen:
                continue
            seen.add(key)
            snippet = doc.page_content[:280].strip()
            if len(doc.page_content) > 280:
                snippet += "…"
            sources.append(
                {
                    "source": doc.metadata.get("source", "unknown"),
                    "page": doc.metadata.get("page", "?"),
                    "snippet": snippet,
                }
            )
        yield f"data: [SOURCES] {json.dumps(sources)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ──────────────────────────── entrypoint ─────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)