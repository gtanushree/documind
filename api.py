"""
DocuMind — FastAPI server
LangChain · Pinecone · Groq · HuggingFace

Run with:  uvicorn api:app --reload --port 8000
Docs at:   http://localhost:8000/docs
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import FastAPI, File, HTTPException, Path as FPath, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, StreamingResponse
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

# ── app setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="DocuMind API",
    description="AI-powered PDF document Q&A — LangChain · Pinecone · Groq · HuggingFace",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Middleware MUST be registered before routes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── chain cache ──────────────────────────────────────────────────────────────

_chain_cache: dict = {}

def _get_chain(namespace: str):
    if namespace not in _chain_cache:
        _chain_cache[namespace] = build_chain(namespace)
    return _chain_cache[namespace]

def _invalidate_chain(namespace: str) -> None:
    _chain_cache.pop(namespace, None)

# ── guards ───────────────────────────────────────────────────────────────────

def _require_keys() -> None:
    if not settings.is_configured():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API keys not configured. Set GROQ_API_KEY and PINECONE_API_KEY.",
        )

def _pinecone_namespace_exists(name: str) -> bool:
    """Check Pinecone directly — works even after a cold start wipes the registry."""
    try:
        return namespace_vector_count(name) > 0
    except Exception:
        return False

def _require_namespace(name: str) -> None:
    """
    Validate that a namespace exists in EITHER the local registry OR Pinecone.

    On Vercel the registry lives in /tmp and resets on cold starts.  Pinecone
    is the durable source of truth, so we always fall back to it.  When a
    namespace is found in Pinecone but missing from the registry (cold-start
    amnesia) we silently recreate the registry entry so subsequent calls work.
    """
    in_registry = name in registry.list_namespaces()
    if in_registry:
        return  # fast path

    if _pinecone_namespace_exists(name):
        # Registry was wiped (cold start) but vectors are still there — restore entry
        registry.create_namespace(name)
        return

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Collection '{name}' not found. Create it first with POST /collections.",
    )

def _pinecone_all_namespaces() -> list[str]:
    """
    Return every non-empty namespace from Pinecone's index stats.
    This is the fallback when the registry is empty after a cold start.
    """
    try:
        from src.vector_store import get_pinecone_client
        pc = get_pinecone_client()
        from src.config import settings as s
        index = pc.Index(s.pinecone_index_name)
        stats = index.describe_index_stats()
        namespaces = stats.get("namespaces") or {}
        return [ns for ns, info in namespaces.items()
                if ns and (info.get("vector_count", 0) if hasattr(info, "get") else getattr(info, "vector_count", 0)) > 0]
    except Exception:
        return []

# ── schemas ──────────────────────────────────────────────────────────────────

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

# ── routes ───────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")

@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health():
    return HealthResponse(
        status="ok",
        configured=settings.is_configured(),
        embedding_model=settings.embedding_model,
        llm_model=settings.llm_model,
        pinecone_index=settings.pinecone_index_name,
    )

# collections

@app.post("/collections", status_code=status.HTTP_201_CREATED, tags=["Collections"],
          summary="Create a new collection")
def create_collection(body: CollectionCreate):
    _require_keys()
    clean = body.name.strip().lower().replace(" ", "-")
    # Check both registry AND Pinecone so we don't create duplicates
    if clean in registry.list_namespaces() or _pinecone_namespace_exists(clean):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Collection '{clean}' already exists.",
        )
    registry.create_namespace(clean)
    return {"name": clean, "message": "Collection created."}

@app.get("/collections", tags=["Collections"], summary="List all collections")
def list_collections():
    """
    Returns collections from the registry merged with namespaces found in
    Pinecone. This means the list is always accurate even after a Vercel
    cold start wipes the ephemeral /tmp registry.
    """
    _require_keys()
    from_registry = set(registry.list_namespaces())
    from_pinecone = set(_pinecone_all_namespaces())
    merged = sorted(from_registry | from_pinecone)

    # Restore any registry entries lost on cold start
    for ns in from_pinecone - from_registry:
        registry.create_namespace(ns)

    return {"collections": merged}

@app.get("/collections/{name}", response_model=CollectionInfo, tags=["Collections"],
         summary="Get collection info")
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

@app.delete("/collections/{name}", status_code=status.HTTP_200_OK, tags=["Collections"],
            summary="Delete a collection and all its vectors")
def delete_collection(name: str = FPath(...)):
    _require_keys()
    _require_namespace(name)
    delete_namespace_vectors(name)
    registry.delete_namespace(name)
    _invalidate_chain(name)
    return {"message": f"Collection '{name}' deleted."}

# documents

@app.post("/collections/{name}/upload", response_model=UploadResponse,
          tags=["Documents"], summary="Upload and index PDF files into a collection")
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
            detail=f"Only PDF files accepted. Rejected: {non_pdf}",
        )

    all_chunks, file_meta = [], []
    for f in files:
        raw = await f.read()
        processed = load_and_chunk_pdf(raw, f.filename or "unknown.pdf")
        all_chunks.extend(processed.chunks)
        file_meta.append({"name": processed.name, "chunks": len(processed.chunks), "pages": processed.num_pages})

    if not all_chunks:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No text could be extracted. Ensure PDFs are not image-only/scanned.",
        )

    index_documents(all_chunks, name)
    registry.register_files(name, file_meta)
    _invalidate_chain(name)

    return UploadResponse(
        files=[m["name"] for m in file_meta],
        total_chunks=len(all_chunks),
        message=f"Successfully indexed {len(files)} file(s) into '{name}'.",
    )

@app.get("/collections/{name}/files", tags=["Documents"],
         summary="List files indexed in a collection")
def list_files(name: str = FPath(...)):
    _require_keys()
    _require_namespace(name)
    info = registry.get_namespace_info(name)
    return {"collection": name, "files": info.get("files", [])}

# chat

@app.post("/collections/{name}/chat", response_model=ChatResponse, tags=["Chat"],
          summary="Single-turn Q&A with optional conversation history")
def chat(name: str = FPath(...), body: ChatRequest = ...):
    _require_keys()
    _require_namespace(name)

    # Use Pinecone vector count as the ground truth for "has data".
    # Registry files list may be empty after a cold start even when
    # vectors are present — Pinecone never lies.
    vec_count = namespace_vector_count(name)
    if vec_count == 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Collection '{name}' has no indexed documents. Upload PDFs first.",
        )

    chain = _get_chain(name)
    history_dicts = [{"role": t.role, "content": t.content} for t in body.history]
    result = chain.invoke({"input": body.question, "chat_history": to_lc_history(history_dicts)})

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
        sources.append(Source(
            source=doc.metadata.get("source", "unknown"),
            page=doc.metadata.get("page", "?"),
            snippet=snippet,
        ))

    return ChatResponse(answer=result["answer"], sources=sources)

@app.post("/collections/{name}/stream", tags=["Chat"],
          summary="Streaming Q&A — server-sent events",
          response_class=StreamingResponse,
          responses={200: {"content": {"text/event-stream": {}}}})
def chat_stream(name: str = FPath(...), body: ChatRequest = ...):
    _require_keys()
    _require_namespace(name)

    if namespace_vector_count(name) == 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Collection '{name}' has no indexed documents.",
        )

    from src.qa_chain import ANSWER_PROMPT, CONDENSE_PROMPT, _format_docs
    from langchain_core.output_parsers import StrOutputParser
    from langchain_groq import ChatGroq
    from src.vector_store import get_vector_store
    import json

    history_dicts = [{"role": t.role, "content": t.content} for t in body.history]
    lc_history = to_lc_history(history_dicts)

    def event_stream():
        retriever = get_vector_store(name).as_retriever(search_kwargs={"k": settings.retrieval_k})
        llm = ChatGroq(
            model=settings.llm_model,
            groq_api_key=settings.groq_api_key,
            temperature=0.1,
        )
        condense_chain = CONDENSE_PROMPT | llm | StrOutputParser()
        standalone = condense_chain.invoke({"input": body.question, "chat_history": lc_history}) if lc_history else body.question
        docs = retriever.invoke(standalone)
        context_str = _format_docs(docs)
        prompt = ANSWER_PROMPT.invoke({"input": body.question, "chat_history": lc_history, "context": context_str})
        for chunk in llm.stream(prompt):
            if chunk.content:
                yield f"data: {chunk.content}\n\n"
        yield "data: [DONE]\n\n"
        sources, seen = [], set()
        for doc in docs:
            key = (doc.metadata.get("source"), doc.metadata.get("page"))
            if key in seen: continue
            seen.add(key)
            snippet = doc.page_content[:280].strip()
            if len(doc.page_content) > 280: snippet += "…"
            sources.append({"source": doc.metadata.get("source", "unknown"), "page": doc.metadata.get("page", "?"), "snippet": snippet})
        yield f"data: [SOURCES] {json.dumps(sources)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

# ── entrypoint ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)