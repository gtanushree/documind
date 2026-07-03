# DocuMind — AI-Powered Document Q&A

Upload PDFs, ask questions in natural language, and get answers grounded in
your documents with cited sources — powered by **LangChain**, **Pinecone**,
**Streamlit**, and the **Groq API**.

## Features

- **Multi-document collections** — group related PDFs into named "collections," each backed by its own isolated Pinecone namespace
- **Conversational Q&A** — follow-up questions ("what about its accuracy?") are automatically resolved against chat history before retrieval
- **Source citations** — every answer links back to the exact document and page number it came from, with a content snippet
- **Persistent vector storage** — documents stay indexed in Pinecone across sessions; re-launching the app doesn't require re-uploading
- **Idempotent re-indexing** — re-uploading the same file overwrites its old vectors instead of duplicating them

## Architecture

```
PDF upload
    │
    ▼
pypdf (per-page text extraction)
    │
    ▼
RecursiveCharacterTextSplitter (chunking + metadata: source, page, chunk_id)
    │
    ▼
HuggingFace Embeddings (BAAI/bge-small-en-v1.5)
    │
    ▼
Pinecone (serverless index, one namespace per collection)
    │
    ▼
┌─────────────── on each question ─────────────────────────┐
│  history-aware retriever (condense follow-ups)           │
│              │                                           │
│              ▼                                           │
│  Pinecone similarity search (top-k chunks)               │
│              │                                           │
│              ▼                                           │
│  ChatGroq (llama-3.3-70b-versatile) — answer from context│
└──────────────────────────────────────────────────────────┘
    │
    ▼
Streamlit chat UI (answer + expandable sources)
```

The RAG chain in `src/qa_chain.py` is built directly with LangChain's
Runnable/LCEL primitives rather than the older `create_retrieval_chain`
helpers, which were removed from the core `langchain` package in its 1.0
restructuring.

## Project structure

```
documind/
├── app.py                     # Streamlit UI
├── src/
│   ├── config.py              # Settings loaded from .env
│   ├── db.py
│   ├── document_processor.py  # PDF text extraction + chunking
│   ├── vector_store.py        # Pinecone index + LangChain vector store
│   ├── qa_chain.py            # Conversational RAG chain (LCEL)
│   └── registry.py            # Local JSON sidecar: which files are in which collection
├── data/registry.json         # Created automatically on first run
├── requirements.txt
└── .env.example
```

## Setup

**1. Clone and install dependencies** (Python 3.10+ recommended):

```bash
pip install -r requirements.txt
```

**2. Get your API keys**

- Groq: https://console.groq.com/keys
- Pinecone: https://app.pinecone.io (free tier supports serverless indexes)

**3. Configure environment variables**

```bash
cp .env.example .env
```

Edit `.env` and fill in:

```
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=pcsk-...
```

The other variables have sensible defaults — change them if you want a
different model, chunk size, or Pinecone region. The Pinecone index is
created automatically on first run if it doesn't already exist.

**4. Run the app**

```bash
streamlit run app.py
```

Open the URL Streamlit prints (typically `http://localhost:8501`).

## Using DocuMind

1. In the sidebar, create a new **collection** (e.g. `research-papers`) — this maps to a Pinecone namespace, keeping unrelated document sets from polluting each other's search results.
2. Upload one or more PDFs and click **Process & Index**.
3. Ask questions in the chat box. Each answer includes an expandable **sources** section showing exactly which document and page it drew from.
4. Switch collections any time from the sidebar dropdown — chat history is scoped per collection.
5. **Delete collection** removes both the Pinecone vectors and the local bookkeeping for that namespace.

## Configuration reference (`.env`)

| Variable | Default | Notes |
|---|---|---|
| `GROQ_API_KEY` | — | required |
| `PINECONE_API_KEY` | — | required |
| `PINECONE_INDEX_NAME` | `documind` | created automatically if missing |
| `PINECONE_CLOUD` / `PINECONE_REGION` | `aws` / `us-east-1` | serverless index location |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | also supports `text-embedding-3-large`, `text-embedding-ada-002` |
| `LLM_MODEL` | `llama-3.3-70b-versatile` | any chat-completion-capable Groq model |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `150` | characters per chunk |
| `RETRIEVAL_K` | `4` | number of chunks retrieved per question |

> Changing `EMBEDDING_MODEL` after documents are already indexed will mismatch
> vector dimensions for that index — use a new `PINECONE_INDEX_NAME` if you switch models.

## Notes & possible extensions

- Scanned/image-only PDFs won't extract text (no OCR is included) — pages with no extractable text are silently skipped during indexing.
- Chat history currently lives in Streamlit session state and resets when the app restarts; the underlying Pinecone index does not.
- Natural extensions: OCR fallback (e.g. `pytesseract`), DOCX/TXT support, per-user auth, streaming token-by-token answers, hybrid (keyword + vector) search.
