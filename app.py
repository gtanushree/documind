"""
DocuMind — AI-Powered Document Q&A
LangChain · Pinecone · Streamlit · OpenAI

Run with:  streamlit run app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

from src import registry
from src.config import settings
from src.document_processor import load_and_chunk_pdf
from src.qa_chain import ask, build_chain
from src.vector_store import delete_namespace_vectors, index_documents, namespace_vector_count

st.set_page_config(page_title="DocuMind", page_icon="📄", layout="wide")

# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # [{"role": "user"|"assistant", "content": str, "sources"?: [...]}]
if "namespace" not in st.session_state:
    st.session_state.namespace = None
if "chain_cache" not in st.session_state:
    st.session_state.chain_cache = {}  # namespace -> built chain


def reset_chat() -> None:
    st.session_state.chat_history = []


def get_chain(namespace: str):
    if namespace not in st.session_state.chain_cache:
        st.session_state.chain_cache[namespace] = build_chain(namespace)
    return st.session_state.chain_cache[namespace]


def render_sources(sources: list[dict]) -> None:
    with st.expander(f" {len(sources)} source(s)"):
        for s in sources:
            st.markdown(f"**{s['source']}** — page {s['page']}")
            st.caption(s["snippet"])


# Sidebar — collection management
with st.sidebar:
    st.title("📄 DocuMind")
    st.caption("AI-powered document Q&A — LangChain · Pinecone · OpenAI")

    if not settings.is_configured():
        st.error("Missing API keys. Add OPENAI_API_KEY and PINECONE_API_KEY to a `.env` file (see `.env.example`).")
        st.stop()

    st.divider()
    st.subheader("Collection")
    st.caption("Each collection is an isolated Pinecone namespace — keep unrelated document sets separate.")

    namespaces = registry.list_namespaces()
    options = ["+ New collection"] + namespaces
    default_index = options.index(st.session_state.namespace) if st.session_state.namespace in options else 0

    choice = st.selectbox("Active collection", options, index=default_index, label_visibility="collapsed")

    if choice == "+ New collection":
        new_name = st.text_input("Collection name", placeholder="e.g. research-papers")
        if st.button("Create", use_container_width=True, disabled=not new_name.strip()):
            clean = new_name.strip().lower().replace(" ", "-")
            registry.create_namespace(clean)
            st.session_state.namespace = clean
            reset_chat()
            st.rerun()
    else:
        if choice != st.session_state.namespace:
            st.session_state.namespace = choice
            reset_chat()

    namespace = st.session_state.namespace

    if namespace:
        st.divider()
        st.subheader("Upload PDFs")
        uploads = st.file_uploader(
            "Add documents to this collection", type=["pdf"], accept_multiple_files=True, label_visibility="collapsed"
        )
        if uploads and st.button("Process & Index", use_container_width=True, type="primary"):
            progress = st.progress(0.0, text="Starting…")
            all_chunks = []
            file_meta = []
            for i, f in enumerate(uploads):
                progress.progress(i / len(uploads), text=f"Reading {f.name}…")
                processed = load_and_chunk_pdf(f.read(), f.name)
                all_chunks.extend(processed.chunks)
                file_meta.append({"name": processed.name, "chunks": len(processed.chunks), "pages": processed.num_pages})

            progress.progress(0.7, text="Embedding & uploading to Pinecone…")
            index_documents(all_chunks, namespace)
            registry.register_files(namespace, file_meta)
            st.session_state.chain_cache.pop(namespace, None)  # force retriever rebuild
            progress.progress(1.0, text="Done!")
            st.success(f"Indexed {len(uploads)} file(s), {len(all_chunks)} chunks.")
            st.rerun()

        st.divider()
        info = registry.get_namespace_info(namespace)
        st.subheader(f"Indexed files ({len(info['files'])})")
        if info["files"]:
            for f in info["files"]:
                st.caption(f"📄 **{f['name']}** — {f['pages']} pages, {f['chunks']} chunks")
        else:
            st.caption("No documents indexed yet.")
        st.caption(f"Vectors in Pinecone: {namespace_vector_count(namespace)}")

        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Clear chat", use_container_width=True):
                reset_chat()
                st.rerun()
        with col2:
            if st.button("Delete collection", use_container_width=True):
                delete_namespace_vectors(namespace)
                registry.delete_namespace(namespace)
                st.session_state.chain_cache.pop(namespace, None)
                st.session_state.namespace = None
                reset_chat()
                st.rerun()

    st.divider()
    with st.expander("⚙️ Settings"):
        st.text(f"Embedding model:  {settings.embedding_model}")
        st.text(f"LLM model:        {settings.llm_model}")
        st.text(f"Chunk size:       {settings.chunk_size}")
        st.text(f"Chunk overlap:    {settings.chunk_overlap}")
        st.text(f"Retrieval top-k:  {settings.retrieval_k}")

# Main panel — chat
st.header("DocuMind")
st.caption("Ask questions about your PDFs and get grounded answers with cited sources.")

if not st.session_state.namespace:
    st.info("Create or select a collection in the sidebar to get started.")
    st.stop()

namespace = st.session_state.namespace
info = registry.get_namespace_info(namespace)

if not info["files"]:
    st.info(f"Collection **{namespace}** is empty. Upload PDFs in the sidebar to begin asking questions.")
    st.stop()

for turn in st.session_state.chat_history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn.get("sources"):
            render_sources(turn["sources"])

question = st.chat_input(f"Ask something about the documents in '{namespace}'…")
if question:
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            chain = get_chain(namespace)
            result = ask(chain, question, st.session_state.chat_history[:-1])
        st.markdown(result["answer"])
        if result["sources"]:
            render_sources(result["sources"])

    st.session_state.chat_history.append(
        {"role": "assistant", "content": result["answer"], "sources": result["sources"]}
    )
