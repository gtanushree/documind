from __future__ import annotations

import time

from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface import HuggingFaceEmbeddings

from .config import settings

_pc_client: Pinecone | None = None


def get_pinecone_client() -> Pinecone:
    global _pc_client
    if _pc_client is None:
        _pc_client = Pinecone(api_key=settings.pinecone_api_key)
    return _pc_client


def ensure_index_exists() -> None:
    pc = get_pinecone_client()
    existing = pc.list_indexes().names()
    if settings.pinecone_index_name in existing:
        return
    pc.create_index(
        name=settings.pinecone_index_name,
        dimension=settings.embedding_dimension,
        metric="cosine",
        spec=ServerlessSpec(cloud=settings.pinecone_cloud, region=settings.pinecone_region),
    )
    while not pc.describe_index(settings.pinecone_index_name).status["ready"]:
        time.sleep(1)


def get_embeddings() -> HuggingFaceEmbeddings:
    # Groq does not offer an embeddings API, so embeddings are generated
    # locally via a HuggingFace sentence-transformers model instead.
    # NOTE: this model outputs 384-dim vectors -- settings.embedding_dimension
    # (used by ensure_index_exists) must be 384, not 1536.
    return HuggingFaceEmbeddings(model_name=settings.embedding_model)


def get_vector_store(namespace: str) -> PineconeVectorStore:
    ensure_index_exists()
    index = get_pinecone_client().Index(settings.pinecone_index_name)
    return PineconeVectorStore(index=index, embedding=get_embeddings(), namespace=namespace)


def index_documents(documents, namespace: str) -> int:
    if not documents:
        return 0
    store = get_vector_store(namespace)
    ids = [d.metadata["chunk_id"] for d in documents]
    store.add_documents(documents, ids=ids)
    return len(documents)


def delete_namespace_vectors(namespace: str) -> None:
    pc = get_pinecone_client()
    index = pc.Index(settings.pinecone_index_name)
    try:
        index.delete(delete_all=True, namespace=namespace)
    except Exception:
        pass  # namespace already empty / never had vectors -> nothing to delete


def namespace_vector_count(namespace: str) -> int:
    pc = get_pinecone_client()
    index = pc.Index(settings.pinecone_index_name)
    stats = index.describe_index_stats()
    namespaces = stats.get("namespaces") or {}
    ns_stats = namespaces.get(namespace)
    if not ns_stats:
        return 0
    return ns_stats.get("vector_count", 0) if hasattr(ns_stats, "get") else getattr(ns_stats, "vector_count", 0)
