# NOTE: This module is no longer used by the app. `src/vector_store.py`
# now creates/connects to the Pinecone index itself via
# `ensure_index_exists()`, using settings from `src/config.py` (correct
# index name + embedding dimension) instead of the hardcoded values below.
# This file is safe to delete. Kept only in case you still reference it
# from a standalone script; if so, it's been updated to pull config from
# the same place the rest of the app does, rather than hardcoding a
# different index name ("my-first-index") and an OpenAI-sized dimension
# (1536) that no longer matches the HuggingFace embeddings in use.

from .config import settings
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key=settings.pinecone_api_key)

# Create the index only if it does not already exist
if settings.pinecone_index_name not in pc.list_indexes().names():
    pc.create_index(
        name=settings.pinecone_index_name,
        dimension=settings.embedding_dimension,
        metric="cosine",
        spec=ServerlessSpec(
            cloud=settings.pinecone_cloud,
            region=settings.pinecone_region,
        )
    )

# Connect to the index and export it for use in other files
index = pc.Index(settings.pinecone_index_name)
