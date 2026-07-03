
from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

#_EMBEDDING_DIMENSIONS = {
#    "text-embedding-3-small": 1536,
#    "text-embedding-3-large": 3072,
#    "text-embedding-ada-002": 1536,
#}
_EMBEDDING_DIMENSIONS = {
    # HuggingFace / sentence-transformers models (in use)
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-large-en-v1.5": 1024,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    # OpenAI models (no longer used, kept for reference)
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


@dataclass
class Settings:
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    pinecone_api_key: str = field(default_factory=lambda: os.getenv("PINECONE_API_KEY", ""))
    pinecone_index_name: str = field(default_factory=lambda: os.getenv("PINECONE_INDEX_NAME", "documind"))
    pinecone_cloud: str = field(default_factory=lambda: os.getenv("PINECONE_CLOUD", "aws"))
    pinecone_region: str = field(default_factory=lambda: os.getenv("PINECONE_REGION", "us-east-1"))

    embedding_model: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"))
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "gemma2-9b-it"))

    chunk_size: int = field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "1000")))
    chunk_overlap: int = field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "150")))
    retrieval_k: int = field(default_factory=lambda: int(os.getenv("RETRIEVAL_K", "4")))

    @property
    def embedding_dimension(self) -> int:
        #return _EMBEDDING_DIMENSIONS.get(self.embedding_model, 1536)
        try:
            return _EMBEDDING_DIMENSIONS[self.embedding_model]
        except KeyError:
            raise ValueError(
                f"Unknown EMBEDDING_MODEL '{self.embedding_model}' - add its output "
                f"dimension to _EMBEDDING_DIMENSIONS in config.py before using it, "
                f"otherwise the Pinecone index will be created with the wrong "
                f"dimension and indexing will fail."
            )

    def is_configured(self) -> bool:
        return bool(self.groq_api_key and self.pinecone_api_key)


settings = Settings()
