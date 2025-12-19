"""
Vector Stores - Modular structure
리팩토링된 모듈 구조
"""

# Base classes
from .base import BaseVectorStore, VectorSearchResult

# Search algorithms
from .search import SearchAlgorithms, AdvancedSearchMixin

# 기존 구현 (임시로 old에서 import)
from ..vector_stores_old import (
    ChromaVectorStore,
    PineconeVectorStore,
    FAISSVectorStore,
    QdrantVectorStore,
    WeaviateVectorStore,
    VectorStore,
    VectorStoreBuilder,
    create_vector_store,
    from_documents
)

__all__ = [
    # Base
    "BaseVectorStore",
    "VectorSearchResult",
    # Search
    "SearchAlgorithms",
    "AdvancedSearchMixin",
    # Implementations
    "ChromaVectorStore",
    "PineconeVectorStore",
    "FAISSVectorStore",
    "QdrantVectorStore",
    "WeaviateVectorStore",
    # Factory
    "VectorStore",
    "VectorStoreBuilder",
    "create_vector_store",
    "from_documents",
]
