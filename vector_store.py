import os
from typing import List, Optional
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

_EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_cached_embeddings = None

def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Initializes or returns cached HuggingFaceEmbeddings model.
    """
    global _cached_embeddings
    if _cached_embeddings is None:
        _cached_embeddings = HuggingFaceEmbeddings(
            model_name=_EMBEDDING_MODEL_NAME
        )
    return _cached_embeddings

def build_vectorstore_from_chunks(
    chunks: List[Document],
    persist_directory: Optional[str] = None,
    collection_name: str = "pdf_collection"
) -> Chroma:
    """
    Creates a new Chroma vector store from document chunks.
    If persist_directory is provided, it stores data in that folder.
    """
    embedding_function = get_embedding_model()
    
    if persist_directory:
        os.makedirs(persist_directory, exist_ok=True)
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_function,
            persist_directory=persist_directory,
            collection_name=collection_name
        )
    else:
        # In-memory vector store for fast session-based RAG
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_function,
            collection_name=collection_name
        )
    
    return vector_store

def load_existing_vectorstore(
    persist_directory: str = "Chroma_DB",
    collection_name: str = "pdf_collection"
) -> Optional[Chroma]:
    """
    Loads an existing Chroma vector store from disk if present.
    """
    if not os.path.exists(persist_directory):
        return None

    try:
        embedding_function = get_embedding_model()
        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_function,
            collection_name=collection_name
        )
        return vector_store
    except Exception:
        return None
