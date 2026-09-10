import os
import tempfile
from typing import List, Tuple, Dict, Any
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def extract_and_split_pdf(
    file_source,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> Tuple[List[Document], Dict[str, Any]]:
    """
    Loads any uploaded PDF or file path, and splits it into chunks.
    
    Args:
        file_source: Either a string file path or a Streamlit UploadedFile object.
        chunk_size: Size of each text chunk.
        chunk_overlap: Overlap between consecutive chunks.
        
    Returns:
        Tuple containing:
            - List of split Document objects
            - Metadata dict with filename, total_pages, and total_chunks
    """
    temp_path = None
    file_name = "Uploaded Document"

    try:
        # Check if file_source is a Streamlit UploadedFile or file path
        if hasattr(file_source, "read") and hasattr(file_source, "name"):
            file_name = file_source.name
            # Create a named temporary file
            suffix = os.path.splitext(file_name)[1] or ".pdf"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                tmp_file.write(file_source.getvalue())
                temp_path = tmp_file.name
            load_target = temp_path
        elif isinstance(file_source, str) and os.path.exists(file_source):
            file_name = os.path.basename(file_source)
            load_target = file_source
        else:
            raise ValueError("Invalid file source provided to PDF processor.")

        # Load PDF using PyPDFLoader
        loader = PyPDFLoader(load_target)
        raw_docs = loader.load()

        total_pages = len(raw_docs)

        # Split documents
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = splitter.split_documents(raw_docs)

        # Enhance chunk metadata with filename
        for chunk in chunks:
            if "source" not in chunk.metadata or not chunk.metadata["source"]:
                chunk.metadata["source"] = file_name
            chunk.metadata["filename"] = file_name

        stats = {
            "filename": file_name,
            "total_pages": total_pages,
            "total_chunks": len(chunks)
        }

        return chunks, stats

    finally:
        # Clean up temp file if created
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
