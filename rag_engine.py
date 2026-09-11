import os
from typing import Dict, Any, List
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

load_dotenv()

# Strict Prompt strictly following the original project prompt
RAG_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    (
        "system",
        """Answer the question using ONLY the provided context.

Do not add information from your own knowledge.

If the answer is not clearly present in the context, say:
"I could not find the answer in the document."

Keep the answer concise and directly related to the context."""
    ),
    (
        "human",
        """Context:
{context}

Question:
{question}"""
    )
])

def get_rag_llm(model_name: str = "mistral-small-2506", temperature: float = 0.2) -> ChatMistralAI:
    """
    Initializes and returns the ChatMistralAI instance.
    """
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("MISTRAL_API_KEY is not set in environment or .env file.")
    
    return ChatMistralAI(
        model=model_name,
        temperature=temperature
    )

def query_rag(
    query: str,
    vector_store: Chroma,
    model_name: str = "mistral-small-2506",
    k: int = 4,
    fetch_k: int = 10,
    lambda_mult: float = 0.5
) -> Dict[str, Any]:
    """
    Executes a RAG query against the vector store using MMR retrieval and strict doc prompt.
    
    Returns:
        Dict containing:
            - 'answer': Generated answer string
            - 'docs': List of retrieved Document objects
            - 'context': Combined context text passed to LLM
    """
    if not vector_store:
        return {
            "answer": "No document is currently loaded in RAG mode. Please upload a PDF first.",
            "docs": [],
            "context": ""
        }

    # Setup MMR retriever matching original project configuration
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "fetch_k": fetch_k,
            "lambda_mult": lambda_mult
        }
    )

    # Retrieve docs
    docs: List[Document] = retriever.invoke(query)

    # Form context string
    context = "\n\n".join(doc.page_content for doc in docs)

    # Initialize LLM
    llm = get_rag_llm(model_name=model_name)

    # Generate response
    formatted_prompt = RAG_PROMPT_TEMPLATE.invoke({
        "context": context,
        "question": query
    })

    response = llm.invoke(formatted_prompt)

    return {
        "answer": response.content,
        "docs": docs,
        "context": context
    }
