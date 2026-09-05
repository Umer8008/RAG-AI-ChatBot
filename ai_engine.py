import os
from typing import List, Dict, Optional
from dotenv import load_dotenv

from langchain_mistralai import ChatMistralAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

AI_SYSTEM_PROMPT = """You are Umer PDF Assistant in AI Knowledge Mode.
You are a highly capable, knowledgeable, and polite AI assistant.
You can answer any general question, explain concepts, provide code, write text, and assist with any topic using your comprehensive knowledge base.
Provide clear, well-structured, and helpful answers."""

def get_ai_llm(model_name: str = "mistral-small-2506", temperature: float = 0.7) -> ChatMistralAI:
    """
    Initializes and returns the ChatMistralAI instance for general AI mode.
    """
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("MISTRAL_API_KEY is not set in environment or .env file.")
    
    return ChatMistralAI(
        model=model_name,
        temperature=temperature
    )

def query_ai(
    query: str,
    chat_history: Optional[List[Dict[str, str]]] = None,
    model_name: str = "mistral-small-2506"
) -> str:
    """
    Executes a direct conversational query to the AI without document restrictions.
    
    Args:
        query: User question
        chat_history: List of past messages e.g. [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
        model_name: Model identifier for Mistral AI
        
    Returns:
        Generated text response from the AI
    """
    llm = get_ai_llm(model_name=model_name)
    
    messages = [SystemMessage(content=AI_SYSTEM_PROMPT)]
    
    # Add previous chat history if available
    if chat_history:
        # Include last few exchanges for context
        for msg in chat_history[-6:]:
            if msg.get("role") == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg.get("role") == "assistant":
                messages.append(AIMessage(content=msg["content"]))
                
    # Add current query
    messages.append(HumanMessage(content=query))
    
    response = llm.invoke(messages)
    return response.content
