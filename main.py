from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# Embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load Chroma database
vector_store = Chroma(
    persist_directory="Chroma_DB",
    embedding_function=embedding_model
)

# Retriever
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 10,
        "lambda_mult": 0.5
    }
)

# Mistral
llm = ChatMistralAI(
    model="mistral-small-2506"
)

# Prompt
prompt = ChatPromptTemplate.from_messages([
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

# Chat loop
print("Welcome to Student RAG Assistant")
print("Press 0 to exit")

while True:

    query = input("\nYou: ")

    if query.strip() == "0":
        break

    # Retrieve documents
    docs = retriever.invoke(query)

    # Create context
    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    # Create prompt
    final_prompt = prompt.invoke({
        "context": context,
        "question": query
    })

    # Get answer
    response = llm.invoke(final_prompt)

    print(f"\nAI: {response.content}")