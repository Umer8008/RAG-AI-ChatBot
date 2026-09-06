
# 📄 Umer PDF Assistant

> **A modern AI-powered PDF assistant with dual-engine architecture: strict document-grounded RAG mode and conversational AI mode.**

Umer PDF Assistant is a **Retrieval-Augmented Generation (RAG) application** built with Python, LangChain, ChromaDB, Hugging Face embeddings, Mistral AI, and Streamlit.

The application allows users to upload PDF documents, build a searchable vector knowledge base, and ask questions grounded strictly in the document content. It also provides a separate **AI Mode** for general-purpose conversations, coding, explanations, and creative tasks.

---

## ✨ Key Features

* 📄 **PDF Document Processing**
* 🔍 **Semantic Vector Search**
* 🧠 **Retrieval-Augmented Generation (RAG)**
* 🎯 **MMR-based Diverse Retrieval**
* 📚 **Page-level Source Citations**
* 🤖 **Mistral AI Integration**
* 💬 **Conversational AI Mode**
* 🧩 **Rolling Chat Memory**
* 💾 **Persistent ChromaDB Vector Store**
* 🔢 **384-dimensional Hugging Face Embeddings**
* 🛡️ **Strict Document Grounding**
* 🚫 **Hallucination Mitigation**
* 🎨 **Modern Glassmorphism UI**
* 🌙 **Dark-themed Streamlit Interface**
* 📱 **Responsive User Interface**

---

# 🏗️ Architecture

The application uses two independent AI engines.

```text
                         ┌─────────────────────┐
                         │   Umer PDF Assistant │
                         └──────────┬──────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
              ┌─────▼─────┐                   ┌─────▼─────┐
              │  RAG Mode │                   │  AI Mode  │
              └─────┬─────┘                   └─────┬─────┘
                    │                               │
             PDF Upload                        User Query
                    │                               │
             PyPDFLoader                       Chat Memory
                    │                               │
             Text Splitting                         │
                    │                               │
              Embeddings                             │
                    │                               │
               ChromaDB                              │
                    │                               │
               MMR Search                            │
                    │                               │
             Retrieved Context                        │
                    │                               │
                    └──────────────┬────────────────┘
                                   │
                              LLM Response
                                   │
                              Streamlit UI
```

---

# 🔎 RAG Pipeline

The RAG pipeline follows:

```text
PDF
 │
 ▼
PyPDFLoader
 │
 ▼
Text Extraction
 │
 ▼
RecursiveCharacterTextSplitter
 │
 ▼
Document Chunks
 │
 ▼
Hugging Face Embeddings
 │
 ▼
ChromaDB
 │
 ▼
MMR Retriever
 │
 ▼
Relevant Context
 │
 ▼
Grounded Prompt
 │
 ▼
LLM
 │
 ▼
Answer + Sources
```

---

# 📚 Document Ingestion

PDF files are processed using LangChain's `PyPDFLoader`.

Each page is extracted together with metadata such as:

* Source filename
* Page number
* Page content

The extracted text is then divided into smaller chunks using:

```text
Chunk Size   = 1000 characters
Chunk Overlap = 200 characters
```

The overlap helps preserve context when important information crosses chunk boundaries.

---

# 🧠 Embedding Model

The project uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

through LangChain's `HuggingFaceEmbeddings`.

Each chunk is transformed into a:

```text
384-dimensional dense vector
```

These vectors allow the system to perform **semantic similarity search**, meaning that the system can retrieve conceptually relevant information rather than relying only on exact keyword matches.

---

# 🗄️ Vector Database

The application uses **ChromaDB** through LangChain's Chroma integration.

The vector store contains:

```text
Embedding
   +
Document Content
   +
Metadata
```

Metadata includes information such as:

```text
filename
page number
source
```

The project supports both:

* **Ephemeral / session-based vector stores**
* **Persistent vector stores**

Persistent knowledge is stored in:

```text
Chroma_DB/
```

This allows previously indexed documents to be reused without rebuilding the entire vector database.

---

# 🎯 Advanced Retrieval with MMR

Instead of relying solely on traditional similarity search, the application uses **Maximal Marginal Relevance (MMR)**.

MMR balances:

```text
Relevance to Query
        +
Diversity Between Results
```

Current configuration:

```text
k = 4
fetch_k = 10
lambda_mult = 0.5
```

### Why MMR?

Traditional similarity search can return several chunks containing almost identical information.

MMR reduces this redundancy and attempts to provide a more diverse set of useful context.

```text
User Query
    │
    ▼
10 Candidate Chunks
    │
    ▼
MMR Selection
    │
    ├── Relevant Chunk
    ├── Diverse Chunk
    ├── Relevant Chunk
    └── Diverse Chunk
    │
    ▼
4 Final Chunks
```

---

# 🛡️ Hallucination Mitigation

RAG Mode is intentionally designed to be **strictly grounded in the uploaded document**.

The system prompt instructs the LLM to:

1. Use only the retrieved document context.
2. Avoid unsupported claims.
3. Not use external knowledge when answering document questions.
4. Clearly state when the required information cannot be found.

When sufficient evidence is unavailable, the assistant uses the fallback:

```text
I could not find the answer in the document.
```

This design significantly reduces the risk of generating unsupported answers.

---

# 🤖 Dual-Engine Architecture

## 📚 RAG Mode

RAG Mode is designed for questions such as:

```text
What is the main conclusion of this document?

Explain the methodology used in the research.

What does page 12 say about the proposed system?
```

The answer is generated using retrieved document context and includes source information.

### RAG Mode Configuration

```text
Temperature = 0.2
Retrieval  = MMR
Top Chunks = 4
```

The low temperature helps make responses more deterministic and fact-oriented.

---

## 💬 AI Mode

AI Mode is designed for general-purpose interaction.

Examples:

```text
Explain recursion in C++.

Write a Python program for binary search.

Explain transformer architecture.

Help me debug this code.

Write a professional email.
```

AI Mode uses **Mistral AI** and maintains rolling conversation memory.

The application keeps up to:

```text
6 recent conversation turns
```

This allows the assistant to maintain conversational context without continuously growing the prompt indefinitely.

---

# 📌 Citation & Source Traceability

One of the important features of the project is **source traceability**.

RAG responses can be accompanied by source cards containing:

```text
📄 File Name
📑 Page Number
📝 Retrieved Text
```

This makes it easier for users to verify where an answer came from.

The overall flow is:

```text
Retrieved Chunk
      │
      ├── Filename
      ├── Page Number
      └── Text
            │
            ▼
      Source Card
```

This improves the **auditability and transparency** of the generated responses.

---

# 🎨 User Interface

The application is built using **Streamlit** with a custom CSS design system.

The UI follows a modern **glassmorphism-inspired dark theme**.

### UI Features

* Glass-effect cards
* Responsive message bubbles
* Radial gradients
* Custom badges
* Dark theme
* Google Fonts
* Code-friendly typography
* Clean chat interface
* Mode switching

Fonts used:

```text
Outfit
JetBrains Mono
```

---

# 🧰 Technology Stack

| Technology                     | Purpose                   |
| ------------------------------ | ------------------------- |
| Python                         | Core programming language |
| LangChain                      | LLM and RAG orchestration |
| Mistral AI                     | Conversational LLM        |
| Hugging Face                   | Text embeddings           |
| Sentence Transformers          | Embedding model           |
| ChromaDB                       | Vector database           |
| PyPDFLoader                    | PDF ingestion             |
| RecursiveCharacterTextSplitter | Document chunking         |
| Streamlit                      | User interface            |
| Python-dotenv                  | Environment configuration |

---

# 📦 Core Dependencies

The project uses packages such as:

```text
langchain
langchain-core
langchain-community
langchain-chroma
langchain-huggingface
langchain-mistralai
sentence-transformers
chromadb
pypdf
streamlit
python-dotenv
```

---

# 📁 Project Structure

A typical project structure is:

```text
Umer-PDF-Assistant/
│
├── app.py
│
├── src/
│   ├── config.py
│   ├── embeddings.py
│   ├── loaders.py
│   ├── splitter.py
│   ├── vectorstore.py
│   ├── retriever.py
│   ├── prompts.py
│   ├── chains.py
│   ├── memory.py
│   └── utils.py
│
├── Chroma_DB/
│
├── .env
├── .gitignore
├── requirements.txt
├── README.md
└── pyproject.toml
```

> The exact structure can be adjusted according to the final implementation.

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone https://github.com/your-username/umer-pdf-assistant.git
cd umer-pdf-assistant
```

## 2. Create a Python 3.10 Virtual Environment

This project is intended to run with Python 3.10.

Using `uv`:

```bash
uv python install 3.10
uv venv --python 3.10
```

Activate the environment.

### Windows PowerShell

```powershell
.venv\Scripts\Activate.ps1
```

---

# 📥 Install Dependencies

Using `uv`:

```bash
uv pip install -r requirements.txt
```

Or install the project dependencies directly:

```bash
uv add langchain langchain-core langchain-community langchain-chroma langchain-huggingface langchain-mistralai sentence-transformers chromadb pypdf streamlit python-dotenv
```

---

# 🔐 Environment Variables

Create a `.env` file in the project root:

```env
MISTRAL_API_KEY=your_mistral_api_key
```

If Gemini is also used in the project:

```env
GEMINI_API_KEY=your_gemini_api_key
```

Never commit your `.env` file.

Add it to `.gitignore`:

```gitignore
.env
.venv/
__pycache__/
Chroma_DB/
*.pyc
```

---

# ▶️ Running the Application

Start Streamlit with:

```bash
streamlit run app.py
```

The application will open in your browser.

---

# 🚀 How to Use

## RAG Mode

1. Launch the application.
2. Select **RAG Mode**.
3. Upload a PDF.
4. The application extracts the document text.
5. Text is split into chunks.
6. Chunks are converted into embeddings.
7. Embeddings are stored in ChromaDB.
8. Ask questions about the document.
9. The retriever selects relevant chunks using MMR.
10. The LLM generates a grounded response.
11. Source information is displayed with the answer.

---

## AI Mode

1. Select **AI Mode**.
2. Enter your question.
3. The assistant processes the request using Mistral AI.
4. Previous conversation turns are retained.
5. Continue the conversation naturally.

---

# 🔄 Complete System Flow

```text
                    USER
                     │
                     ▼
              Streamlit Interface
                     │
             ┌───────┴────────┐
             │                │
             ▼                ▼
         RAG MODE          AI MODE
             │                │
             ▼                ▼
         PDF Upload       User Prompt
             │                │
             ▼                ▼
       PyPDFLoader        Chat Memory
             │                │
             ▼                │
       Text Chunking           │
             │                │
             ▼                │
        Embeddings             │
             │                │
             ▼                │
          ChromaDB             │
             │                │
             ▼                │
          MMR Search           │
             │                │
             ▼                │
       Retrieved Context       │
             │                │
             └───────┬────────┘
                     │
                     ▼
                    LLM
                     │
                     ▼
              Generated Answer
                     │
                     ▼
             Streamlit Response
```

---

# 🧪 Design Principles

The project follows several important AI engineering principles:

### 1. Grounding

The RAG engine should answer from retrieved evidence rather than relying blindly on the LLM's internal knowledge.

### 2. Retrieval Quality

MMR is used to improve diversity and reduce redundant retrieved chunks.

### 3. Traceability

Retrieved document metadata is preserved so users can identify the source of generated answers.

### 4. Separation of Concerns

Document retrieval and general-purpose AI conversation are separated into different modes.

### 5. Controlled Context

Conversation memory is limited to a fixed number of turns to control context size.

### 6. Persistent Knowledge

ChromaDB persistence allows indexed knowledge to survive application restarts.

---

# 🔮 Future Improvements

Potential future versions can extend the system with:

* 🔹 Multi-PDF knowledge bases
* 🔹 PDF page previews
* 🔹 Hybrid BM25 + vector retrieval
* 🔹 Reranking models
* 🔹 Query rewriting
* 🔹 Conversational RAG
* 🔹 Streaming LLM responses
* 🔹 Document deletion and management
* 🔹 Multiple embedding models
* 🔹 Multi-modal PDF processing
* 🔹 OCR for scanned PDFs
* 🔹 Advanced metadata filtering
* 🔹 Evaluation with RAG metrics
* 🔹 LangGraph-based agentic workflows
* 🔹 Authentication and user accounts
* 🔹 Cloud deployment

---

# 📊 Project Goals

The primary goal of **Umer PDF Assistant** is to demonstrate how modern AI applications can combine:

```text
LLMs
 +
Embeddings
 +
Vector Databases
 +
Semantic Retrieval
 +
Prompt Engineering
 +
Source Attribution
 +
Conversational Memory
 +
Modern UI
```

into a practical end-to-end AI application.

---

# 👨‍💻 Author

**Umer Nawaz**

Computer Science Undergraduate
Focused on:

```text
Artificial Intelligence
Machine Learning
Deep Learning
LLMs
RAG
Agentic AI
```

---

# ⭐ Project Status

```text
Status: Active Development
Version: 0.x
```

The architecture is intentionally designed to be extensible toward more advanced **RAG and Agentic AI systems**.

---

# 📜 License

This project is intended for educational and development purposes.

Add an appropriate open-source license before publicly distributing the project.
