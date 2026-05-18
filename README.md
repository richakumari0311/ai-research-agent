# AI Research Agent

A production-style multi-agent RAG system that autonomously researches, retrieves, and generates structured reports on Indian fintech and banking topics. Built with CrewAI, ChromaDB, and Groq.

---

## What It Does

Three specialized AI agents collaborate in sequence:

1. **Researcher Agent** — searches the web for latest news and data via Serper API
2. **RAG Agent** — queries a ChromaDB vector store built from internal fintech documents
3. **Writer Agent** — synthesizes both sources into a structured markdown report

All agents are powered by Groq (LLaMA 3.3 70B) and orchestrated via CrewAI.

---

## Architecture

```
User Input (topic)
       │
       ▼
Researcher Agent  ──►  Web Search (Serper API)
       │
       ▼
RAG Agent  ──►  ChromaDB Vector Store (LangChain + HuggingFace Embeddings)
       │
       ▼
Writer Agent  ──►  Structured Report (outputs/report.md)
```

---

## Tech Stack

| Component         | Tool                          |
|-------------------|-------------------------------|
| Agent Framework   | CrewAI                        |
| LLM               | Groq — LLaMA 3.3 70B Versatile |
| Vector Store      | ChromaDB (in-memory)          |
| Embeddings        | HuggingFace all-MiniLM-L6-v2  |
| Document Loaders  | LangChain Community           |
| Web Search        | Serper API                    |
| UI                | Streamlit                     |
| Language          | Python 3.11                   |

---

## Project Structure

```
ai-research-agent/
├── app.py                   # Streamlit web UI entry point
├── main.py                  # CLI entry point
├── requirements.txt
├── .gitignore
├── README.md
├── data/
│   └── sample_docs/         # Domain knowledge base (.txt files)
├── outputs/
│   └── report.md            # Generated at runtime (gitignored)
└── src/
    ├── __init__.py
    ├── agents.py            # CrewAI agent definitions
    ├── tasks.py             # Task definitions with context chaining
    ├── rag/
    │   ├── __init__.py
    │   ├── embedder.py      # ChromaDB vector store builder
    │   └── retriever.py     # Semantic retrieval logic
    └── tools/
        ├── __init__.py
        └── rag_tool.py      # CrewAI-compatible RAG tool
```

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/richakumari0311/ai-research-agent.git
cd ai-research-agent
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Add API keys

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key
SERPER_API_KEY=your_serper_api_key
HUGGINGFACE_HUB_TOKEN=your_huggingface_token
```

Get free keys at:
- Groq: https://console.groq.com
- Serper: https://serper.dev
- HuggingFace: https://huggingface.co/settings/tokens

### 4. Run the agent

**Streamlit UI (recommended):**
```bash
streamlit run app.py
```

**CLI:**
```bash
python main.py "Indian fintech credit risk and AI in banking 2025"
```

The report is saved to `outputs/report.md`.

---

## Report Structure

Every generated report includes:

- Executive Summary
- Key Findings from web research
- Knowledge Base Insights from RAG retrieval
- Tools and Technologies
- Industry Trends and Outlook
- Sources

---

## Key Concepts Demonstrated

- Multi-agent orchestration with role-based specialization
- RAG pipeline with semantic chunking and vector retrieval
- Tool use and agent-tool binding in CrewAI
- Context chaining between tasks (task output feeds next agent)
- Rate limit handling with automatic retry and exponential backoff
- Separation of concerns across agents, tasks, tools, and RAG modules

---

## Limitations

- Groq free tier has token-per-minute limits; the system retries automatically with exponential backoff on 429 errors
- ChromaDB runs in-memory; for production consider Pinecone or Qdrant
- Report quality depends on Serper search result relevance

---

## Future Improvements

- Add memory across agent runs using CrewAI long-term memory
- Replace ChromaDB with Pinecone for cloud-native vector storage
- Expose the agent as a REST API via FastAPI
- Add FAISS as an alternative retriever for benchmarking
- Add RAG evaluation metrics (MRR, NDCG)
- Support PDF ingestion for RBI circulars and annual reports

---

## Domain

Fintech and BFSI — Indian credit risk, AI in banking, UPI ecosystem, regulatory AI (RBI), and lending intelligence.

---

## Skills Demonstrated

CrewAI · RAG · ChromaDB · LangChain · Groq · Multi-agent design · Tool use · Prompt engineering · Python