# AI Research Agent - Multi-Agent RAG System for Fintech Intelligence

A production-style multi-agent AI system built with CrewAI, RAG, and ChromaDB
that autonomously researches, retrieves, and generates structured reports on
Indian fintech and banking topics.

---

## What This Project Does

Three specialized AI agents collaborate in sequence to produce a research report:

1. Researcher Agent - searches the web for latest news and data using Serper API
2. RAG Agent - queries a ChromaDB vector store built from internal fintech documents
3. Writer Agent - synthesizes both sources into a structured markdown report

All agents are powered by Groq (LLaMA 3.3 70B) and orchestrated via CrewAI.

---

## Architecture

```
User Input (topic)
      |
      v
Researcher Agent  -->  Web Search (Serper API)
      |
      v
RAG Agent  -->  ChromaDB Vector Store (LangChain + HuggingFace Embeddings)
      |
      v
Writer Agent  -->  Structured Report (outputs/report.md)
```

---

## Tech Stack

| Component | Tool |
|---|---|
| Agent Framework | CrewAI |
| LLM | Groq - LLaMA 3.3 70B Versatile |
| Vector Store | ChromaDB |
| Embeddings | HuggingFace all-MiniLM-L6-v2 |
| Document Loaders | LangChain Community |
| Web Search | Serper API |
| Language | Python 3.11 |

---

## Project Structure

```
ai-research-agent/
├── main.py                  # Entry point - runs the full crew
├── requirements.txt
├── .env                     # API keys (not committed)
├── data/
│   └── sample_docs/         # Domain knowledge base (.txt files)
├── outputs/
│   └── report.md            # Generated report
└── src/
    ├── agents.py            # CrewAI agent definitions
    ├── tasks.py             # Task definitions with context chaining
    ├── rag/
    │   ├── embedder.py      # ChromaDB vector store builder
    │   └── retriever.py     # Semantic retrieval logic
    └── tools/
        ├── search_tool.py   # Serper web search wrapper
        └── rag_tool.py      # CrewAI-compatible RAG tool
```

---

## Setup and Run

### 1. Clone the repo

```bash
git clone https://github.com/richakumari0311/ai-research-agent.git
cd ai-research-agent
```

### 2. Create virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Add API keys to .env

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

```bash
python main.py "Indian fintech credit risk and AI in banking 2025"
```

Report is saved to outputs/report.md

---

## Sample Output

The system generates structured reports with:
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
- Rate limit handling with automatic retry and backoff
- Separation of concerns across agents, tasks, tools, and RAG modules

---

## Why This Matters

Indian banks and fintech companies are actively building internal RAG
systems to query regulatory documents, earnings reports, and credit
policy manuals. This project demonstrates the exact architecture used
in production at companies like Navi, Zerodha, and ICICI Bank AI teams.

Skills demonstrated: CrewAI, RAG, ChromaDB, LangChain, Groq, multi-agent
design, tool use, prompt engineering, and Python engineering best practices.

---

## Limitations and Known Issues

- Groq free tier has token-per-minute limits; the system handles this
  with automatic retry and exponential backoff
- ChromaDB is local and in-memory; for production use Pinecone or Qdrant
- Report quality depends on Serper search result relevance

---

## Future Improvements

- Add memory across agent runs using CrewAI long-term memory
- Replace ChromaDB with Pinecone for cloud-native vector storage
- Add a FastAPI layer to expose the agent as a REST API
- Add FAISS as an alternative retriever for benchmarking
- Add evaluation metrics for RAG retrieval quality (MRR, NDCG)
- Support PDF ingestion for RBI circulars and annual reports

---

## Domain

Fintech and BFSI - Indian credit risk, AI in banking, UPI ecosystem,
regulatory AI (RBI), and lending intelligence.