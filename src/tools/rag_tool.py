from crewai.tools import BaseTool
from src.rag.retriever import retrieve_context


class RAGTool(BaseTool):
    name: str = "Knowledge Base Search"
    description: str = (
        "Search the internal knowledge base using semantic similarity. "
        "Use this to retrieve relevant context about RAG, AI agents, "
        "ChromaDB, FAISS, CrewAI, or any topic stored in the vector store. "
        "Input should be a plain text question or topic."
    )

    def _run(self, query: str) -> str:
        return retrieve_context(query)