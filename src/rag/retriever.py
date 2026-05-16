from src.rag.embedder import load_vector_store


def retrieve_context(query: str, k: int = 4) -> str:
    vector_store = load_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": k})

    docs = retriever.invoke(query)

    if not docs:
        return "No relevant context found in the knowledge base."

    context_parts = []
    for i, doc in enumerate(docs):
        context_parts.append(f"--- Chunk {i+1} ---\n{doc.page_content}")

    return "\n\n".join(context_parts)