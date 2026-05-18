import os
from crewai import Agent, LLM
from crewai_tools import SerperDevTool
from src.tools.rag_tool import RAGTool
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


def get_llm():
    llm = LLM(
        model="groq/llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),        
        temperature=0.1,
        max_tokens=500
    )
    return llm

def get_researcher_agent():
    llm = get_llm()
    search_tool = SerperDevTool()

    agent = Agent(
        role="Senior AI Research Specialist",
        goal=(
            "Search the web and find accurate, up to date information "
            "about the given topic. Focus on recent developments, "
            "practical use cases, and industry trends."
        ),
        backstory=(
            "You are an expert AI researcher with 10 years of experience "
            "tracking developments in fintech, credit risk, and AI in banking. "
            "You always find credible sources and summarize key insights clearly."
        ),
        tools=[search_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=3
    )
    return agent


def get_rag_agent():
    llm = get_llm()
    rag_tool = RAGTool()

    agent = Agent(
        role="Knowledge Base Specialist",
        goal=(
            "Search the internal knowledge base to retrieve relevant context "
            "about the topic. Combine retrieved chunks into a coherent summary "
            "that supports the final report."
        ),
        backstory=(
            "You are a specialist in information retrieval and RAG systems. "
            "You know how to extract the most relevant chunks from a vector store "
            "and present them in a way that is useful for report generation."
        ),
        tools=[rag_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=3
    )
    return agent


def get_writer_agent():
    llm = get_llm()

    agent = Agent(
        role="Technical Report Writer",
        goal=(
            "Write a clear, well structured, and insightful technical report "
            "using the research findings and knowledge base context provided. "
            "The report should be professional, accurate, and easy to read."
        ),
        backstory=(
            "You are a senior technical writer who specializes in fintech and AI topics. "
            "You have written hundreds of research reports for top banks and fintech companies. "
            "You always structure reports with an executive summary, key findings, "
            "and actionable conclusions."
        ),
        tools=[],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=2
    )
    return agent