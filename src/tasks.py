from crewai import Task

def get_research_task(topic: str, researcher_agent) -> Task:
    task = Task(
        description=(
            f"Search the web for the latest information about: {topic}\n\n"
            "Your job:\n"
            "1. Run at least 2 different search queries related to the topic\n"
            "2. Collect key facts, statistics, tools, and trends\n"
            "3. Note the sources (URLs) where you found the information\n"
            "4. Summarize findings in clear bullet points\n\n"
            "Do not make up information. Only use what you find in search results."
        ),
        expected_output=(
            "A structured summary with:\n"
            "- 5 to 8 key findings about the topic\n"
            "- Relevant tools or technologies mentioned\n"
            "- At least 3 source URLs\n"
            "- Any notable trends or statistics"
        ),
        agent=researcher_agent
    )
    return task


def get_rag_task(topic: str, rag_agent, research_task: Task) -> Task:
    task = Task(
        description=(
            f"Search the internal knowledge base for context related to: {topic}\n\n"
            "Your job:\n"
            "1. Query the knowledge base with 2 to 3 different search queries\n"
            "2. Retrieve the most relevant chunks of information\n"
            "3. Identify how the internal knowledge connects to the web research\n"
            "4. Highlight any gaps or additional insights the knowledge base provides\n\n"
            "Use the Knowledge Base Search tool for every query."
        ),
        expected_output=(
            "A structured summary with:\n"
            "- Key context retrieved from the knowledge base\n"
            "- How it relates to the web research findings\n"
            "- Any unique insights not found in web search\n"
            "- The exact queries you used to retrieve context"
        ),
        agent=rag_agent,
        context=[research_task]
    )
    return task


def get_writer_task(topic: str, writer_agent, research_task: Task, rag_task: Task) -> Task:
    task = Task(
        description=(
            f"Write a professional technical report on the topic: {topic}\n\n"
            "Use the findings from the researcher and the knowledge base specialist.\n\n"
            "Write a report with these sections: Title, Executive Summary, "
            "Key Findings, Knowledge Base Insights, Tools and Technologies, "
            "Industry Trends, Conclusion, Sources. "
            "Use findings from the researcher and RAG agent. "
            "Be concise and professional."
        ),
        expected_output=(
            "A complete markdown formatted report covering all 8 sections. "
            "Minimum 400 words. Save ready for outputs/report.md."
        ),
        agent=writer_agent,
        context=[research_task, rag_task],
        output_file="outputs/report.md"
    )
    return task