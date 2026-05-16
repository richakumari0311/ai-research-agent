import os
import sys
import time
from crewai import Crew, Process
from dotenv import load_dotenv
from src.rag.embedder import build_vector_store
from src.agents import get_researcher_agent, get_rag_agent, get_writer_agent
from src.tasks import get_research_task, get_rag_task, get_writer_task

load_dotenv()


def run_agent(topic: str, max_retries: int = 3):
    print("=" * 60)
    print(f"AI Research Agent Starting")
    print(f"Topic: {topic}")
    print("=" * 60)

    print("\nStep 1: Building vector store from local documents...")
    build_vector_store()

    print("\nStep 2: Initializing agents...")
    researcher = get_researcher_agent()
    rag_agent = get_rag_agent()
    writer = get_writer_agent()

    print("\nStep 3: Creating tasks...")
    research_task = get_research_task(topic, researcher)
    rag_task = get_rag_task(topic, rag_agent, research_task)
    writer_task = get_writer_task(topic, writer, research_task, rag_task)

    print("\nStep 4: Assembling crew...")
    crew = Crew(
        agents=[researcher, rag_agent, writer],
        tasks=[research_task, rag_task, writer_task],
        process=Process.sequential,
        verbose=True
    )

    print("\nStep 5: Running crew...\n")
    print("Waiting 30 seconds to let rate limit window reset...")
    time.sleep(30)

    for attempt in range(1, max_retries + 1):
        try:
            print(f"\nAttempt {attempt} of {max_retries}...")
            result = crew.kickoff()
            print("\n" + "=" * 60)
            print("Crew finished. Report saved to outputs/report.md")
            print("=" * 60)
            return result

        except Exception as e:
            error_msg = str(e).lower()
            if "rate_limit" in error_msg or "429" in error_msg or "ratelimit" in error_msg:
                wait_time = 60 * attempt
                print(f"\nRate limit hit. Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print(f"\nUnexpected error: {e}")
                raise

    print("Max retries reached. Try running again in a few minutes.")
    return None


if __name__ == "__main__":
    if len(sys.argv) > 1:
        topic = " ".join(sys.argv[1:])
    else:
        topic = "Indian fintech credit risk and AI in banking 2025"

    run_agent(topic)