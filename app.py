import os
import time
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

for key in ("GROQ_API_KEY", "SERPER_API_KEY", "HUGGINGFACE_HUB_TOKEN"):
    if key not in os.environ:
        try:
            os.environ[key] = st.secrets[key]
        except KeyError:
            pass

st.set_page_config(
    page_title="AI Research Agent",
    page_icon="assets/icon.png" if False else None,
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0a0a0a;
    color: #e8e8e8;
}

.stApp {
    background-color: #0a0a0a;
}

h1, h2, h3 {
    font-family: 'IBM Plex Mono', monospace;
    color: #f0f0f0;
    letter-spacing: -0.5px;
}

.hero-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2.4rem;
    font-weight: 600;
    color: #ffffff;
    line-height: 1.2;
    margin-bottom: 0.3rem;
}

.hero-sub {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 1rem;
    color: #888;
    margin-bottom: 2rem;
    font-weight: 300;
}

.agent-card {
    background: #141414;
    border: 1px solid #2a2a2a;
    border-radius: 8px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.8rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
    color: #aaa;
}

.agent-card .label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #00e5a0;
    margin-bottom: 0.3rem;
}

.agent-card .name {
    font-size: 0.95rem;
    color: #fff;
    font-weight: 600;
    margin-bottom: 0.2rem;
}

.status-idle    { color: #555; }
.status-running { color: #f5a623; }
.status-done    { color: #00e5a0; }
.status-error   { color: #ff4b4b; }

.report-box {
    background: #111;
    border: 1px solid #222;
    border-left: 3px solid #00e5a0;
    border-radius: 6px;
    padding: 1.5rem 2rem;
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.92rem;
    line-height: 1.75;
    color: #ddd;
    white-space: pre-wrap;
}

.tag {
    display: inline-block;
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 4px;
    padding: 0.2rem 0.7rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: #00e5a0;
    margin-right: 0.4rem;
    margin-bottom: 0.4rem;
}

.metric-row {
    display: flex;
    gap: 1rem;
    margin-bottom: 1.5rem;
}

.metric-box {
    background: #141414;
    border: 1px solid #2a2a2a;
    border-radius: 6px;
    padding: 0.9rem 1.2rem;
    flex: 1;
    font-family: 'IBM Plex Mono', monospace;
}

.metric-box .val {
    font-size: 1.5rem;
    font-weight: 600;
    color: #00e5a0;
}

.metric-box .lbl {
    font-size: 0.7rem;
    color: #555;
    text-transform: uppercase;
    letter-spacing: 1px;
}

div[data-testid="stTextArea"] textarea {
    background: #141414;
    border: 1px solid #2a2a2a;
    border-radius: 6px;
    color: #e8e8e8;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.88rem;
}

div[data-testid="stButton"] button {
    background: #00e5a0;
    color: #000;
    border: none;
    border-radius: 6px;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    font-size: 0.85rem;
    padding: 0.6rem 2rem;
    letter-spacing: 0.5px;
    width: 100%;
}

div[data-testid="stButton"] button:hover {
    background: #00c88a;
    color: #000;
}

div[data-testid="stDownloadButton"] button {
    background: #1a1a1a;
    color: #00e5a0;
    border: 1px solid #2a2a2a;
    border-radius: 6px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
    width: 100%;
}

.stSpinner > div {
    border-top-color: #00e5a0 !important;
}

hr {
    border-color: #1e1e1e;
    margin: 1.5rem 0;
}
</style>
""", unsafe_allow_html=True)


def run_crew(topic: str):
    from src.rag.embedder import build_vector_store
    from src.agents import get_researcher_agent, get_rag_agent, get_writer_agent
    from src.tasks import get_research_task, get_rag_task, get_writer_task
    from crewai import Crew, Process

    build_vector_store()

    researcher = get_researcher_agent()
    rag_agent = get_rag_agent()
    writer = get_writer_agent()

    research_task = get_research_task(topic, researcher)
    rag_task = get_rag_task(topic, rag_agent, research_task)
    writer_task = get_writer_task(topic, writer, research_task, rag_task)

    crew = Crew(
        agents=[researcher, rag_agent, writer],
        tasks=[research_task, rag_task, writer_task],
        process=Process.sequential,
        verbose=False
    )

    time.sleep(15)

    for attempt in range(1, 4):
        try:
            result = crew.kickoff()
            return result
        except Exception as e:
            error_msg = str(e).lower()
            if "rate_limit" in error_msg or "429" in error_msg or "ratelimit" in error_msg:
                time.sleep(60 * attempt)
            else:
                raise


col_left, col_right = st.columns([1, 2], gap="large")

with col_left:
    st.markdown('<div class="hero-title">AI Research<br/>Agent</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Multi-agent RAG system powered by CrewAI + Groq</div>', unsafe_allow_html=True)

    st.markdown('<div class="tag">CrewAI</div><div class="tag">RAG</div><div class="tag">ChromaDB</div><div class="tag">Groq</div><div class="tag">LangChain</div>', unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown("**Research Topic**")
    topic = st.text_area(
        label="topic",
        label_visibility="collapsed",
        value="Indian fintech credit risk and AI in banking 2025",
        height=90
    )

    run_btn = st.button("Run Agent Crew")

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown("**Agents**")

    if "agent_status" not in st.session_state:
        st.session_state.agent_status = {
            "researcher": "idle",
            "rag": "idle",
            "writer": "idle"
        }

    status_icons = {
        "idle": "○",
        "running": "◉",
        "done": "●",
        "error": "✕"
    }

    for key, label, role in [
        ("researcher", "Researcher", "Web search + source collection"),
        ("rag", "RAG Specialist", "Vector store retrieval"),
        ("writer", "Report Writer", "Synthesis + report generation")
    ]:
        s = st.session_state.agent_status[key]
        icon = status_icons[s]
        st.markdown(f"""
        <div class="agent-card">
            <div class="label">{icon} {s.upper()}</div>
            <div class="name">{label}</div>
            {role}
        </div>
        """, unsafe_allow_html=True)

with col_right:
    if "report" not in st.session_state:
        st.session_state.report = None
    if "run_time" not in st.session_state:
        st.session_state.run_time = None

    if run_btn:
        if not topic.strip():
            st.error("Please enter a research topic.")
        else:
            st.session_state.report = None
            st.session_state.agent_status = {
                "researcher": "running",
                "rag": "idle",
                "writer": "idle"
            }

            start = time.time()

            with st.spinner("Agents are working... this takes 2 to 4 minutes on the free tier"):
                try:
                    result = run_crew(topic.strip())
                    st.session_state.run_time = round(time.time() - start)

                    try:
                        with open("outputs/report.md", "r") as f:
                            st.session_state.report = f.read()
                    except Exception:
                        st.session_state.report = str(result)

                    st.session_state.agent_status = {
                        "researcher": "done",
                        "rag": "done",
                        "writer": "done"
                    }
                    st.rerun()

                except Exception as e:
                    st.session_state.agent_status = {
                        "researcher": "error",
                        "rag": "error",
                        "writer": "error"
                    }
                    st.error(f"Error: {str(e)[:300]}")

    if st.session_state.report:
        run_time = st.session_state.run_time or 0
        word_count = len(st.session_state.report.split())

        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-box"><div class="val">{word_count}</div><div class="lbl">Words Generated</div></div>
            <div class="metric-box"><div class="val">{run_time}s</div><div class="lbl">Run Time</div></div>
            <div class="metric-box"><div class="val">3</div><div class="lbl">Agents Used</div></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Generated Report**")
        st.markdown(f'<div class="report-box">{st.session_state.report}</div>', unsafe_allow_html=True)

        st.markdown("<br/>", unsafe_allow_html=True)
        st.download_button(
            label="Download report.md",
            data=st.session_state.report,
            file_name="fintech_ai_report.md",
            mime="text/markdown"
        )
    else:
        st.markdown("""
        <div style="
            height: 400px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            border: 1px dashed #222;
            border-radius: 8px;
            color: #333;
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.85rem;
            text-align: center;
            line-height: 2;
        ">
            Enter a topic on the left<br/>
            and click Run Agent Crew<br/><br/>
            <span style="font-size: 2rem; opacity: 0.3">◎</span>
        </div>
        """, unsafe_allow_html=True)