import os
from typing import Any

import requests
import streamlit as st
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain.tools import tool

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
load_dotenv()

st.set_page_config(
    page_title="AI Agent • Search + Weather",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
WEATHERSTACK_API_KEY = os.getenv("WEATHERSTACK_API_KEY")

# ---------------------------------------------------------
# Custom styling
# ---------------------------------------------------------
st.markdown(
    """
    <style>
        .stApp {
            background:
                radial-gradient(circle at 10% 10%, rgba(99,102,241,.12), transparent 30%),
                radial-gradient(circle at 90% 20%, rgba(14,165,233,.10), transparent 28%),
                #0b1020;
        }

        .block-container {
            max-width: 1200px;
            padding-top: 2rem;
            padding-bottom: 5rem;
        }

        .hero {
            padding: 28px 30px;
            border: 1px solid rgba(255,255,255,.10);
            border-radius: 24px;
            background: linear-gradient(135deg, rgba(30,41,59,.92), rgba(15,23,42,.78));
            box-shadow: 0 20px 60px rgba(0,0,0,.25);
            margin-bottom: 22px;
        }

        .hero h1 {
            margin: 0;
            font-size: 2.35rem;
            letter-spacing: -1px;
        }

        .hero p {
            margin: 8px 0 0;
            color: #aab4c8;
            font-size: 1.02rem;
        }

        .badge {
            display: inline-block;
            padding: 5px 10px;
            margin-bottom: 10px;
            border-radius: 999px;
            background: rgba(99,102,241,.16);
            border: 1px solid rgba(129,140,248,.25);
            color: #c7d2fe;
            font-size: .82rem;
        }

        .status-card {
            padding: 14px 16px;
            border-radius: 16px;
            background: rgba(255,255,255,.045);
            border: 1px solid rgba(255,255,255,.08);
            margin-bottom: 10px;
        }

        .small {
            color: #94a3b8;
            font-size: .85rem;
        }

        [data-testid="stChatMessage"] {
            border-radius: 18px;
            border: 1px solid rgba(255,255,255,.06);
        }

        .tool-card {
            padding: 12px 14px;
            border-radius: 14px;
            background: rgba(15,23,42,.75);
            border: 1px solid rgba(255,255,255,.07);
            margin: 8px 0;
        }

        div[data-testid="stSidebar"] {
            background: #080d1a;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Tools
# ---------------------------------------------------------
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city using Weatherstack."""
    api_key = os.getenv("WEATHERSTACK_API_KEY")

    if not api_key:
        return "Weather is unavailable because WEATHERSTACK_API_KEY is not configured."

    try:
        response = requests.get(
            "https://api.weatherstack.com/current",
            params={"access_key": api_key, "query": city},
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()

        if "error" in data:
            error_info = data.get("error", {})
            return f"Weather API error: {error_info.get('info', 'Unknown error')}"

        current = data.get("current")
        location = data.get("location", {})

        if not current:
            return f"Could not find current weather for {city}."

        description = (
            current.get("weather_descriptions") or ["Not available"]
        )[0]

        return (
            f"City: {location.get('name', city)}\n"
            f"Region: {location.get('region', '')}\n"
            f"Country: {location.get('country', '')}\n"
            f"Temperature: {current.get('temperature', 'N/A')}°C\n"
            f"Feels like: {current.get('feelslike', 'N/A')}°C\n"
            f"Weather: {description}\n"
            f"Humidity: {current.get('humidity', 'N/A')}%\n"
            f"Wind: {current.get('wind_speed', 'N/A')} km/h"
        )

    except requests.RequestException as exc:
        return f"Unable to reach the weather service: {exc}"
    except ValueError:
        return "Weather service returned invalid data."


@st.cache_resource(show_spinner=False)
def build_agent():
    """Create the LangChain agent once and reuse it."""
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is missing in the .env file.")
    if not TAVILY_API_KEY:
        raise RuntimeError("TAVILY_API_KEY is missing in the .env file.")

    # langchain-tavily uses max_results.
    search_tool = TavilySearch(max_results=5)

    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        api_key=GROQ_API_KEY,
        temperature=0.2,
    )

    return create_agent(
        model=llm,
        tools=[search_tool, get_weather],
        system_prompt=(
            "You are a helpful AI agent. "
            "Answer clearly and accurately. "
            "Use web search for current, recent, or factual information that "
            "requires external sources. Use the weather tool for current weather. "
            "Do not claim to have searched or checked something unless you actually "
            "used the appropriate tool."
        ),
    )


def extract_final_text(result: dict[str, Any]) -> str:
    """Safely extract the final assistant message from a LangChain agent result."""
    messages = result.get("messages", [])

    for message in reversed(messages):
        content = getattr(message, "content", None)

        if content is None and isinstance(message, dict):
            content = message.get("content")

        if isinstance(content, str) and content.strip():
            return content

        # Some model integrations return a list of content blocks.
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict) and block.get("text"):
                    parts.append(str(block["text"]))
                elif isinstance(block, str):
                    parts.append(block)
            text = "\n".join(parts).strip()
            if text:
                return text

    return "I couldn't generate a response."


def run_agent(query: str) -> str:
    agent = build_agent()

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": query,
                }
            ]
        }
    )

    return extract_final_text(result)


# ---------------------------------------------------------
# Session state
# ---------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("## 🤖 AI Agent")
    st.caption("Groq + LangChain + Tavily + Weatherstack")

    st.markdown("### Connection status")

    if GROQ_API_KEY:
        st.markdown('<div class="status-card">🟢 Groq API configured</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-card">🔴 Groq API missing</div>', unsafe_allow_html=True)

    if TAVILY_API_KEY:
        st.markdown('<div class="status-card">🟢 Tavily Search configured</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-card">🔴 Tavily API missing</div>', unsafe_allow_html=True)

    if WEATHERSTACK_API_KEY:
        st.markdown('<div class="status-card">🟢 Weather configured</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-card">🟡 Weather unavailable</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Try these")
    examples = [
        "What is an LLM?",
        "What are the latest AI developments?",
        "Find the current weather in Ongole",
        "Compare LangChain and LangGraph",
        "What are the benefits of agentic AI?",
    ]

    for example in examples:
        if st.button(example, use_container_width=True):
            st.session_state.pending_prompt = example

    st.markdown("---")
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown(
        '<p class="small">Your API keys stay in your local .env file.</p>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------
# Main UI
# ---------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <div class="badge">SINGLE AI AGENT</div>
        <h1>Ask anything. Let the agent choose the tool.</h1>
        <p>
            A clean AI assistant powered by Groq, with web search and live weather tools.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Sidebar example click
pending_prompt = st.session_state.pop("pending_prompt", None)

prompt = st.chat_input("Ask your AI agent anything…")

if pending_prompt and not prompt:
    prompt = pending_prompt

if prompt:
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking and selecting the right tool…"):
            try:
                answer = run_agent(prompt)
            except Exception as exc:
                answer = (
                    "### ⚠️ Agent error\n\n"
                    f"`{type(exc).__name__}: {exc}`\n\n"
                    "Check your `.env` API keys and installed packages."
                )

        st.markdown(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
