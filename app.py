import hashlib
import logging
import time

import streamlit as st
from langgraph.checkpoint.memory import MemorySaver

from config import setup_logging
from ollama_check import OllamaModelsError, ensure_ollama_models

setup_logging()

logger = logging.getLogger(__name__)

st.set_page_config(page_title="ollama-llm-streamlit-agent")

try:
    ensure_ollama_models()
except OllamaModelsError as e:
    st.error(str(e))
    st.stop()

from agent import create_agent
from rag import create_vector_store

st.title("🧠 Ollama LLM Streamlit Agent")

if "checkpointer" not in st.session_state:
    st.session_state.checkpointer = MemorySaver()

if "vectordb" not in st.session_state:
    st.session_state.vectordb = None

if "agent" not in st.session_state:
    logger.info("Streamlit: creating agent (first load)")
    st.session_state.agent = create_agent(
        st.session_state.vectordb,
        st.session_state.checkpointer,
    )


# File upload
uploaded_file = st.file_uploader("Upload text file for RAG")

if uploaded_file:
    # UploadedFile is a stream: without seek(0), later reruns read 0 bytes and wipe RAG.
    uploaded_file.seek(0)
    raw = uploaded_file.read()
    name = getattr(uploaded_file, "name", "upload")
    content_hash = hashlib.sha256(raw).hexdigest()
    if st.session_state.get("rag_content_hash") != content_hash:
        text = raw.decode()
        logger.info(
            "RAG upload (new or replaced file): file=%r bytes=%s chars=%s",
            name,
            len(raw),
            len(text),
        )
        st.session_state.vectordb = create_vector_store(text)
        st.session_state.agent = create_agent(
            st.session_state.vectordb,
            st.session_state.checkpointer,
        )
        st.session_state.rag_content_hash = content_hash
        logger.info("Streamlit: agent rebuilt with RAG")
        st.success("Document loaded")
    else:
        logger.debug(
            "RAG: reusing indexed corpus (same bytes as last index), file=%r",
            name,
        )


user_input = st.chat_input("Ask something...")

if user_input:
    preview = user_input if len(user_input) <= 500 else user_input[:500] + "…"
    logger.info(
        "Chat invoke: chars=%s preview=%r thread_id=streamlit_chat",
        len(user_input),
        preview,
    )
    t0 = time.perf_counter()
    result = st.session_state.agent.invoke(
        {"messages": [{"role": "user", "content": user_input}]},
        {"configurable": {"thread_id": "streamlit_chat"}},
    )
    elapsed = time.perf_counter() - t0

    messages = result.get("messages", [])
    logger.info(
        "Chat done: %.2fs, result_messages=%s",
        elapsed,
        len(messages),
    )
    if logger.isEnabledFor(logging.DEBUG):
        for i, m in enumerate(messages):
            typ = type(m).__name__
            body = getattr(m, "content", str(m))
            if isinstance(body, str) and len(body) > 400:
                body = body[:400] + "…"
            logger.debug("  msg[%s] %s: %s", i, typ, body)

    last = messages[-1] if messages else None
    response = getattr(last, "content", None) if last is not None else str(result)
    if response is None:
        response = str(result)

    st.write("### 🤖")
    st.write(response)
