import logging
import os

from langchain.agents import create_agent as lc_create_agent
from langgraph.checkpoint.memory import MemorySaver

from llm import get_llm
from tools import analyze_image, build_search_documents_tool, calculator

logger = logging.getLogger(__name__)


def create_agent(vectordb=None, checkpointer=None):
    tools = [
        calculator,
        analyze_image,
    ]
    if vectordb is not None:
        tools.append(build_search_documents_tool(vectordb))

    # Verbose LangGraph node transitions (stderr); enable with AGENT_GRAPH_DEBUG=1
    graph_debug = os.environ.get("AGENT_GRAPH_DEBUG", "").lower() in (
        "1",
        "true",
        "yes",
    )

    if checkpointer is None:
        checkpointer = MemorySaver()

    system_prompt = (
        "You are a helpful assistant with access to calculator and image analysis tools. "
        "When the user asks about a local image file (path, screenshot, or picture on disk), "
        "you must call analyze_image with that path first and base your answer on its result. "
        "Do not invent PIL, NumPy, or OpenCV tutorials or code blocks as a substitute for that tool. "
        "Only show code if the user explicitly asks for code."
    )
    if vectordb is not None:
        system_prompt += (
            " You have access to search_documents: use it to answer questions "
            "about the user's loaded text."
        )

    logger.info(
        "Building agent: model from get_llm(), tools=%s, graph_debug=%s, rag=%s",
        [t.name for t in tools],
        graph_debug,
        vectordb is not None,
    )

    graph = lc_create_agent(
        model=get_llm(),
        tools=tools,
        system_prompt=system_prompt,
        checkpointer=checkpointer,
        debug=graph_debug,
    )

    logger.info("Agent graph compiled: %s", type(graph).__name__)
    return graph
