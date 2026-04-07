import logging
from langchain_ollama import ChatOllama, OllamaEmbeddings
from config import EMBED_MODEL, LLM_MODEL, TEMPERATURE, VISION_MODEL

logger = logging.getLogger(__name__)

def get_llm():
    logger.debug("get_llm: model=%s temperature=%s", LLM_MODEL, TEMPERATURE)
    return ChatOllama(
        model=LLM_MODEL,
        temperature=TEMPERATURE,
    )


def get_embeddings():
    logger.info("get_embeddings: model=%s", EMBED_MODEL)
    return OllamaEmbeddings(
        model=EMBED_MODEL,
    )


def get_vision_llm():
    logger.debug("get_vision_llm: model=%s", VISION_MODEL)
    return ChatOllama(
        model=VISION_MODEL,
        temperature=0,
    )
