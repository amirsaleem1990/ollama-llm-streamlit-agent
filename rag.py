import logging

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_OVERLAP, CHUNK_SIZE
from llm import get_embeddings

logger = logging.getLogger(__name__)


def create_vector_store(text):
    logger.info(
        "create_vector_store: input_chars=%s chunk_size=%s overlap=%s",
        len(text),
        CHUNK_SIZE,
        CHUNK_OVERLAP,
    )
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks = splitter.split_text(text)
    docs = [Document(page_content=t) for t in chunks]
    logger.info("create_vector_store: chunks=%s", len(docs))

    vectordb = Chroma.from_documents(
        docs,
        embedding=get_embeddings(),
    )

    logger.info("create_vector_store: Chroma index ready")
    return vectordb


def retrieve(vectordb, query):
    logger.debug("retrieve: query=%r", query)
    out = vectordb.similarity_search(query, k=3)
    logger.debug("retrieve: hits=%s", len(out))
    return out
