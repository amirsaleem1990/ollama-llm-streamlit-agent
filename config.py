from pathlib import Path

_ROOT = Path(__file__).resolve().parent

try:
    from dotenv import load_dotenv

    load_dotenv(_ROOT / ".env")
except ImportError:
    pass

import logging
import os
import sys

LLM_MODEL = "llama3.2"
EMBED_MODEL = "mxbai-embed-large"
VISION_MODEL = "llava"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

TEMPERATURE = 0.6

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def setup_logging() -> None:
    """Configure root logging once. Honors LOG_LEVEL (default INFO)."""
    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        stream=sys.stderr,
        force=True,
    )
    # Quiet noisy HTTP clients used by Ollama / embeddings
    for name in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)
