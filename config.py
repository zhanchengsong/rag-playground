"""
All knobs live here. Everything you'll perturb in Milestone 3 (k, chunk size,
ordering) is a single edit in this file -- that's the whole point.

Override any of these with environment variables, e.g.:
    CHAT_BASE_URL=http://localhost:8080/v1 TOP_K=8 python rag.py
"""
import os

# --- Endpoints -------------------------------------------------------------
# llama.cpp's server (and MLX's server) speak the OpenAI API. The `api_key` is
# ignored by local servers but the client library requires *something*.
#
# Common setup: run the generation model and the embedding model as TWO
# servers on different ports. If you serve both from one process, just point
# both URLs at the same address.
API_KEY        = os.getenv("API_KEY", "local-no-key")
CHAT_BASE_URL  = os.getenv("CHAT_BASE_URL",  "http://localhost:8080/v1")
EMBED_BASE_URL = os.getenv("EMBED_BASE_URL", "http://localhost:8080/v1")

# Model names. llama.cpp largely ignores this field, so any string works; set
# it to whatever your MLX/vLLM server expects if you switch backends.
CHAT_MODEL  = os.getenv("CHAT_MODEL",  "qwen3")
EMBED_MODEL = os.getenv("EMBED_MODEL", "qwen3-embedding")

# --- Qwen3-Embedding instruction prefix ------------------------------------
# Qwen3-Embedding is instruction-aware: prefixing the QUERY (not the documents)
# with a task instruction reliably nudges recall up a few points. Set
# USE_QUERY_INSTRUCTION=0 to A/B test how much it actually buys you.
USE_QUERY_INSTRUCTION = os.getenv("USE_QUERY_INSTRUCTION", "1") == "1"
QUERY_INSTRUCTION = (
    "Instruct: Given a question, retrieve passages that answer the question\nQuery: "
)

# --- Retrieval / chunking knobs --------------------------------------------
TOP_K         = int(os.getenv("TOP_K", "4"))     # chunks fed to the model
CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE", "200"))    # words per chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "40"))  # words shared between neighbors
EMBED_BATCH   = int(os.getenv("EMBED_BATCH", "32"))    # texts per embedding request

# --- Corpus ----------------------------------------------------------------
# Primary source: a cleaned HuggingFace dataset of ~200 PG essays (title + text).
# Fallback: drop your own .txt files in data/essays/ and they'll be used instead.
HF_DATASET   = os.getenv("HF_DATASET", "sgoel9/paul_graham_essays")
LOCAL_ESSAYS = os.path.join(os.path.dirname(__file__), "data", "essays")
CACHE_DIR    = os.path.join(os.path.dirname(__file__), "data")
