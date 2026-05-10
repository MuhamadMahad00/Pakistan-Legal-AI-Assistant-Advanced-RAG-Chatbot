# ============================================================
#  config.py — All project settings in one place
# ============================================================

import os
from dotenv import load_dotenv

load_dotenv()

# ── Groq LLM ────────────────────────────────────────────────
GROQ_API_KEY   = os.getenv("GROQ_API_KEY")
GROQ_MODEL     = "llama-3.1-8b-instant" # High-limit model to avoid rate limiting

# ── Local Embedding Model (runs on your machine, no API key) ─
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ── Reranker Model (Bonus Feature) ──────────────────────────
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ── Documents & Vector Store ────────────────────────────────
DOCS_FOLDER    = "documents"               # put your PDFs in subfolders here
VECTORSTORE_DIR = "vectorstore"            # FAISS indices saved here
BM25_DIR       = "bm25_indices"            # BM25 indices saved here

# ── Chunking Strategy ───────────────────────────────────────
CHUNK_SIZE     = 1000                      # characters per chunk
CHUNK_OVERLAP  = 200                       # overlap between chunks

# ── Retrieval ───────────────────────────────────────────────
RETRIEVER_K    = 10                        # chunks to retrieve initially (hybrid)
TOP_K_RESULTS  = 3                         # final top-3 chunks after reranking

# ── LLM Generation ──────────────────────────────────────────
MAX_TOKENS     = 1024
TEMPERATURE    = 0.0                       # 0 = no hallucination
