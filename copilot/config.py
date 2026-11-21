"""Paths, model names and tunables. Override any model with an environment variable."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
FILINGS = RAW / "filings"
DB_PATH = PROCESSED / "warehouse.duckdb"
LOG_DIR = ROOT / "logs"

EMBED_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
# bge models want this prefix on queries (not on passages)
EMBED_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
RERANK_MODEL = os.environ.get("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
LLM_MODEL = os.environ.get("LLM_MODEL", "mlx-community/Qwen2.5-7B-Instruct-4bit")

CHUNK_WORDS = 220          # target chunk size for both chunkers
FIXED_OVERLAP_WORDS = 40   # overlap for the fixed-size baseline
SEMANTIC_BREAK_PCTL = 20   # split where adjacent-paragraph similarity is in the bottom 20%

RETRIEVE_K = 30            # candidates per retriever before fusion/rerank
CONTEXT_K = 8              # chunks passed to the LLM
RRF_K = 60                 # reciprocal rank fusion constant
MIN_RERANK_SCORE = -4.0    # below this the best chunk is treated as "no evidence"

TICKERS = {
    "WMT": ["walmart", "wal-mart", "sam's club", "sams club"],
    "TGT": ["target"],
    "COST": ["costco"],
    "HD": ["home depot"],
    "LOW": ["lowe's", "lowes", "lowe’s"],
    "BBY": ["best buy"],
}
