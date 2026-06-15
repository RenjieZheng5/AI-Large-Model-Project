import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"
CHART_DIR = RESULTS_DIR / "charts"

DATA_PATH = os.environ.get(
    "DATA_PATH",
    str(DATA_DIR / "sustech_qa_pairs.jsonl")
)

INDEX_DIR = os.environ.get(
    "INDEX_DIR",
    str(DATA_DIR / "rag_index_sustech_qwen3_embedding")
)

EMBED_MODEL_PATH = os.environ.get(
    "EMBED_MODEL_PATH",
    "/path/to/Qwen3-Embedding-0.6B"
)

LLM_MODEL_PATH = os.environ.get(
    "LLM_MODEL_PATH",
    "/path/to/Qwen3-32B"
)

VLLM_URL = os.environ.get(
    "VLLM_URL",
    "http://127.0.0.1:8000/v1/chat/completions"
)

VLLM_MODEL = os.environ.get(
    "VLLM_MODEL",
    "qwen3-32b"
)

TOP_K = int(os.environ.get("TOP_K", "30"))

# 是否启用 reranker
USE_RERANKER = os.environ.get("USE_RERANKER", "1") == "1"

# reranker 模型路径
RERANK_MODEL_PATH = os.environ.get(
    "RERANK_MODEL_PATH",
    "/path/to/bge-reranker-v2-m3"
)

# reranker 重排后保留多少条进入 LLM
RERANK_TOP_N = int(os.environ.get("RERANK_TOP_N", "5"))

# reranker batch size
RERANK_BATCH_SIZE = int(os.environ.get("RERANK_BATCH_SIZE", "8"))

EMBED_DEVICE = os.environ.get("EMBED_DEVICE", "cuda")

MAX_EMBED_LEN = int(os.environ.get("MAX_EMBED_LEN", "8192"))
