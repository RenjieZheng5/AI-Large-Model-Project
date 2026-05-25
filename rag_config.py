import os

DATA_PATH = os.environ.get(
    "DATA_PATH",
    "/root/AI-Large-Model-Project/sustech_qa_pairs.jsonl"
)

INDEX_DIR = os.environ.get(
    "INDEX_DIR",
    "/root/AI-Large-Model-Project/rag_index_sustech_qwen3_embedding"
)

EMBED_MODEL_PATH = os.environ.get(
    "EMBED_MODEL_PATH",
    "/root/autodl-tmp/models/Qwen/Qwen3-Embedding-0___6B"
)

LLM_MODEL_PATH = os.environ.get(
    "LLM_MODEL_PATH",
    "/autodl-fs/data/models/Qwen3-32B"
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
    "/autodl-fs/data/models/bge-reranker-v2-m3"
)

# reranker 重排后保留多少条进入 LLM
RERANK_TOP_N = int(os.environ.get("RERANK_TOP_N", "5"))

# reranker batch size
RERANK_BATCH_SIZE = int(os.environ.get("RERANK_BATCH_SIZE", "8"))

EMBED_DEVICE = os.environ.get("EMBED_DEVICE", "cuda")

MAX_EMBED_LEN = int(os.environ.get("MAX_EMBED_LEN", "8192"))