import json
import os
import re
import textwrap
from typing import Any, Dict, List

import faiss
import requests
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from rag_config import (
    EMBED_DEVICE,
    EMBED_MODEL_PATH,
    INDEX_DIR,
    MAX_EMBED_LEN,
    RERANK_BATCH_SIZE,
    RERANK_MODEL_PATH,
    RERANK_TOP_N,
    TOP_K,
    USE_RERANKER,
    VLLM_MODEL,
    VLLM_URL,
)
from reranker import BGEReranker


def strip_thinking(text: str) -> str:
    """Remove Qwen thinking traces if the backend ignores enable_thinking=False."""
    text = text or ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


def last_token_pool(last_hidden_states, attention_mask):
    """Pooling method recommended by Qwen3-Embedding."""
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]

    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device),
        sequence_lengths,
    ]


def get_query_instruction(query: str) -> str:
    task = "Given a Chinese user question, retrieve relevant passages that answer the question"
    return f"Instruct: {task}\nQuery:{query}"


class Qwen3Embedder:
    def __init__(self, model_path: str, device: str = "cuda", max_length: int = 8192):
        self.device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
        self.model_path = model_path
        self.max_length = max_length

        print(f"[INFO] Loading embedding tokenizer from: {model_path}", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            padding_side="left",
            trust_remote_code=True,
            local_files_only=True,
        )

        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        print(f"[INFO] Loading embedding model on {self.device}, dtype={dtype}", flush=True)
        self.model = AutoModel.from_pretrained(
            model_path,
            dtype=dtype,
            trust_remote_code=True,
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode_query(self, query: str):
        query_text = get_query_instruction(query)
        batch = self.tokenizer(
            [query_text],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model(**batch)
        emb = last_token_pool(outputs.last_hidden_state, batch["attention_mask"])
        emb = F.normalize(emb, p=2, dim=1)
        return emb.float().cpu().numpy().astype("float32")


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


class FaissRetriever:
    def __init__(self, index_dir: str, embedder: Qwen3Embedder):
        self.index_path = os.path.join(index_dir, "index.faiss")
        self.items_path = os.path.join(index_dir, "items.jsonl")
        self.meta_path = os.path.join(index_dir, "meta.json")

        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"FAISS index not found: {self.index_path}")
        if not os.path.exists(self.items_path):
            raise FileNotFoundError(f"Items file not found: {self.items_path}")

        print(f"[INFO] Loading FAISS index: {self.index_path}", flush=True)
        self.index = faiss.read_index(self.index_path)
        print(f"[INFO] Loading items: {self.items_path}", flush=True)
        self.items = load_jsonl(self.items_path)
        self.embedder = embedder
        print(f"[INFO] Loaded {len(self.items)} indexed QA items.", flush=True)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        q_emb = self.embedder.encode_query(query)
        scores, ids = self.index.search(q_emb, top_k)

        results = []
        for rank, idx in enumerate(ids[0], start=1):
            if idx < 0:
                continue

            item = self.items[int(idx)]
            score = float(scores[0][rank - 1])
            results.append(
                {
                    "rank": rank,
                    "score": score,
                    "id": item.get("id", int(idx)),
                    "question": item.get("question", ""),
                    "answer": item.get("answer", ""),
                    "index_text": item.get("index_text", ""),
                    "raw": item.get("raw", {}),
                }
            )

        return results


def build_context(results: List[Dict[str, Any]], max_chars: int = 6000) -> str:
    """Build the retrieved QA context passed to the generator."""
    parts = []
    for r in results:
        q = str(r.get("question", "")).strip()
        a = str(r.get("answer", "")).strip()
        rank = r.get("rank", "?")
        faiss_score = r.get("faiss_score", r.get("score", 0.0))
        rerank_score = r.get("rerank_score", None)

        text = str(r.get("index_text", "")).strip() if not (q or a) else f"问题：{q}\n答案：{a}"
        if rerank_score is not None:
            header = f"[资料{rank} | FAISS相似度 {faiss_score:.4f} | Rerank分数 {rerank_score:.4f}]"
        else:
            header = f"[资料{rank} | 相似度 {faiss_score:.4f}]"
        parts.append(f"{header}\n{text}")

    context = "\n\n".join(parts)
    if len(context) > max_chars:
        context = context[:max_chars] + "\n\n[提示：后续参考资料因长度限制已截断]"
    return context


def call_vllm(question: str, context: str) -> str:
    """Call the OpenAI-compatible vLLM API."""
    system_prompt = (
        "你是一个严谨的中文校园知识库问答助手。"
        "必须优先依据给定参考资料回答问题。"
        "如果参考资料不足以回答，请只回答“根据现有资料无法确定”。"
        "不要编造参考资料中不存在的政策、时间、地点、数字或流程。"
        "回答要清晰、简洁、准确。"
    )
    user_prompt = f"""请根据下面的参考资料回答用户问题。

参考资料：
{context}

用户问题：{question}

回答要求：
1. 优先依据参考资料回答。
2. 如果资料不足，请直接说“根据现有资料无法确定”。
3. 不要编造学校政策、时间、地点、数字、流程。
4. 用中文回答。"""

    payload = {
        "model": VLLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
        "top_p": 0.8,
        "max_tokens": 512,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    try:
        resp = requests.post(VLLM_URL, json=payload, timeout=300)
    except requests.exceptions.ConnectionError as e:
        raise RuntimeError(
            f"无法连接 vLLM 服务：{VLLM_URL}\n"
            "请确认你已经在另一个终端或 tmux 中启动 vllm serve。"
        ) from e

    if resp.status_code >= 400 and "chat_template_kwargs" in resp.text:
        payload.pop("chat_template_kwargs", None)
        resp = requests.post(VLLM_URL, json=payload, timeout=300)

    if resp.status_code != 200:
        raise RuntimeError(f"vLLM 请求失败，状态码：{resp.status_code}\n返回内容：\n{resp.text}")

    data = resp.json()
    try:
        return strip_thinking(data["choices"][0]["message"]["content"])
    except Exception as e:
        raise RuntimeError(f"vLLM 返回格式异常：\n{json.dumps(data, ensure_ascii=False, indent=2)}") from e


def print_retrieval_results(results: List[Dict[str, Any]], title: str = "检索到的参考资料"):
    print(f"\n{title}:")
    for r in results:
        q = str(r.get("question", "")).replace("\n", " ").strip()
        a = str(r.get("answer", "")).replace("\n", " ").strip()
        q = q[:100] + "..." if len(q) > 100 else q
        a = a[:160] + "..." if len(a) > 160 else a

        faiss_score = r.get("faiss_score", r.get("score", 0.0))
        rerank_score = r.get("rerank_score", None)

        print("-" * 80)
        if rerank_score is not None:
            print(
                f"[{r['rank']}] faiss_rank={r.get('faiss_rank', '?')}, "
                f"faiss_score={faiss_score:.4f}, rerank_score={rerank_score:.4f}, id={r.get('id')}"
            )
        else:
            print(f"[{r['rank']}] score={faiss_score:.4f}, id={r.get('id')}")
        print(f"Q: {q}")
        print(f"A: {a}")


def main():
    print("=" * 80)
    print("[RAG Chat]")
    print(f"INDEX_DIR        : {INDEX_DIR}")
    print(f"EMBED_MODEL_PATH : {EMBED_MODEL_PATH}")
    print(f"EMBED_DEVICE     : {EMBED_DEVICE}")
    print(f"VLLM_URL         : {VLLM_URL}")
    print(f"VLLM_MODEL       : {VLLM_MODEL}")
    print(f"TOP_K            : {TOP_K}")
    print(f"USE_RERANKER     : {USE_RERANKER}")
    print(f"RERANK_MODEL_PATH: {RERANK_MODEL_PATH}")
    print(f"RERANK_TOP_N     : {RERANK_TOP_N}")
    print("=" * 80)

    embedder = Qwen3Embedder(EMBED_MODEL_PATH, device=EMBED_DEVICE, max_length=MAX_EMBED_LEN)
    retriever = FaissRetriever(index_dir=INDEX_DIR, embedder=embedder)

    reranker = None
    if USE_RERANKER:
        reranker = BGEReranker(
            model_path=RERANK_MODEL_PATH,
            device=os.environ.get("RERANK_DEVICE", EMBED_DEVICE),
            batch_size=RERANK_BATCH_SIZE,
        )

    print("\n输入问题开始 RAG 问答。输入 exit / quit / q 退出。")
    while True:
        question = input("\n请输入问题：").strip()
        if question.lower() in {"exit", "quit", "q"}:
            print("已退出。")
            break
        if not question:
            continue

        results = retriever.search(question, top_k=TOP_K)
        if not results:
            print("没有检索到相关资料。")
            continue

        print_retrieval_results(results[:10], title=f"FAISS 初筛结果 Top {min(10, len(results))}")
        final_results = reranker.rerank(question, results, top_n=RERANK_TOP_N) if reranker else results[:RERANK_TOP_N]
        if reranker:
            print_retrieval_results(final_results, title=f"BGE Reranker 重排结果 Top {len(final_results)}")

        print("\n正在调用 vLLM 生成回答...\n")
        answer = call_vllm(question, build_context(final_results))
        print("=" * 80)
        print("最终回答：")
        print(textwrap.fill(answer.strip(), width=100))
        print("=" * 80)


if __name__ == "__main__":
    main()
