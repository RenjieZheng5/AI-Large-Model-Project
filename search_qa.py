import os
import json
from typing import List, Dict, Any

import faiss
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import CrossEncoder

from rag_config import (
    INDEX_DIR,
    EMBED_MODEL_PATH,
    TOP_K,
    MAX_EMBED_LEN,
    EMBED_DEVICE,
)
USE_RERANKER = os.environ.get("USE_RERANKER", "1") == "1"

RERANK_MODEL_PATH = os.environ.get(
    "RERANK_MODEL_PATH",
    "/autodl-fs/data/models/bge-reranker-v2-m3"
)

RERANK_TOP_N = int(os.environ.get("RERANK_TOP_N", "5"))

RERANK_BATCH_SIZE = int(os.environ.get("RERANK_BATCH_SIZE", "8"))

RERANK_DEVICE = os.environ.get("RERANK_DEVICE", "cuda")

def last_token_pool(last_hidden_states, attention_mask):
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]

    if left_padding:
        return last_hidden_states[:, -1]

    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]

    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device),
        sequence_lengths
    ]


def get_query_instruction(query):
    task = "Given a Chinese user question, retrieve relevant passages that answer the question"
    return f"Instruct: {task}\nQuery:{query}"


class Qwen3Embedder:
    def __init__(self, model_path, device="cuda", max_length=8192):
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.max_length = max_length

        print(f"[INFO] Loading embedding tokenizer: {model_path}", flush=True)
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
        ).to(self.device)

        self.model.eval()

    @torch.no_grad()
    def encode_query(self, query):
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
def load_items(path):
    items = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))

    return items
def faiss_search(index, items, embedder, query, top_k):
    q_emb = embedder.encode_query(query)
    scores, ids = index.search(q_emb, top_k)

    results = []

    for rank, idx in enumerate(ids[0], start=1):
        if idx < 0:
            continue

        item = items[int(idx)]
        score = float(scores[0][rank - 1])

        results.append({
            "rank": rank,
            "score": score,
            "faiss_rank": rank,
            "faiss_score": score,
            "id": item.get("id", int(idx)),
            "question": item.get("question", ""),
            "answer": item.get("answer", ""),
            "index_text": item.get("index_text", ""),
            "raw": item.get("raw", {}),
        })

    return results
class BGEReranker:
    def __init__(self, model_path, device="cuda", batch_size=8):
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"

        self.model_path = model_path
        self.device = device
        self.batch_size = batch_size

        print(f"[INFO] Loading BGE reranker: {model_path}", flush=True)
        print(f"[INFO] Reranker device: {self.device}", flush=True)

        self.model = CrossEncoder(
            model_path,
            device=self.device
        )

    def build_doc_text(self, item: Dict[str, Any]) -> str:
        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()
        index_text = str(item.get("index_text", "")).strip()
        raw = item.get("raw", {})

        category = ""
        source_title = ""
        source_url = ""

        if isinstance(raw, dict):
            category = str(raw.get("category", "")).strip()
            source_title = str(raw.get("source_title", "")).strip()
            source_url = str(raw.get("source_url", "")).strip()

        if question or answer:
            text = f"问题：{question}\n答案：{answer}"
        else:
            text = index_text

        if category:
            text += f"\n类别：{category}"

        if source_title:
            text += f"\n来源标题：{source_title}"

        if source_url:
            text += f"\n来源链接：{source_url}"

        return text

    def rerank(self, query, candidates, top_n=5):
        if not candidates:
            return []

        pairs = []

        for item in candidates:
            doc_text = self.build_doc_text(item)
            pairs.append([query, doc_text])

        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            convert_to_numpy=True
        )

        reranked = []

        for item, score in zip(candidates, scores):
            new_item = dict(item)
            new_item["rerank_score"] = float(score)
            reranked.append(new_item)

        reranked.sort(
            key=lambda x: x["rerank_score"],
            reverse=True
        )

        for new_rank, item in enumerate(reranked[:top_n], start=1):
            item["rank"] = new_rank

        return reranked[:top_n]
def shorten(text, max_len=300):
    text = str(text).replace("\n", " ").strip()
    if len(text) > max_len:
        return text[:max_len] + "..."
    return text


def print_faiss_results(results):
    print("\n" + "=" * 100)
    print("FAISS 初筛结果")
    print("=" * 100)

    for r in results:
        print("-" * 100)
        print(f"[FAISS Rank {r['rank']}] score={r['score']:.4f} id={r.get('id')}")
        print("问题：", shorten(r.get("question", ""), 220))
        print("答案：", shorten(r.get("answer", ""), 360))


def print_rerank_results(results):
    print("\n" + "=" * 100)
    print("BGE Reranker 重排结果")
    print("=" * 100)

    for r in results:
        print("-" * 100)
        print(
            f"[Rerank Rank {r['rank']}] "
            f"rerank_score={r.get('rerank_score', 0):.4f} | "
            f"faiss_rank={r.get('faiss_rank')} | "
            f"faiss_score={r.get('faiss_score', 0):.4f} | "
            f"id={r.get('id')}"
        )
        print("问题：", shorten(r.get("question", ""), 220))
        print("答案：", shorten(r.get("answer", ""), 360))


def build_context(results, max_chars=5000):
    """
    这个函数不调用 LLM，只是把最终 top_n 资料拼成可复制给 LLM 的 context。
    """
    parts = []

    for r in results:
        q = str(r.get("question", "")).strip()
        a = str(r.get("answer", "")).strip()

        block = f"""[资料{r['rank']}]
FAISS 排名：{r.get('faiss_rank')}
FAISS 分数：{r.get('faiss_score', 0):.4f}
Rerank 分数：{r.get('rerank_score', 0):.4f}
问题：{q}
答案：{a}
"""
        parts.append(block)

    context = "\n".join(parts)

    if len(context) > max_chars:
        context = context[:max_chars] + "\n\n[提示：后续资料因长度限制已截断]"

    return context
def main():
    index_path = os.path.join(INDEX_DIR, "index.faiss")
    items_path = os.path.join(INDEX_DIR, "items.jsonl")

    print("=" * 100)
    print("[Search QA with BGE Reranker]")
    print("INDEX_DIR        :", INDEX_DIR)
    print("INDEX_PATH       :", index_path)
    print("ITEMS_PATH       :", items_path)
    print("EMBED_MODEL_PATH :", EMBED_MODEL_PATH)
    print("EMBED_DEVICE     :", EMBED_DEVICE)
    print("MAX_EMBED_LEN    :", MAX_EMBED_LEN)
    print("TOP_K            :", TOP_K)
    print("USE_RERANKER     :", USE_RERANKER)
    print("RERANK_MODEL_PATH:", RERANK_MODEL_PATH)
    print("RERANK_TOP_N     :", RERANK_TOP_N)
    print("RERANK_DEVICE    :", RERANK_DEVICE)
    print("=" * 100)

    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index not found: {index_path}")

    if not os.path.exists(items_path):
        raise FileNotFoundError(f"Items file not found: {items_path}")

    print("[INFO] Loading FAISS index...", flush=True)
    index = faiss.read_index(index_path)

    print("[INFO] Loading indexed items...", flush=True)
    items = load_items(items_path)
    print(f"[INFO] Loaded items: {len(items)}", flush=True)

    embedder = Qwen3Embedder(
        EMBED_MODEL_PATH,
        device=EMBED_DEVICE,
        max_length=MAX_EMBED_LEN,
    )

    reranker = None
    if USE_RERANKER:
        reranker = BGEReranker(
            model_path=RERANK_MODEL_PATH,
            device=RERANK_DEVICE,
            batch_size=RERANK_BATCH_SIZE,
        )

    print("\n请输入问题，输入 exit / quit / q 退出。")

    while True:
        query = input("\n请输入问题：").strip()

        if query.lower() in ["exit", "quit", "q"]:
            print("已退出。")
            break

        if not query:
            continue

        # 1. FAISS 初筛
        faiss_results = faiss_search(
            index=index,
            items=items,
            embedder=embedder,
            query=query,
            top_k=TOP_K,
        )

        if not faiss_results:
            print("没有检索到结果。")
            continue

        print_faiss_results(faiss_results[:min(10, len(faiss_results))])

        # 2. BGE Reranker 重排
        if reranker is not None:
            final_results = reranker.rerank(
                query=query,
                candidates=faiss_results,
                top_n=RERANK_TOP_N,
            )

            print_rerank_results(final_results)
        else:
            final_results = faiss_results[:RERANK_TOP_N]

        # 3. 输出可直接给 LLM 的 context
        context = build_context(final_results)

        print("\n" + "=" * 100)
        print("可直接给大模型的 Context：")
        print("=" * 100)
        print(context)


if __name__ == "__main__":
    main()