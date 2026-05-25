import os
import json
import textwrap
from typing import List, Dict, Any

import faiss
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

from rag_config import (
    INDEX_DIR,
    EMBED_MODEL_PATH,
    TOP_K,
    MAX_EMBED_LEN,
    EMBED_DEVICE,
    USE_RERANKER,
    RERANK_MODEL_PATH,
    RERANK_TOP_N,
    RERANK_BATCH_SIZE,
)

from reranker import BGEReranker


LLM_MODEL_PATH = os.environ.get(
    "LLM_MODEL_PATH",
    "/autodl-fs/data/models/Qwen3-32B"
)

# 单独给 reranker 一个设备配置，避免它和 embedding 强绑定。
# 显存不够时可以 export RERANK_DEVICE=cpu
RERANK_DEVICE = os.environ.get("RERANK_DEVICE", "cuda")

# 生成参数
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "512"))
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.3"))
LLM_TOP_P = float(os.environ.get("LLM_TOP_P", "0.8"))


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


def get_query_instruction(query: str) -> str:
    task = "Given a Chinese user question, retrieve relevant passages that answer the question"
    return f"Instruct: {task}\nQuery:{query}"


class Qwen3Embedder:
    def __init__(self, model_path: str, device: str = "cuda", max_length: int = 8192):
        self.device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
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

        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"FAISS index not found: {self.index_path}")

        if not os.path.exists(self.items_path):
            raise FileNotFoundError(f"Items file not found: {self.items_path}")

        print(f"[INFO] Loading FAISS index: {self.index_path}", flush=True)
        self.index = faiss.read_index(self.index_path)

        print(f"[INFO] Loading items: {self.items_path}", flush=True)
        self.items = load_jsonl(self.items_path)

        self.embedder = embedder

        print(f"[INFO] Loaded FAISS items: {len(self.items)}", flush=True)

    def search(self, query: str, top_k: int = 30) -> List[Dict[str, Any]]:
        q_emb = self.embedder.encode_query(query)
        scores, ids = self.index.search(q_emb, top_k)

        results = []

        for rank, idx in enumerate(ids[0], start=1):
            if idx < 0:
                continue

            item = self.items[int(idx)]
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


def build_context(results: List[Dict[str, Any]], max_chars: int = 6000) -> str:
    parts = []

    for r in results:
        q = str(r.get("question", "")).strip()
        a = str(r.get("answer", "")).strip()
        rank = r.get("rank", "?")

        faiss_score = r.get("faiss_score", r.get("score", 0.0))
        rerank_score = r.get("rerank_score", None)

        raw = r.get("raw", {})
        source_url = ""
        source_title = ""
        category = ""

        if isinstance(raw, dict):
            source_url = str(raw.get("source_url", "")).strip()
            source_title = str(raw.get("source_title", "")).strip()
            category = str(raw.get("category", "")).strip()

        if q or a:
            text = f"问题：{q}\n答案：{a}"
        else:
            text = str(r.get("index_text", "")).strip()

        extra = []
        if category:
            extra.append(f"类别：{category}")
        if source_title:
            extra.append(f"来源标题：{source_title}")
        if source_url:
            extra.append(f"来源链接：{source_url}")

        if extra:
            text += "\n" + "\n".join(extra)

        if rerank_score is not None:
            header = (
                f"[资料{rank} | "
                f"FAISS排名 {r.get('faiss_rank', '?')} | "
                f"FAISS分数 {faiss_score:.4f} | "
                f"Rerank分数 {rerank_score:.4f}]"
            )
        else:
            header = f"[资料{rank} | 相似度 {faiss_score:.4f}]"

        parts.append(f"{header}\n{text}")

    context = "\n\n".join(parts)

    if len(context) > max_chars:
        context = context[:max_chars] + "\n\n[提示：后续参考资料因长度限制已截断]"

    return context


def build_messages(question: str, context: str):
    system_prompt = (
        "你是一个严谨的中文问答助手。"
        "你必须优先根据给定参考资料回答问题。"
        "如果参考资料中没有足够信息，请回答“根据现有资料无法确定”。"
        "不要编造参考资料中不存在的内容。"
        "回答要清晰、简洁、准确。"
    )

    user_prompt = f"""请根据下面参考资料回答问题。

参考资料：
{context}

用户问题：
{question}

回答要求：
1. 优先依据参考资料。
2. 如果资料不足，回答“根据现有资料无法确定”。
3. 不要编造学校政策、时间、地点、数字等信息。
4. 用中文回答，简洁准确。"""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


class QwenGenerator:
    def __init__(self, model_path: str):
        print(f"[INFO] Loading LLM tokenizer: {model_path}", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
        )

        print(f"[INFO] Loading LLM model: {model_path}", flush=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map={"": 0},
            trust_remote_code=True,
            local_files_only=True,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
        )

        self.model.eval()

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @torch.no_grad()
    def generate(self, question: str, context: str):
        messages = build_messages(question, context)

        try:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        inputs = self.tokenizer(
            [text],
            return_tensors="pt",
            padding=True,
        ).to("cuda")

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=LLM_TEMPERATURE,
            top_p=LLM_TOP_P,
            top_k=20,
            repetition_penalty=1.05,
            remove_invalid_values=True,
            renormalize_logits=True,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        answer = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        return answer.strip()


def shorten(text: str, max_len: int = 160) -> str:
    text = str(text).replace("\n", " ").strip()
    if len(text) > max_len:
        return text[:max_len] + "..."
    return text


def print_sources(results: List[Dict[str, Any]], title: str):
    print(f"\n{title}：")

    for r in results:
        q = shorten(r.get("question", ""), 120)
        a = shorten(r.get("answer", ""), 180)

        faiss_score = r.get("faiss_score", r.get("score", 0.0))
        rerank_score = r.get("rerank_score", None)

        print("-" * 80)

        if rerank_score is not None:
            print(
                f"[{r['rank']}] "
                f"rerank_score={rerank_score:.4f}, "
                f"faiss_rank={r.get('faiss_rank')}, "
                f"faiss_score={faiss_score:.4f}, "
                f"id={r.get('id')}"
            )
        else:
            print(f"[{r['rank']}] score={faiss_score:.4f}, id={r.get('id')}")

        print("Q:", q)
        print("A:", a)


def main():
    print("=" * 80)
    print("[RAG Chat - Transformers Backend + BGE Reranker]")
    print("INDEX_DIR         :", INDEX_DIR)
    print("EMBED_MODEL_PATH  :", EMBED_MODEL_PATH)
    print("LLM_MODEL_PATH    :", LLM_MODEL_PATH)
    print("EMBED_DEVICE      :", EMBED_DEVICE)
    print("TOP_K             :", TOP_K)
    print("USE_RERANKER      :", USE_RERANKER)
    print("RERANK_MODEL_PATH :", RERANK_MODEL_PATH)
    print("RERANK_TOP_N      :", RERANK_TOP_N)
    print("RERANK_DEVICE     :", RERANK_DEVICE)
    print("=" * 80)

    embedder = Qwen3Embedder(
        EMBED_MODEL_PATH,
        device=EMBED_DEVICE,
        max_length=MAX_EMBED_LEN,
    )

    retriever = FaissRetriever(
        INDEX_DIR,
        embedder,
    )

    reranker = None

    if USE_RERANKER:
        reranker = BGEReranker(
            model_path=RERANK_MODEL_PATH,
            device=RERANK_DEVICE,
            batch_size=RERANK_BATCH_SIZE,
        )

    generator = QwenGenerator(LLM_MODEL_PATH)

    print("\n输入问题开始问答。输入 exit / quit / q 退出。")

    while True:
        question = input("\n请输入问题：").strip()

        if question.lower() in {"exit", "quit", "q"}:
            print("已退出。")
            break

        if not question:
            continue

        # 1. FAISS 初筛
        faiss_results = retriever.search(question, top_k=TOP_K)

        if not faiss_results:
            print("没有检索到相关资料。")
            continue

        print_sources(
            faiss_results[:min(10, len(faiss_results))],
            title=f"FAISS 初筛结果 Top {min(10, len(faiss_results))}"
        )

        # 2. BGE Reranker 重排
        if reranker is not None:
            final_results = reranker.rerank(
                query=question,
                candidates=faiss_results,
                top_n=RERANK_TOP_N,
            )

            print_sources(
                final_results,
                title=f"BGE Reranker 重排结果 Top {len(final_results)}"
            )
        else:
            final_results = faiss_results[:RERANK_TOP_N]

        # 3. 构造 context
        context = build_context(final_results)

        print("\n正在生成回答...\n")

        # 4. 本地 Transformers LLM 生成
        answer = generator.generate(question, context)

        print("=" * 80)
        print("最终回答：")
        print(textwrap.fill(answer, width=100))
        print("=" * 80)


if __name__ == "__main__":
    main()