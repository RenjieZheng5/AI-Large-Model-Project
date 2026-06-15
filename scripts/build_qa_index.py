import os
import json
import sys
from pathlib import Path

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from rag_config import DATA_PATH, INDEX_DIR, EMBED_MODEL_PATH, MAX_EMBED_LEN, EMBED_DEVICE


def pick_first(record, keys, default=""):
    for k in keys:
        if k in record and record[k] is not None:
            value = str(record[k]).strip()
            if value:
                return value
    return default


def load_qa_jsonl(path):
    items = []

    with open(path, "r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f"[WARN] line {line_id} is not valid json, skipped")
                continue

            question = pick_first(
                record,
                ["question", "query", "q", "instruction", "prompt", "input", "title"]
            )

            answer = pick_first(
                record,
                ["answer", "response", "a", "output", "completion", "content"]
            )

            # 如果字段名不标准，就把整个 json 当成文本兜底
            if not question and not answer:
                text = json.dumps(record, ensure_ascii=False)
                question = text[:300]
                answer = text

            index_text = f"问题：{question}\n答案：{answer}".strip()

            items.append({
                "id": line_id,
                "question": question,
                "answer": answer,
                "index_text": index_text,
                "raw": record
            })

    return items


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


class Qwen3Embedder:
    def __init__(self, model_path, device="cuda", max_length=8192):
        self.model_path = model_path
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.max_length = max_length

        print(f"[INFO] loading embedding tokenizer from: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            padding_side="left",
            trust_remote_code=True,
            local_files_only=True,
        )

        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

        print(f"[INFO] loading embedding model on {self.device}, dtype={dtype}")
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
    def encode_documents(self, texts, batch_size=8):
        all_embeddings = []

        for start in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
            batch_texts = texts[start:start + batch_size]

            batch = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            outputs = self.model(**batch)
            emb = last_token_pool(outputs.last_hidden_state, batch["attention_mask"])
            emb = F.normalize(emb, p=2, dim=1)

            all_embeddings.append(emb.float().cpu().numpy())

        return np.vstack(all_embeddings).astype("float32")


def main():
    print("[INFO] DATA_PATH:", DATA_PATH)
    print("[INFO] INDEX_DIR:", INDEX_DIR)
    print("[INFO] EMBED_MODEL_PATH:", EMBED_MODEL_PATH)

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(DATA_PATH)

    if not os.path.exists(EMBED_MODEL_PATH):
        raise FileNotFoundError(EMBED_MODEL_PATH)

    os.makedirs(INDEX_DIR, exist_ok=True)

    items = load_qa_jsonl(DATA_PATH)
    print(f"[INFO] loaded QA items: {len(items)}")

    if not items:
        raise RuntimeError("No QA items loaded. Please check jsonl format.")

    texts = [x["index_text"] for x in items]

    embedder = Qwen3Embedder(
        EMBED_MODEL_PATH,
        device=EMBED_DEVICE,
        max_length=MAX_EMBED_LEN,
    )

    embeddings = embedder.encode_documents(texts, batch_size=8)

    dim = embeddings.shape[1]
    print("[INFO] embedding dim:", dim)

    # embeddings 已经 normalize，因此 IndexFlatIP 等价于 cosine similarity
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    index_path = os.path.join(INDEX_DIR, "index.faiss")
    items_path = os.path.join(INDEX_DIR, "items.jsonl")
    meta_path = os.path.join(INDEX_DIR, "meta.json")

    faiss.write_index(index, index_path)

    with open(items_path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    meta = {
        "data_path": DATA_PATH,
        "embedding_model": EMBED_MODEL_PATH,
        "num_items": len(items),
        "dimension": dim,
        "index_type": "faiss.IndexFlatIP",
        "note": "Embeddings are L2-normalized, inner product equals cosine similarity."
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("[OK] saved index:", index_path)
    print("[OK] saved items:", items_path)
    print("[OK] saved meta:", meta_path)


if __name__ == "__main__":
    main()
