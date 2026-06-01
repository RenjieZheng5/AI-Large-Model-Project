"""
Batch evaluation template for the SUSTech RAG project.

What this script does:
1) Loads the test questions from the Excel file.
2) Runs each question in Direct Q&A mode (no retrieval).
3) Runs each question in RAG mode (call your existing project function here).
4) Saves all results to a new Excel file.

You need to edit ONLY the two functions:
- direct_answer(question)
- rag_answer(question)

For direct_answer, the easiest path is to call your running vLLM server.
For rag_answer, import the answer function from your project code (the same logic used by Gradio).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import requests


INPUT_XLSX = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "sustech_rag_test_questions.xlsx",
)
OUTPUT_XLSX = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "sustech_rag_batch_results.xlsx",
)
SHEET_NAME = "TestSet"  # 测试问题在 TestSet sheet 中

VLLM_API_URL = os.environ.get(
    "VLLM_URL",
    "http://127.0.0.1:8000/v1/chat/completions",
)
MODEL_NAME = os.environ.get("VLLM_MODEL", "qwen3-32b")


def direct_answer(question: str) -> str:
    """Direct Q&A: send only the question to the LLM, without retrieval."""
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question},
        ],
        "temperature": 0.0,
    }
    resp = requests.post(VLLM_API_URL, json=payload, timeout=300)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


def rag_answer(question: str) -> str:
    """
    RAG answer: 调用项目完整的检索+重排+生成流程。
    首次调用会自动初始化 embedder / retriever / reranker（需一定时间）。
    """
    # 延迟导入 + 模块级缓存，避免重复初始化
    global _rag_embedder, _rag_retriever, _rag_reranker

    if "_rag_embedder" not in globals():
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

        from rag_config import (
            INDEX_DIR, EMBED_MODEL_PATH, TOP_K,
            EMBED_DEVICE, USE_RERANKER, RERANK_MODEL_PATH,
            RERANK_TOP_N, RERANK_BATCH_SIZE,
        )
        from rag_chat import Qwen3Embedder, FaissRetriever, build_context, call_vllm
        from reranker import BGEReranker

        print("[INFO] 初始化 RAG 组件 (首次调用)...")
        _rag_embedder = Qwen3Embedder(
            model_path=EMBED_MODEL_PATH,
            device=EMBED_DEVICE,
            max_length=8192,
        )
        _rag_retriever = FaissRetriever(
            index_dir=INDEX_DIR,
            embedder=_rag_embedder,
        )
        _rag_reranker = None
        if USE_RERANKER:
            _rag_reranker = BGEReranker(
                model_path=RERANK_MODEL_PATH,
                device=os.environ.get("RERANK_DEVICE", "cpu"),
                batch_size=RERANK_BATCH_SIZE,
            )
        print("[INFO] RAG 组件初始化完成。")

    # 1. FAISS 检索
    faiss_results = _rag_retriever.search(question, top_k=30)
    if not faiss_results:
        return "没有检索到相关资料。"

    # 2. Reranker 重排
    if _rag_reranker is not None:
        final_results = _rag_reranker.rerank(
            query=question,
            candidates=faiss_results,
            top_n=5,
        )
    else:
        final_results = faiss_results[:5]

    # 3. 构建上下文 + 调用 LLM
    context = build_context(final_results)
    return call_vllm(question, context)


def main() -> None:
    df = pd.read_excel(INPUT_XLSX, sheet_name=SHEET_NAME)
    required_cols = {"Test_ID", "Test_Question", "Gold_Answer"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in input file: {missing}")

    direct_outputs = []
    rag_outputs = []

    for i, row in df.iterrows():
        q = str(row["Test_Question"]).strip()
        print(f"[{i+1}/{len(df)}] {q}")

        try:
            direct = direct_answer(q)
        except Exception as e:
            direct = f"[ERROR] {type(e).__name__}: {e}"

        try:
            rag = rag_answer(q)
        except Exception as e:
            rag = f"[NOT_IMPLEMENTED_OR_ERROR] {type(e).__name__}: {e}"

        direct_outputs.append(direct)
        rag_outputs.append(rag)

        # small pause to avoid hammering the server too hard
        time.sleep(0.1)

    df["Direct_Answer"] = direct_outputs
    df["RAG_Answer"] = rag_outputs
    df.to_excel(OUTPUT_XLSX, index=False)
    print(f"Saved to: {OUTPUT_XLSX}")


if __name__ == "__main__":
    main()
