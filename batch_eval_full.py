"""
SUSTech RAG 批量评测脚本 —— System Effect Comparison
=====================================================
功能：
1. 对测试集中的每个问题运行 Direct Q&A（无检索）和 RAG（带检索）
2. 保存 Direct/RAG 答案、标准答案和 RAG 检索结果，供人工评估
3. 生成不含 LLM 自动评分的 Excel 和 JSON 摘要

用法：
    conda activate rag_qwen_b128
    export TOP_K=30 USE_RERANKER=1 RERANK_TOP_N=5
    export EMBED_DEVICE=cpu RERANK_DEVICE=cpu
    python batch_eval_full.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import pandas as pd
import requests

# 把项目目录加入 sys.path，方便导入项目模块
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from rag_config import (
    INDEX_DIR,
    EMBED_MODEL_PATH,
    TOP_K,
    VLLM_URL,
    VLLM_MODEL,
    EMBED_DEVICE,
    USE_RERANKER,
    RERANK_MODEL_PATH,
    RERANK_TOP_N,
    RERANK_BATCH_SIZE,
)
from rag_chat import Qwen3Embedder, FaissRetriever, build_context, call_vllm
from reranker import BGEReranker


# ═══════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════

INPUT_XLSX = os.path.join(PROJECT_DIR, "sustech_rag_test_questions.xlsx")
OUTPUT_XLSX = os.path.join(PROJECT_DIR, "sustech_rag_batch_results_full.xlsx")
OUTPUT_JSON = os.path.join(PROJECT_DIR, "sustech_rag_batch_results_full.json")
SHEET_NAME = "TestSet"

# Direct Q&A 用，尽量确定；RAG 生成参数由 rag_chat.call_vllm 统一控制。
LLM_TEMP = 0.0
LLM_MAX_TOKENS = 512

print("=" * 60)
print("[INFO] 正在初始化 RAG 系统组件...")
print(f"  VLLM URL: {VLLM_URL}")
print(f"  Embed model: {EMBED_MODEL_PATH}")
print(f"  Embed device: {EMBED_DEVICE}")
print(f"  Reranker: {'ON' if USE_RERANKER else 'OFF'}")
print("=" * 60)

# 初始化 Embedder
embedder = Qwen3Embedder(
    model_path=EMBED_MODEL_PATH,
    device=EMBED_DEVICE,
    max_length=8192,
)

# 初始化 Retriever
retriever = FaissRetriever(
    index_dir=INDEX_DIR,
    embedder=embedder,
)

# 初始化 Reranker
reranker = None
if USE_RERANKER:
    RERANK_DEVICE = os.environ.get("RERANK_DEVICE", "cpu")
    reranker = BGEReranker(
        model_path=RERANK_MODEL_PATH,
        device=RERANK_DEVICE,
        batch_size=RERANK_BATCH_SIZE,
    )

print("[INFO] RAG 系统初始化完成。\n")


# ═══════════════════════════════════════════════════════════
# 核心函数
# ═══════════════════════════════════════════════════════════

def direct_answer(question: str) -> str:
    """
    Direct Q&A：不提供任何参考资料，直接问 LLM。
    使用确定性 temperature=0，避免随机性干扰对比。
    """
    system_prompt = "你是一个有帮助的AI助手。请简洁准确地回答问题。如果不知道，就说不知道。"

    payload = {
        "model": VLLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        "temperature": LLM_TEMP,
        "max_tokens": LLM_MAX_TOKENS,
    }

    resp = requests.post(VLLM_URL, json=payload, timeout=300)
    if resp.status_code >= 400 and "chat_template_kwargs" in resp.text:
        payload.pop("chat_template_kwargs", None)
        resp = requests.post(VLLM_URL, json=payload, timeout=300)

    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


def rag_answer(question: str) -> Tuple[str, List[Dict[str, Any]]]:
    """
    RAG 问答：完整检索 + 重排 + 生成流程。
    返回 (answer_text, retrieved_docs) 以便人工分析检索质量。
    """
    # 1. FAISS 检索
    faiss_results = retriever.search(question, top_k=TOP_K)

    if not faiss_results:
        return "没有检索到相关资料。", []

    # 2. BGE Reranker 重排
    if reranker is not None:
        final_results = reranker.rerank(
            query=question,
            candidates=faiss_results,
            top_n=RERANK_TOP_N,
        )
    else:
        final_results = faiss_results[:RERANK_TOP_N]

    # 3. 构建 context 并调用 vLLM
    context = build_context(final_results)
    answer = call_vllm(question, context)

    return answer, final_results


def format_retrieved_questions(retrieved: List[Dict[str, Any]]) -> str:
    return " | ".join([r.get("question", "")[:60] for r in retrieved])


def format_retrieved_details(retrieved: List[Dict[str, Any]]) -> str:
    details = []

    for r in retrieved:
        raw = r.get("raw", {}) if isinstance(r.get("raw"), dict) else {}
        details.append({
            "rank": r.get("rank"),
            "id": r.get("id"),
            "question": r.get("question", ""),
            "answer": r.get("answer", ""),
            "faiss_rank": r.get("faiss_rank"),
            "faiss_score": r.get("faiss_score", r.get("score")),
            "rerank_score": r.get("rerank_score"),
            "category": raw.get("category", ""),
            "source_title": raw.get("source_title", ""),
            "source_url": raw.get("source_url", ""),
        })

    return json.dumps(details, ensure_ascii=False)


def json_value(value: Any) -> Any:
    """把 pandas/numpy 标量转成 json.dump 可以直接处理的基础类型。"""
    if pd.isna(value):
        return ""

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    return value


# ═══════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════

def main() -> None:
    # 1. 加载测试数据
    df = pd.read_excel(INPUT_XLSX, sheet_name=SHEET_NAME)
    required_cols = {"Test_ID", "Test_Question"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"TestSet 缺少列: {missing}\n实际列: {list(df.columns)}")

    has_gold_answer = "Gold_Answer" in df.columns
    n = len(df)
    print(f"[INFO] 加载了 {n} 条测试问题。")
    print("[INFO] 已取消 LLM 自动评分；本脚本只生成答案和检索记录，供人工评估。\n")

    # 2. 初始化结果容器
    direct_answers = []
    rag_answers_list = []
    rag_retrievals = []

    # 3. 逐题生成 Direct 和 RAG 答案
    for i, row in df.iterrows():
        qid = row["Test_ID"]
        question = str(row["Test_Question"]).strip()
        difficulty = row.get("Difficulty", "?")

        print(f"[{i + 1}/{n}] Q{qid} [{difficulty}] {question[:60]}...")

        if has_gold_answer:
            gold = str(row.get("Gold_Answer", "")).strip()
            print(f"      标准答案: {gold[:80]}...")

        # Direct Q&A
        try:
            direct = direct_answer(question)
        except Exception as e:
            direct = f"[ERROR] {type(e).__name__}: {e}"
        direct_answers.append(direct)
        print(f"      Direct: {direct[:100]}...")

        # RAG Q&A
        try:
            rag, retrieved = rag_answer(question)
        except Exception as e:
            rag = f"[ERROR] {type(e).__name__}: {e}"
            retrieved = []
        rag_answers_list.append(rag)
        rag_retrievals.append(retrieved)
        print(f"      RAG:    {rag[:100]}...")
        print(f"      检索数量: {len(retrieved)}")

        # 短暂暂停，避免打爆 vLLM。取消 LLM 评分后，每题只剩 Direct + RAG 两次生成请求。
        time.sleep(0.5)
        print()

    # 4. 保存详细结果到 Excel
    df["Direct_Answer"] = direct_answers
    df["RAG_Answer"] = rag_answers_list
    df["RAG_Retrieved_Questions"] = [
        format_retrieved_questions(ret)
        for ret in rag_retrievals
    ]
    df["RAG_Retrieved_Details"] = [
        format_retrieved_details(ret)
        for ret in rag_retrievals
    ]
    df["RAG_Retrieved_Count"] = [len(ret) for ret in rag_retrievals]

    df.to_excel(OUTPUT_XLSX, index=False)
    print(f"[INFO] 详细结果已保存到: {OUTPUT_XLSX}")

    # 5. 输出不含 LLM 评分的人工评估摘要
    direct_error_count = sum(1 for answer in direct_answers if answer.startswith("[ERROR]"))
    rag_error_count = sum(1 for answer in rag_answers_list if answer.startswith("[ERROR]"))
    no_retrieval_count = sum(1 for ret in rag_retrievals if not ret)
    avg_retrieved_count = (
        sum(len(ret) for ret in rag_retrievals) / n
        if n else 0.0
    )

    print("\n" + "=" * 60)
    print("批量生成摘要（无 LLM 自动评分）")
    print("=" * 60)
    print(f"总题数: {n}")
    print(f"Direct 生成错误数: {direct_error_count}")
    print(f"RAG 生成错误数: {rag_error_count}")
    print(f"RAG 无检索结果题数: {no_retrieval_count}")
    print(f"RAG 平均检索条数: {avg_retrieved_count:.2f}")
    print("\n前 5 题答案示例：")
    print("-" * 60)

    for i in range(min(5, n)):
        print(f"\n--- Q{df.iloc[i]['Test_ID']} [{df.iloc[i].get('Difficulty', '?')}] {df.iloc[i]['Test_Question']} ---")
        if has_gold_answer:
            print(f"【标准答案】{str(df.iloc[i].get('Gold_Answer', ''))[:200]}")
        print(f"【Direct Q&A】{direct_answers[i][:200]}")
        print(f"【RAG      】{rag_answers_list[i][:200]}")
        print(f"【检索资料数】{len(rag_retrievals[i])}")

    # 6. 保存 JSON 摘要
    report = {
        "summary": {
            "total_questions": n,
            "llm_scoring_enabled": False,
            "direct_error_count": direct_error_count,
            "rag_error_count": rag_error_count,
            "rag_no_retrieval_count": no_retrieval_count,
            "rag_avg_retrieved_count": avg_retrieved_count,
            "output_xlsx": OUTPUT_XLSX,
        },
        "config": {
            "top_k": TOP_K,
            "use_reranker": USE_RERANKER,
            "rerank_top_n": RERANK_TOP_N,
            "vllm_model": VLLM_MODEL,
        },
        "items": [],
    }

    for i, row in df.iterrows():
        item = {
            "test_id": json_value(row["Test_ID"]),
            "question": json_value(row["Test_Question"]),
            "difficulty": json_value(row.get("Difficulty", "")),
            "theme": json_value(row.get("Theme", "")),
            "gold_answer": json_value(row.get("Gold_Answer", "")) if has_gold_answer else "",
            "direct_answer": direct_answers[i],
            "rag_answer": rag_answers_list[i],
            "rag_retrieved_count": len(rag_retrievals[i]),
            "rag_retrieved_questions": [
                r.get("question", "")
                for r in rag_retrievals[i]
            ],
        }
        report["items"].append(item)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n[INFO] JSON 摘要已保存到: {OUTPUT_JSON}")
    print("[INFO] 批量生成完成，未执行 LLM 自动评分。")


if __name__ == "__main__":
    main()
