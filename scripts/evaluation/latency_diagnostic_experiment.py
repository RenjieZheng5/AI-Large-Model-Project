"""Latency diagnostics for Direct QA vs Basic RAG.

This experiment explains why Basic RAG can be faster than Direct QA even though
it adds retrieval. It reruns both systems, records API token usage when vLLM
returns it, separates retrieval and generation time, and creates diagnostic
figures for the report.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from statistics import mean
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_DIR = os.path.join(PROJECT_DIR, "src")
sys.path.insert(0, SRC_DIR)

from rag_config import RERANK_TOP_N, TOP_K, VLLM_MODEL, VLLM_URL  # noqa: E402
from rag_chat import FaissRetriever, Qwen3Embedder, build_context, strip_thinking  # noqa: E402
from rag_config import EMBED_DEVICE, EMBED_MODEL_PATH, INDEX_DIR, MAX_EMBED_LEN  # noqa: E402


DATA_DIR = os.path.join(PROJECT_DIR, "data")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
INPUT_XLSX = os.path.join(DATA_DIR, "sustech_rag_test_questions.xlsx")
OUTPUT_JSON = os.path.join(RESULTS_DIR, "latency_diagnostic_results.json")
OUTPUT_CSV = os.path.join(RESULTS_DIR, "latency_diagnostic_results.csv")
CHART_DIR = os.path.join(RESULTS_DIR, "charts")
SHEET_NAME = os.environ.get("EVAL_SHEET_NAME", "TestSet")


def estimate_tokens(text: Any) -> int:
    return math.ceil(len(str(text or "")) / 2)


def qualify_question(question: str) -> str:
    question = str(question or "").strip()
    if "南方科技大学" in question or "SUSTech" in question:
        return question
    return f"关于南方科技大学，{question}"


def call_chat(payload: Dict[str, Any], timeout: int = 300) -> Dict[str, Any]:
    start = time.perf_counter()
    resp = requests.post(VLLM_URL, json=payload, timeout=timeout)
    if resp.status_code >= 400 and "chat_template_kwargs" in resp.text:
        payload = dict(payload)
        payload.pop("chat_template_kwargs", None)
        resp = requests.post(VLLM_URL, json=payload, timeout=timeout)
    resp.raise_for_status()
    elapsed = time.perf_counter() - start
    data = resp.json()
    content = strip_thinking(data["choices"][0]["message"]["content"])
    usage = data.get("usage") or {}
    return {
        "answer": content,
        "api_latency_s": elapsed,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
    }


def direct_payload(question: str, max_tokens: int) -> Dict[str, Any]:
    return {
        "model": VLLM_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "你是一个有帮助的中文问答助手。请简洁准确地回答；如果不知道就说不知道。",
            },
            {"role": "user", "content": question},
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def basic_rag_payload(question: str, context: str, max_tokens: int) -> Dict[str, Any]:
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
    return {
        "model": VLLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
        "top_p": 0.8,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def init_retriever() -> FaissRetriever:
    print("[INFO] Loading retriever", flush=True)
    embedder = Qwen3Embedder(
        model_path=EMBED_MODEL_PATH,
        device=EMBED_DEVICE,
        max_length=MAX_EMBED_LEN,
    )
    return FaissRetriever(index_dir=INDEX_DIR, embedder=embedder)


def nonnull_mean(values: List[Optional[float]]) -> Optional[float]:
    clean = [float(v) for v in values if v is not None and not pd.isna(v)]
    return round(mean(clean), 4) if clean else None


def summarize(df: pd.DataFrame) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for system, group in df.groupby("system"):
        gen_tokens = group["completion_tokens"].dropna()
        gen_latency = group["generation_latency_s"].dropna()
        tps_values = [
            row.completion_tokens / row.generation_latency_s
            for row in group.itertuples()
            if pd.notna(row.completion_tokens) and row.generation_latency_s > 0
        ]
        summary[system] = {
            "n": int(len(group)),
            "avg_total_latency_s": nonnull_mean(group["total_latency_s"].tolist()),
            "avg_retrieval_latency_s": nonnull_mean(group["retrieval_latency_s"].tolist()),
            "avg_generation_latency_s": nonnull_mean(group["generation_latency_s"].tolist()),
            "avg_prompt_tokens": nonnull_mean(group["prompt_tokens"].tolist()),
            "avg_completion_tokens": nonnull_mean(group["completion_tokens"].tolist()),
            "avg_total_tokens": nonnull_mean(group["total_tokens"].tolist()),
            "avg_answer_chars": nonnull_mean(group["answer_chars"].tolist()),
            "avg_tokens_per_second": round(mean(tps_values), 4) if tps_values else None,
        }
        if len(gen_tokens) > 1 and len(gen_latency) > 1:
            summary[system]["latency_completion_token_corr"] = round(
                float(group[["generation_latency_s", "completion_tokens"]].corr().iloc[0, 1]),
                4,
            )
    return summary


def make_charts(df: pd.DataFrame, summary: Dict[str, Any]) -> List[str]:
    os.makedirs(CHART_DIR, exist_ok=True)
    paths: List[str] = []
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] matplotlib unavailable: {type(exc).__name__}: {exc}", flush=True)
        return paths

    systems = ["Direct", "Basic_RAG"]
    colors = {"Direct": "#4C78A8", "Basic_RAG": "#59A14F"}

    fig, ax = plt.subplots(figsize=(9.5, 5.2), dpi=180)
    x = range(len(systems))
    retrieval = [summary[s]["avg_retrieval_latency_s"] or 0.0 for s in systems]
    generation = [summary[s]["avg_generation_latency_s"] or 0.0 for s in systems]
    ax.bar(x, retrieval, label="Retrieval", color="#9CA3AF")
    for idx, system in enumerate(systems):
        ax.bar(
            idx,
            generation[idx],
            bottom=retrieval[idx],
            label=f"{system} generation",
            color=colors[system],
        )
    for idx, s in enumerate(systems):
        total = summary[s]["avg_total_latency_s"] or 0.0
        comp = summary[s]["avg_completion_tokens"] or 0.0
        prompt = summary[s]["avg_prompt_tokens"] or 0.0
        ax.text(idx, total + 0.12, f"{total:.2f}s\nout={comp:.1f}, in={prompt:.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(list(x))
    ax.set_xticklabels(systems, fontweight="semibold")
    ax.set_ylabel("Seconds")
    ax.set_title("Latency Decomposition: Direct vs Basic RAG", fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = os.path.join(CHART_DIR, "rag_latency_diagnostic_breakdown.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(9.5, 5.2), dpi=180)
    for system in systems:
        g = df[df["system"] == system]
        ax.scatter(
            g["completion_tokens"],
            g["generation_latency_s"],
            label=system,
            color=colors[system],
            alpha=0.78,
            s=42,
        )
    ax.set_xlabel("Completion tokens")
    ax.set_ylabel("LLM generation latency (s)")
    ax.set_title("Generation Latency Tracks Output Length", fontweight="bold")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = os.path.join(CHART_DIR, "rag_latency_vs_output_tokens.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    return paths


def run(args: argparse.Namespace) -> Dict[str, Any]:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(CHART_DIR, exist_ok=True)

    df = pd.read_excel(INPUT_XLSX, sheet_name=SHEET_NAME)
    if args.limit:
        df = df.head(args.limit)

    retriever = init_retriever()
    rows: List[Dict[str, Any]] = []

    for rep in range(1, args.repetitions + 1):
        for idx, row in df.iterrows():
            test_id = row.get("Test_ID", idx + 1)
            question = qualify_question(str(row["Test_Question"]).strip())
            print(f"[{rep}/{args.repetitions}] Q{test_id} Direct", flush=True)

            direct = call_chat(direct_payload(question, args.max_tokens), timeout=args.timeout)
            rows.append(
                {
                    "repeat": rep,
                    "test_id": test_id,
                    "system": "Direct",
                    "retrieval_latency_s": 0.0,
                    "generation_latency_s": round(direct["api_latency_s"], 4),
                    "total_latency_s": round(direct["api_latency_s"], 4),
                    "prompt_tokens": direct["prompt_tokens"],
                    "completion_tokens": direct["completion_tokens"],
                    "total_tokens": direct["total_tokens"],
                    "answer_chars": len(direct["answer"]),
                    "approx_prompt_tokens": estimate_tokens(question),
                    "approx_completion_tokens": estimate_tokens(direct["answer"]),
                }
            )

            print(f"[{rep}/{args.repetitions}] Q{test_id} Basic_RAG", flush=True)
            retrieval_start = time.perf_counter()
            retrieved = retriever.search(question, top_k=TOP_K)
            retrieval_latency = time.perf_counter() - retrieval_start
            context = build_context(retrieved[:RERANK_TOP_N])
            rag = call_chat(basic_rag_payload(question, context, args.max_tokens), timeout=args.timeout)
            total_latency = retrieval_latency + rag["api_latency_s"]
            rows.append(
                {
                    "repeat": rep,
                    "test_id": test_id,
                    "system": "Basic_RAG",
                    "retrieval_latency_s": round(retrieval_latency, 4),
                    "generation_latency_s": round(rag["api_latency_s"], 4),
                    "total_latency_s": round(total_latency, 4),
                    "prompt_tokens": rag["prompt_tokens"],
                    "completion_tokens": rag["completion_tokens"],
                    "total_tokens": rag["total_tokens"],
                    "answer_chars": len(rag["answer"]),
                    "approx_prompt_tokens": estimate_tokens(question + "\n" + context),
                    "approx_completion_tokens": estimate_tokens(rag["answer"]),
                }
            )

            pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    result_df = pd.DataFrame(rows)
    summary = summarize(result_df)
    chart_paths = make_charts(result_df, summary)
    result = {
        "config": {
            "input_xlsx": INPUT_XLSX,
            "sheet_name": SHEET_NAME,
            "vllm_url": VLLM_URL,
            "vllm_model": VLLM_MODEL,
            "top_k": TOP_K,
            "context_top_n": RERANK_TOP_N,
            "repetitions": args.repetitions,
            "max_tokens": args.max_tokens,
        },
        "summary": summary,
        "charts": chart_paths,
        "csv": OUTPUT_CSV,
    }
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="Run only the first N questions.")
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
