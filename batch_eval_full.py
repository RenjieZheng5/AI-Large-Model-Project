"""
Full RAG evaluation for the SUSTech campus QA project.

The script compares:
- Direct LLM: no retrieval.
- Basic RAG: FAISS retrieval + generation.
- RAG + rerank: FAISS retrieval + BGE reranker + generation, when USE_RERANKER=1.

It reports retrieval metrics, answer-quality metrics, trust metrics, and latency/cost proxies.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import time
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse, urlunparse

import pandas as pd
import requests

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from rag_config import (  # noqa: E402
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
from rag_chat import FaissRetriever, Qwen3Embedder, build_context, call_vllm, strip_thinking  # noqa: E402
from reranker import BGEReranker  # noqa: E402


INPUT_XLSX = os.path.join(PROJECT_DIR, "sustech_rag_test_questions.xlsx")
OUTPUT_XLSX = os.path.join(PROJECT_DIR, "sustech_rag_batch_results_full.xlsx")
OUTPUT_JSON = os.path.join(PROJECT_DIR, "sustech_rag_batch_results_full.json")
PARTIAL_OUTPUT_XLSX = os.path.join(PROJECT_DIR, "sustech_rag_batch_results_full.partial.xlsx")
CHART_DIR = os.path.join(PROJECT_DIR, "eval_charts")
SHEET_NAME = os.environ.get("EVAL_SHEET_NAME", "TestSet")

LLM_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "512"))
JUDGE_MAX_TOKENS = int(os.environ.get("JUDGE_MAX_TOKENS", "512"))
REWRITE_MAX_TOKENS = int(os.environ.get("REWRITE_MAX_TOKENS", "96"))
RUN_LLM_JUDGE = os.environ.get("RUN_LLM_JUDGE", "1") == "1"
ENABLE_QUERY_REWRITE = os.environ.get("ENABLE_QUERY_REWRITE", "1") == "1"
RETRIEVAL_KS = [1, 3, 5]

_embedder = None
_retriever = None
_reranker = None


def log(message: str = "") -> None:
    print(message, flush=True)


def normalize_text(text: Any) -> str:
    text = str(text or "").lower().strip()
    text = re.sub(r"\s+", "", text)
    return text


def normalize_url(url: Any) -> str:
    url = str(url or "").strip()
    if not url:
        return ""
    parsed = urlparse(url.split("#")[0])
    if not parsed.scheme:
        return url.rstrip("/").lower()
    normalized = urlunparse(
        (
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            parsed.path.rstrip("/"),
            "",
            parsed.query,
            "",
        )
    )
    return normalized.lower()


def text_similarity(a: Any, b: Any) -> float:
    a_norm = normalize_text(a)
    b_norm = normalize_text(b)
    if not a_norm or not b_norm:
        return 0.0
    return SequenceMatcher(None, a_norm, b_norm).ratio()


def estimate_tokens(text: Any) -> int:
    # A simple language-agnostic proxy for relative cost comparisons.
    return math.ceil(len(str(text or "")) / 2)


def parse_bool(value: Any, default: bool = True) -> bool:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return default
    text = str(value).strip().lower()
    if text in {"false", "0", "no", "n", "否", "不可回答", "unanswerable"}:
        return False
    if text in {"true", "1", "yes", "y", "是", "可回答", "answerable"}:
        return True
    return default


def row_answerable(row: pd.Series) -> bool:
    for col in ["Answerable", "answerable", "Is_Answerable", "是否可回答"]:
        if col in row:
            return parse_bool(row.get(col), default=True)
    return True


def source_url_from_item(item: Dict[str, Any]) -> str:
    raw = item.get("raw", {})
    if isinstance(raw, dict):
        return str(raw.get("source_url") or raw.get("url") or "")
    return ""


def source_title_from_item(item: Dict[str, Any]) -> str:
    raw = item.get("raw", {})
    if isinstance(raw, dict):
        return str(raw.get("source_title") or raw.get("title") or "")
    return ""


def relevance_grade(item: Dict[str, Any], row: pd.Series) -> int:
    """0-3 evidence relevance grade used for deterministic retrieval metrics."""
    expected_url = normalize_url(row.get("Source_URL", ""))
    expected_q = normalize_text(row.get("Original_Question", ""))
    gold = str(row.get("Gold_Answer", "")).strip()

    item_url = normalize_url(source_url_from_item(item))
    item_q = normalize_text(item.get("question", ""))
    item_a = str(item.get("answer", "")).strip()

    question_match = bool(expected_q and item_q and expected_q == item_q)
    source_match = bool(expected_url and item_url and expected_url == item_url)
    answer_sim = text_similarity(gold, item_a)

    if question_match or answer_sim >= 0.82:
        return 3
    if source_match:
        return 3 if answer_sim >= 0.45 else 2
    if answer_sim >= 0.55:
        return 2
    if answer_sim >= 0.35:
        return 1
    return 0


def dcg(relevance: Iterable[int]) -> float:
    return sum((2**rel - 1) / math.log2(i + 2) for i, rel in enumerate(relevance))


def retrieval_metrics(results: List[Dict[str, Any]], row: pd.Series, k_values: List[int]) -> Dict[str, Any]:
    grades = [relevance_grade(item, row) for item in results]
    metrics: Dict[str, Any] = {
        "evidence_rank": None,
        "context_relevance": max(grades) if grades else 0,
    }

    for i, grade in enumerate(grades, start=1):
        if grade >= 2:
            metrics["evidence_rank"] = i
            break

    for k in k_values:
        top = grades[:k]
        hit = any(g >= 2 for g in top)
        metrics[f"recall@{k}"] = 1.0 if hit else 0.0
        metrics[f"precision@{k}"] = (sum(1 for g in top if g >= 2) / k) if k else 0.0

        first_rank = next((idx for idx, g in enumerate(top, start=1) if g >= 2), None)
        metrics[f"mrr@{k}"] = 1.0 / first_rank if first_rank else 0.0

        ideal = sorted(grades, reverse=True)[:k]
        ideal_dcg = dcg(ideal)
        metrics[f"ndcg@{k}"] = dcg(top) / ideal_dcg if ideal_dcg > 0 else 0.0

    return metrics


def post_chat(payload: Dict[str, Any], timeout: int = 300) -> str:
    resp = requests.post(VLLM_URL, json=payload, timeout=timeout)
    if resp.status_code >= 400 and "chat_template_kwargs" in resp.text:
        payload = dict(payload)
        payload.pop("chat_template_kwargs", None)
        resp = requests.post(VLLM_URL, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return strip_thinking(data["choices"][0]["message"]["content"])


def qualify_question(question: str) -> str:
    question = str(question or "").strip()
    if "南方科技大学" in question:
        return question
    return f"关于南方科技大学，{question}"


def rewrite_query(question: str) -> Tuple[str, float, str]:
    """Use the LLM to rewrite a user question into a retrieval-oriented SUSTech query."""
    start = time.perf_counter()
    prompt = f"""请将下面的问题改写成适合检索“南方科技大学”校园知识库的中文查询。
要求：
1. 必须保留原问题的事实约束和询问目标。
2. 必须包含“南方科技大学”这个限定。
3. 不要回答问题，不要解释，只输出一个改写后的查询。

原问题：{question}
改写查询："""
    payload = {
        "model": VLLM_MODEL,
        "messages": [
            {"role": "system", "content": "你是检索查询改写助手，只输出改写后的查询。"},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": REWRITE_MAX_TOKENS,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    try:
        rewritten = post_chat(payload, timeout=120)
        rewritten = strip_thinking(rewritten).strip()
        rewritten = re.sub(r"^改写查询[:：]\s*", "", rewritten).strip()
        rewritten = rewritten.strip("\"'“”‘’` \n\r\t")
        rewritten = rewritten.splitlines()[0].strip() if rewritten else ""
        rewritten = qualify_question(rewritten or question)
        return rewritten, time.perf_counter() - start, ""
    except Exception as e:
        return qualify_question(question), time.perf_counter() - start, f"{type(e).__name__}: {e}"


def direct_answer(question: str) -> str:
    payload = {
        "model": VLLM_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "你是一个有帮助的中文问答助手。请简洁准确地回答；如果不知道就说不知道。",
            },
            {"role": "user", "content": question},
        ],
        "temperature": 0.0,
        "max_tokens": LLM_MAX_TOKENS,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    return post_chat(payload, timeout=300)


def init_rag() -> None:
    global _embedder, _retriever, _reranker
    if _retriever is not None:
        return

    print("=" * 80)
    print("[INFO] Initializing RAG components")
    print(f"  VLLM URL      : {VLLM_URL}")
    print(f"  VLLM model    : {VLLM_MODEL}")
    print(f"  Index dir     : {INDEX_DIR}")
    print(f"  Embed model   : {EMBED_MODEL_PATH}")
    print(f"  Embed device  : {EMBED_DEVICE}")
    print(f"  TOP_K         : {TOP_K}")
    print(f"  Reranker      : {'ON' if USE_RERANKER else 'OFF'}")
    print("=" * 80)

    _embedder = Qwen3Embedder(
        model_path=EMBED_MODEL_PATH,
        device=EMBED_DEVICE,
        max_length=MAX_EMBED_LEN,
    )
    _retriever = FaissRetriever(index_dir=INDEX_DIR, embedder=_embedder)

    if USE_RERANKER:
        _reranker = BGEReranker(
            model_path=RERANK_MODEL_PATH,
            device=os.environ.get("RERANK_DEVICE", "cpu"),
            batch_size=RERANK_BATCH_SIZE,
        )


def run_rag(
    question: str,
    use_reranker: bool,
    retrieval_query: Optional[str] = None,
    faiss_results: Optional[List[Dict[str, Any]]] = None,
    retrieval_s: Optional[float] = None,
) -> Tuple[str, List[Dict[str, Any]], Dict[str, float], str]:
    init_rag()
    timings: Dict[str, float] = {}
    search_query = retrieval_query or question

    if faiss_results is None:
        start = time.perf_counter()
        faiss_results = _retriever.search(search_query, top_k=TOP_K)
        timings["retrieval_s"] = time.perf_counter() - start
    else:
        timings["retrieval_s"] = float(retrieval_s or 0.0)

    if not faiss_results:
        return "没有检索到相关资料。", [], timings, ""

    start = time.perf_counter()
    if use_reranker and _reranker is not None:
        final_results = _reranker.rerank(search_query, faiss_results, top_n=RERANK_TOP_N)
    else:
        final_results = faiss_results[:RERANK_TOP_N]
    timings["rerank_s"] = time.perf_counter() - start

    context = build_context(final_results)
    start = time.perf_counter()
    answer = call_vllm(question, context)
    timings["generation_s"] = time.perf_counter() - start
    timings["total_s"] = sum(timings.values())
    return answer, final_results, timings, context


def extract_json_object(raw: str) -> Dict[str, Any]:
    text = strip_thinking(raw)
    text = re.sub(r"^```(?:json)?", "", text.strip(), flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text.strip()).strip()

    start = text.find("{")
    if start < 0:
        raise ValueError(f"No JSON object found: {raw[:160]}")

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(text[start : i + 1])

    raise ValueError(f"Unclosed JSON object: {raw[:160]}")


def clamp_int(value: Any, low: int, high: int, default: int = -1) -> int:
    try:
        number = int(value)
    except Exception:
        return default
    return max(low, min(high, number))


def judge_answer(
    question: str,
    gold: str,
    answer: str,
    context: str,
    answerable: bool,
    system_name: str,
) -> Dict[str, Any]:
    if not RUN_LLM_JUDGE:
        return {
            "answer_correctness": -1,
            "answer_relevance": -1,
            "completeness": -1,
            "faithfulness": -1,
            "groundedness": -1,
            "citation_accuracy": -1,
            "hallucination": -1,
            "refusal_correct": -1,
            "judge_reason": "LLM judge disabled",
        }

    context_block = context if context else "（该系统没有提供检索上下文）"
    prompt = f"""你是严格的 RAG 评测员。请根据问题、标准答案、系统答案和检索上下文打分。

评分规则：
- answer_correctness, answer_relevance, completeness: 0-3 分，3 最好。
- faithfulness: 0-3 分，评估答案是否被检索上下文支持；无检索上下文的 Direct LLM 填 -1。
- groundedness: 0-3 分，评估答案是否有明确依据；无检索上下文的 Direct LLM 填 -1。
- citation_accuracy: 0-3 分，评估检索来源/上下文是否真的支持答案；无检索上下文的 Direct LLM 填 -1。
- hallucination: 0 或 1，若答案包含无依据编造、与标准答案冲突或知识库不足时乱答，填 1，否则填 0。
- refusal_correct: 0 或 1。若 answerable=false 且系统正确拒答/说明资料不足，填 1；其他情况填 0。

请只输出一个 JSON 对象，不要 Markdown，不要解释。

系统：{system_name}
answerable: {str(answerable).lower()}
问题：{question}
标准答案：{gold}
检索上下文：
{context_block}

系统答案：
{answer}

JSON 格式：
{{
  "answer_correctness": 0,
  "answer_relevance": 0,
  "completeness": 0,
  "faithfulness": -1,
  "groundedness": -1,
  "citation_accuracy": -1,
  "hallucination": 0,
  "refusal_correct": 0,
  "reason": "一句话说明"
}}"""

    payload = {
        "model": VLLM_MODEL,
        "messages": [
            {"role": "system", "content": "你是严格的评测员，只输出合法 JSON。"},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": JUDGE_MAX_TOKENS,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    raw = post_chat(payload, timeout=180)
    parsed = extract_json_object(raw)
    return {
        "answer_correctness": clamp_int(parsed.get("answer_correctness"), 0, 3),
        "answer_relevance": clamp_int(parsed.get("answer_relevance"), 0, 3),
        "completeness": clamp_int(parsed.get("completeness"), 0, 3),
        "faithfulness": clamp_int(parsed.get("faithfulness"), -1, 3),
        "groundedness": clamp_int(parsed.get("groundedness"), -1, 3),
        "citation_accuracy": clamp_int(parsed.get("citation_accuracy"), -1, 3),
        "hallucination": clamp_int(parsed.get("hallucination"), 0, 1),
        "refusal_correct": clamp_int(parsed.get("refusal_correct"), 0, 1),
        "judge_reason": str(parsed.get("reason", "")).strip(),
    }


def safe_judge(*args, **kwargs) -> Dict[str, Any]:
    try:
        return judge_answer(*args, **kwargs)
    except Exception as e:
        return {
            "answer_correctness": -1,
            "answer_relevance": -1,
            "completeness": -1,
            "faithfulness": -1,
            "groundedness": -1,
            "citation_accuracy": -1,
            "hallucination": -1,
            "refusal_correct": -1,
            "judge_reason": f"[JUDGE_ERROR] {type(e).__name__}: {e}",
        }


def valid_values(rows: List[Dict[str, Any]], column: str) -> List[float]:
    values = []
    for row in rows:
        value = row.get(column)
        if value is None:
            continue
        try:
            value = float(value)
        except Exception:
            continue
        if value >= 0:
            values.append(value)
    return values


def mean(values: List[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def score_summary(rows: List[Dict[str, Any]], prefix: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for metric in [
        "answer_correctness",
        "answer_relevance",
        "completeness",
        "faithfulness",
        "groundedness",
        "citation_accuracy",
    ]:
        values = valid_values(rows, f"{prefix}_{metric}")
        out[metric] = {
            "count": len(values),
            "avg": round(mean(values), 4) if values else None,
            "pass_rate_ge_2": round(sum(v >= 2 for v in values) / len(values), 4) if values else None,
            "perfect_rate_3": round(sum(v == 3 for v in values) / len(values), 4) if values else None,
        }

    hallucinations = valid_values(rows, f"{prefix}_hallucination")
    out["hallucination_rate"] = round(mean(hallucinations), 4) if hallucinations else None

    refusals = valid_values(rows, f"{prefix}_refusal_correct")
    out["refusal_correct_rate"] = round(mean(refusals), 4) if refusals else None

    latencies = valid_values(rows, f"{prefix}_total_latency_s")
    out["avg_total_latency_s"] = round(mean(latencies), 4) if latencies else None

    input_tokens = valid_values(rows, f"{prefix}_approx_input_tokens")
    output_tokens = valid_values(rows, f"{prefix}_approx_output_tokens")
    out["avg_approx_input_tokens"] = round(mean(input_tokens), 2) if input_tokens else None
    out["avg_approx_output_tokens"] = round(mean(output_tokens), 2) if output_tokens else None
    return out


def retrieval_summary(rows: List[Dict[str, Any]], prefix: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for metric in ["context_relevance", "evidence_rank"]:
        values = valid_values(rows, f"{prefix}_{metric}")
        out[metric] = round(mean(values), 4) if values else None
    for k in RETRIEVAL_KS:
        for metric in [f"recall@{k}", f"precision@{k}", f"mrr@{k}", f"ndcg@{k}"]:
            values = valid_values(rows, f"{prefix}_{metric}")
            out[metric] = round(mean(values), 4) if values else None
    return out


def add_judge_columns(record: Dict[str, Any], prefix: str, judge: Dict[str, Any]) -> None:
    for key, value in judge.items():
        record[f"{prefix}_{key}"] = value


def add_retrieval_columns(record: Dict[str, Any], prefix: str, metrics: Dict[str, Any]) -> None:
    for key, value in metrics.items():
        record[f"{prefix}_{key}"] = value


def compact_retrieved(results: List[Dict[str, Any]]) -> Tuple[str, str]:
    questions = []
    sources = []
    for item in results:
        q = str(item.get("question", "")).replace("\n", " ").strip()
        questions.append(q[:80])
        src = source_url_from_item(item) or source_title_from_item(item)
        sources.append(str(src)[:120])
    return " | ".join(questions), " | ".join(sources)


def evaluate() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df = pd.read_excel(INPUT_XLSX, sheet_name=SHEET_NAME)
    required_cols = {"Test_ID", "Test_Question", "Gold_Answer"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"TestSet missing columns: {missing}; actual columns: {list(df.columns)}")

    rows: List[Dict[str, Any]] = []
    use_rerank_system = USE_RERANKER
    use_rewrite_system = ENABLE_QUERY_REWRITE and USE_RERANKER

    log(f"[INFO] Loaded {len(df)} evaluation questions from {INPUT_XLSX}:{SHEET_NAME}")
    log(f"[INFO] LLM judge: {'ON' if RUN_LLM_JUDGE else 'OFF'}")
    log(f"[INFO] Query rewrite system: {'ON' if use_rewrite_system else 'OFF'}")

    for index, row in df.iterrows():
        raw_question = str(row["Test_Question"]).strip()
        question = qualify_question(raw_question)
        gold = str(row["Gold_Answer"]).strip()
        answerable = row_answerable(row)
        record = row.to_dict()
        record["Answerable"] = answerable
        record["Eval_Question"] = question

        log(f"\n[{index + 1}/{len(df)}] Q{row['Test_ID']} {question[:80]}")

        direct_start = time.perf_counter()
        try:
            direct = direct_answer(question)
        except Exception as e:
            direct = f"[ERROR] {type(e).__name__}: {e}"
        direct_latency = time.perf_counter() - direct_start

        record["Direct_Answer"] = direct
        record["Direct_total_latency_s"] = round(direct_latency, 4)
        record["Direct_approx_input_tokens"] = estimate_tokens(question)
        record["Direct_approx_output_tokens"] = estimate_tokens(direct)
        add_judge_columns(
            record,
            "Direct",
            safe_judge(question, gold, direct, "", answerable, "Direct LLM"),
        )
        log(f"  Direct correctness={record['Direct_answer_correctness']} hallucination={record['Direct_hallucination']}")

        rag_systems = [("Basic_RAG", False)]
        if use_rerank_system:
            rag_systems.append(("RAG_Rerank", True))

        try:
            init_rag()
            shared_start = time.perf_counter()
            shared_faiss_results = _retriever.search(question, top_k=TOP_K)
            shared_retrieval_s = time.perf_counter() - shared_start
            log(f"  FAISS retrieved={len(shared_faiss_results)} latency={shared_retrieval_s:.4f}s")
        except Exception as e:
            shared_faiss_results = None
            shared_retrieval_s = None
            log(f"  FAISS error: {type(e).__name__}: {e}")

        for prefix, use_reranker in rag_systems:
            try:
                answer, retrieved, timings, context = run_rag(
                    question,
                    use_reranker=use_reranker,
                    faiss_results=shared_faiss_results,
                    retrieval_s=shared_retrieval_s,
                )
            except Exception as e:
                answer, retrieved, timings, context = f"[ERROR] {type(e).__name__}: {e}", [], {}, ""

            record[f"{prefix}_Answer"] = answer
            record[f"{prefix}_retrieved_count"] = len(retrieved)
            record[f"{prefix}_retrieval_latency_s"] = round(timings.get("retrieval_s", 0.0), 4)
            record[f"{prefix}_rerank_latency_s"] = round(timings.get("rerank_s", 0.0), 4)
            record[f"{prefix}_generation_latency_s"] = round(timings.get("generation_s", 0.0), 4)
            record[f"{prefix}_total_latency_s"] = round(timings.get("total_s", 0.0), 4)
            record[f"{prefix}_approx_input_tokens"] = estimate_tokens(question + "\n" + context)
            record[f"{prefix}_approx_output_tokens"] = estimate_tokens(answer)

            retrieved_questions, retrieved_sources = compact_retrieved(retrieved)
            record[f"{prefix}_Retrieved_Questions"] = retrieved_questions
            record[f"{prefix}_Retrieved_Sources"] = retrieved_sources

            ret_metrics = retrieval_metrics(retrieved, row, RETRIEVAL_KS)
            add_retrieval_columns(record, prefix, ret_metrics)
            add_judge_columns(
                record,
                prefix,
                safe_judge(question, gold, answer, context, answerable, prefix),
            )
            log(
                f"  {prefix} recall@5={record.get(f'{prefix}_recall@5')} "
                f"correctness={record.get(f'{prefix}_answer_correctness')} "
                f"faithfulness={record.get(f'{prefix}_faithfulness')} "
                f"latency={record.get(f'{prefix}_total_latency_s')}s"
            )

        if use_rewrite_system:
            prefix = "RAG_Rewrite_Rerank"
            rewritten_query, rewrite_latency, rewrite_error = rewrite_query(question)
            record[f"{prefix}_Rewritten_Query"] = rewritten_query
            record[f"{prefix}_rewrite_latency_s"] = round(rewrite_latency, 4)
            record[f"{prefix}_rewrite_error"] = rewrite_error
            log(f"  Rewrite query: {rewritten_query[:100]}")

            try:
                init_rag()
                rewrite_retrieval_start = time.perf_counter()
                rewrite_faiss_results = _retriever.search(rewritten_query, top_k=TOP_K)
                rewrite_retrieval_s = time.perf_counter() - rewrite_retrieval_start
                answer, retrieved, timings, context = run_rag(
                    question,
                    use_reranker=True,
                    retrieval_query=rewritten_query,
                    faiss_results=rewrite_faiss_results,
                    retrieval_s=rewrite_retrieval_s,
                )
                timings["rewrite_s"] = rewrite_latency
                timings["total_s"] = timings.get("total_s", 0.0) + rewrite_latency
            except Exception as e:
                answer, retrieved, timings, context = f"[ERROR] {type(e).__name__}: {e}", [], {}, ""
                timings["rewrite_s"] = rewrite_latency
                timings["total_s"] = rewrite_latency

            record[f"{prefix}_Answer"] = answer
            record[f"{prefix}_retrieved_count"] = len(retrieved)
            record[f"{prefix}_retrieval_latency_s"] = round(timings.get("retrieval_s", 0.0), 4)
            record[f"{prefix}_rerank_latency_s"] = round(timings.get("rerank_s", 0.0), 4)
            record[f"{prefix}_generation_latency_s"] = round(timings.get("generation_s", 0.0), 4)
            record[f"{prefix}_total_latency_s"] = round(timings.get("total_s", 0.0), 4)
            record[f"{prefix}_approx_input_tokens"] = estimate_tokens(rewritten_query + "\n" + question + "\n" + context)
            record[f"{prefix}_approx_output_tokens"] = estimate_tokens(answer)

            retrieved_questions, retrieved_sources = compact_retrieved(retrieved)
            record[f"{prefix}_Retrieved_Questions"] = retrieved_questions
            record[f"{prefix}_Retrieved_Sources"] = retrieved_sources

            ret_metrics = retrieval_metrics(retrieved, row, RETRIEVAL_KS)
            add_retrieval_columns(record, prefix, ret_metrics)
            add_judge_columns(
                record,
                prefix,
                safe_judge(question, gold, answer, context, answerable, prefix),
            )
            log(
                f"  {prefix} recall@5={record.get(f'{prefix}_recall@5')} "
                f"correctness={record.get(f'{prefix}_answer_correctness')} "
                f"faithfulness={record.get(f'{prefix}_faithfulness')} "
                f"latency={record.get(f'{prefix}_total_latency_s')}s"
            )

        rows.append(record)
        pd.DataFrame(rows).to_excel(PARTIAL_OUTPUT_XLSX, index=False)
        log(f"  partial saved: {PARTIAL_OUTPUT_XLSX}")
        time.sleep(0.2)

    result_df = pd.DataFrame(rows)

    system_prefixes = ["Direct", "Basic_RAG"] + (["RAG_Rerank"] if use_rerank_system else [])
    if use_rewrite_system:
        system_prefixes.append("RAG_Rewrite_Rerank")
    report: Dict[str, Any] = {
        "config": {
            "input_xlsx": INPUT_XLSX,
            "sheet_name": SHEET_NAME,
            "vllm_url": VLLM_URL,
            "vllm_model": VLLM_MODEL,
            "index_dir": INDEX_DIR,
            "embedding_model": EMBED_MODEL_PATH,
            "top_k": TOP_K,
            "rerank_top_n": RERANK_TOP_N,
            "use_reranker": USE_RERANKER,
            "query_rewrite": use_rewrite_system,
            "llm_judge": RUN_LLM_JUDGE,
        },
        "total_questions": len(rows),
        "answerable_count": int(sum(1 for r in rows if r.get("Answerable"))),
        "unanswerable_count": int(sum(1 for r in rows if not r.get("Answerable"))),
        "systems": {prefix: score_summary(rows, prefix) for prefix in system_prefixes},
        "retrieval": {
            prefix: retrieval_summary(rows, prefix)
            for prefix in system_prefixes
            if prefix != "Direct"
        },
        "improvement_vs_direct": {},
    }

    direct_correct = report["systems"]["Direct"]["answer_correctness"]["pass_rate_ge_2"]
    direct_hallu = report["systems"]["Direct"]["hallucination_rate"]
    for prefix in system_prefixes:
        if prefix == "Direct":
            continue
        sys_correct = report["systems"][prefix]["answer_correctness"]["pass_rate_ge_2"]
        sys_hallu = report["systems"][prefix]["hallucination_rate"]
        report["improvement_vs_direct"][prefix] = {
            "answer_correctness_pass_rate_delta": (
                round(sys_correct - direct_correct, 4)
                if sys_correct is not None and direct_correct is not None
                else None
            ),
            "hallucination_rate_delta": (
                round(sys_hallu - direct_hallu, 4)
                if sys_hallu is not None and direct_hallu is not None
                else None
            ),
        }

    return result_df, report


def generate_charts(report: Dict[str, Any]) -> List[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        log(f"[WARN] Matplotlib unavailable, using Pillow chart fallback: {type(e).__name__}: {e}")
        return generate_charts_with_pillow(report)

    os.makedirs(CHART_DIR, exist_ok=True)
    systems = list(report.get("systems", {}).keys())
    colors = ["#4C78A8", "#59A14F", "#F28E2B", "#B279A2", "#E15759"]
    chart_paths: List[str] = []

    plt.rcParams.update(
        {
            "font.weight": "semibold",
            "axes.labelweight": "semibold",
            "axes.titleweight": "bold",
        }
    )

    def style_axis(ax: Any) -> None:
        ax.tick_params(axis="both", labelsize=10, width=1.2)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight("semibold")
        for spine in ax.spines.values():
            spine.set_linewidth(1.1)

    def value(path: List[str], default: float = 0.0) -> float:
        node: Any = report
        for key in path:
            if not isinstance(node, dict) or key not in node or node[key] is None:
                return default
            node = node[key]
        return float(node)

    quality_metrics = [
        ("Correctness pass", lambda s: value(["systems", s, "answer_correctness", "pass_rate_ge_2"])),
        ("Faithfulness pass", lambda s: value(["systems", s, "faithfulness", "pass_rate_ge_2"])),
        ("Citation pass", lambda s: value(["systems", s, "citation_accuracy", "pass_rate_ge_2"])),
        ("Hallucination rate", lambda s: value(["systems", s, "hallucination_rate"])),
    ]

    fig, ax = plt.subplots(figsize=(12, 6), dpi=160)
    width = 0.18
    x_positions = list(range(len(quality_metrics)))
    for i, system in enumerate(systems):
        values = [round(getter(system), 2) for _, getter in quality_metrics]
        xs = [x + (i - (len(systems) - 1) / 2) * width for x in x_positions]
        ax.bar(xs, values, width=width, label=system, color=colors[i % len(colors)])
        for x, y in zip(xs, values):
            ax.text(
                x,
                min(y + 0.025, 1.04),
                f"{y:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="semibold",
            )
    ax.set_title("SUSTech RAG Answer Quality Comparison", fontsize=16, fontweight="bold")
    ax.set_ylabel("Rate", fontsize=12, fontweight="semibold")
    ax.set_xticks(x_positions)
    ax.set_xticklabels([name for name, _ in quality_metrics], rotation=0, ha="center", fontweight="semibold")
    ax.set_ylim(0, 1.08)
    ax.grid(axis="y", alpha=0.25)
    style_axis(ax)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=min(len(systems), 4),
        frameon=False,
        prop={"weight": "semibold", "size": 10},
    )
    fig.tight_layout()
    quality_path = os.path.join(CHART_DIR, "rag_eval_quality_comparison.png")
    fig.savefig(quality_path, bbox_inches="tight")
    plt.close(fig)
    chart_paths.append(quality_path)

    retrieval_systems = list(report.get("retrieval", {}).keys())
    if retrieval_systems:
        retrieval_metrics_to_plot = ["recall@5", "precision@5", "mrr@5", "ndcg@5"]
        fig, ax = plt.subplots(figsize=(11, 5.5), dpi=160)
        width = 0.2
        x_positions = list(range(len(retrieval_metrics_to_plot)))
        for i, system in enumerate(retrieval_systems):
            values = [round(value(["retrieval", system, metric]), 2) for metric in retrieval_metrics_to_plot]
            xs = [x + (i - (len(retrieval_systems) - 1) / 2) * width for x in x_positions]
            ax.bar(xs, values, width=width, label=system, color=colors[(i + 1) % len(colors)])
            for x, y in zip(xs, values):
                ax.text(
                    x,
                    min(y + 0.025, 1.04),
                    f"{y:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="semibold",
                )
        ax.set_title("SUSTech RAG Retrieval Quality @5", fontsize=16, fontweight="bold")
        ax.set_ylabel("Score", fontsize=12, fontweight="semibold")
        ax.set_xticks(x_positions)
        ax.set_xticklabels(retrieval_metrics_to_plot, fontweight="semibold")
        ax.set_ylim(0, 1.08)
        ax.grid(axis="y", alpha=0.25)
        style_axis(ax)
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
            ncol=min(len(retrieval_systems), 3),
            frameon=False,
            prop={"weight": "semibold", "size": 10},
        )
        fig.tight_layout()
        retrieval_path = os.path.join(CHART_DIR, "rag_eval_retrieval_comparison.png")
        fig.savefig(retrieval_path, bbox_inches="tight")
        plt.close(fig)
        chart_paths.append(retrieval_path)

    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=160)
    latencies = [value(["systems", system, "avg_total_latency_s"]) for system in systems]
    bars = ax.bar(systems, latencies, color=[colors[i % len(colors)] for i in range(len(systems))])
    for bar, latency in zip(bars, latencies):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            latency + 0.08,
            f"{latency:.2f}s",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="semibold",
        )
    ax.set_title("SUSTech RAG Average Latency", fontsize=16, fontweight="bold")
    ax.set_ylabel("Seconds", fontsize=12, fontweight="semibold")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=0)
    style_axis(ax)
    fig.tight_layout()
    latency_path = os.path.join(CHART_DIR, "rag_eval_latency_comparison.png")
    fig.savefig(latency_path, bbox_inches="tight")
    plt.close(fig)
    chart_paths.append(latency_path)

    return chart_paths


def generate_charts_with_pillow(report: Dict[str, Any]) -> List[str]:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception as e:
        log(f"[WARN] Chart generation skipped: {type(e).__name__}: {e}")
        return []

    os.makedirs(CHART_DIR, exist_ok=True)
    systems = list(report.get("systems", {}).keys())
    retrieval_systems = list(report.get("retrieval", {}).keys())
    colors = ["#4C78A8", "#59A14F", "#F28E2B", "#B279A2", "#E15759"]
    chart_paths: List[str] = []

    def value(path: List[str], default: float = 0.0) -> float:
        node: Any = report
        for key in path:
            if not isinstance(node, dict) or key not in node or node[key] is None:
                return default
            node = node[key]
        return float(node)

    def load_bold_font(size: int) -> Any:
        font_candidates = [
            os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts", "arialbd.ttf"),
            os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts", "tahomabd.ttf"),
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
        ]
        for font_path in font_candidates:
            if os.path.exists(font_path):
                try:
                    return ImageFont.truetype(font_path, size=size)
                except Exception:
                    pass
        return ImageFont.load_default()

    def draw_centered_text(draw: Any, center_x: float, y: float, text: str, font: Any, fill: str) -> None:
        bbox = draw.textbbox((0, 0), text, font=font)
        draw.text((center_x - (bbox[2] - bbox[0]) / 2, y), text, fill=fill, font=font)

    def draw_grouped_bar(
        filename: str,
        title: str,
        categories: List[str],
        series: List[str],
        data: List[List[float]],
        y_max: float,
        suffix: str = "",
    ) -> str:
        width, height = 1500, 850
        margin_left, margin_right, margin_top, margin_bottom = 120, 80, 110, 210
        plot_w = width - margin_left - margin_right
        plot_h = height - margin_top - margin_bottom
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)
        font = load_bold_font(26)
        value_font = load_bold_font(24)
        title_font = load_bold_font(42)
        legend_font = load_bold_font(26)

        draw.text((margin_left, 38), title, fill="#222222", font=title_font)
        for i in range(6):
            y_value = y_max * i / 5
            y = margin_top + plot_h - int((y_value / y_max) * plot_h)
            draw.line((margin_left, y, width - margin_right, y), fill="#E5E7EB", width=1)
            draw.text((42, y - 7), f"{y_value:.1f}{suffix}", fill="#555555", font=font)

        group_w = plot_w / max(len(categories), 1)
        bar_w = min(70, group_w / (max(len(series), 1) + 1.2))
        for c_idx, category in enumerate(categories):
            center = margin_left + group_w * c_idx + group_w / 2
            draw_centered_text(draw, center, height - margin_bottom + 32, category, font, "#333333")
            for s_idx, system in enumerate(series):
                val = data[s_idx][c_idx]
                x0 = center + (s_idx - (len(series) - 1) / 2) * bar_w * 1.15 - bar_w / 2
                x1 = x0 + bar_w
                y1 = margin_top + plot_h
                y0 = y1 - int((min(val, y_max) / y_max) * plot_h)
                draw.rectangle((x0, y0, x1, y1), fill=colors[s_idx % len(colors)])
                draw_centered_text(draw, (x0 + x1) / 2, y0 - 32, f"{val:.2f}", value_font, "#333333")

        legend_items = []
        legend_gap = 54
        for system in series:
            text_bbox = draw.textbbox((0, 0), system, font=legend_font)
            text_w = text_bbox[2] - text_bbox[0]
            legend_items.append((system, 24 + 10 + text_w))
        total_legend_w = sum(width for _, width in legend_items) + legend_gap * max(len(legend_items) - 1, 0)
        legend_x = max(margin_left, int((width - total_legend_w) / 2))
        legend_y = height - 105
        cursor_x = legend_x
        for i, (system, item_w) in enumerate(legend_items):
            x = cursor_x
            y = legend_y
            draw.rectangle((x, y, x + 24, y + 16), fill=colors[i % len(colors)])
            draw.text((x + 34, y - 7), system, fill="#333333", font=legend_font)
            cursor_x += item_w + legend_gap

        path = os.path.join(CHART_DIR, filename)
        image.save(path)
        return path

    def draw_simple_bar(
        filename: str,
        title: str,
        categories: List[str],
        values: List[float],
        y_max: float,
        suffix: str = "",
    ) -> str:
        width, height = 1500, 850
        margin_left, margin_right, margin_top, margin_bottom = 130, 80, 110, 150
        plot_w = width - margin_left - margin_right
        plot_h = height - margin_top - margin_bottom
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)
        font = load_bold_font(26)
        value_font = load_bold_font(26)
        title_font = load_bold_font(42)

        draw.text((margin_left, 38), title, fill="#222222", font=title_font)
        for i in range(6):
            y_value = y_max * i / 5
            y = margin_top + plot_h - int((y_value / y_max) * plot_h)
            draw.line((margin_left, y, width - margin_right, y), fill="#E5E7EB", width=1)
            draw.text((40, y - 7), f"{y_value:.1f}{suffix}", fill="#555555", font=font)

        group_w = plot_w / max(len(categories), 1)
        bar_w = min(105, group_w * 0.38)
        for idx, (category, val) in enumerate(zip(categories, values)):
            center = margin_left + group_w * idx + group_w / 2
            x0 = center - bar_w / 2
            x1 = center + bar_w / 2
            y1 = margin_top + plot_h
            y0 = y1 - int((min(val, y_max) / y_max) * plot_h)
            draw.rectangle((x0, y0, x1, y1), fill=colors[idx % len(colors)])

            value_label = f"{val:.2f}{suffix}" if suffix else f"{val:.2f}"
            draw_centered_text(draw, center, y0 - 34, value_label, value_font, "#333333")
            draw_centered_text(draw, center, height - margin_bottom + 36, category, font, "#333333")

        path = os.path.join(CHART_DIR, filename)
        image.save(path)
        return path

    quality_categories = ["Correct", "Faithful", "Citation", "Hallucination"]
    quality_data = []
    for system in systems:
        quality_data.append(
            [
                round(value(["systems", system, "answer_correctness", "pass_rate_ge_2"]), 2),
                round(value(["systems", system, "faithfulness", "pass_rate_ge_2"]), 2),
                round(value(["systems", system, "citation_accuracy", "pass_rate_ge_2"]), 2),
                round(value(["systems", system, "hallucination_rate"]), 2),
            ]
        )
    chart_paths.append(
        draw_grouped_bar(
            "rag_eval_quality_comparison.png",
            "SUSTech RAG Answer Quality Comparison",
            quality_categories,
            systems,
            quality_data,
            1.0,
        )
    )

    if retrieval_systems:
        retrieval_categories = ["Recall@5", "Precision@5", "MRR@5", "nDCG@5"]
        retrieval_data = [
            [
                round(value(["retrieval", system, "recall@5"]), 2),
                round(value(["retrieval", system, "precision@5"]), 2),
                round(value(["retrieval", system, "mrr@5"]), 2),
                round(value(["retrieval", system, "ndcg@5"]), 2),
            ]
            for system in retrieval_systems
        ]
        chart_paths.append(
            draw_grouped_bar(
                "rag_eval_retrieval_comparison.png",
                "SUSTech RAG Retrieval Quality @5",
                retrieval_categories,
                retrieval_systems,
                retrieval_data,
                1.0,
            )
        )

    max_latency = max([value(["systems", system, "avg_total_latency_s"]) for system in systems] + [1.0])
    chart_paths.append(
        draw_simple_bar(
            "rag_eval_latency_comparison.png",
            "SUSTech RAG Average Latency",
            systems,
            [value(["systems", system, "avg_total_latency_s"]) for system in systems],
            max_latency * 1.15,
            "s",
        )
    )
    return chart_paths


def main() -> None:
    result_df, report = evaluate()
    report["charts"] = generate_charts(report)

    result_df.to_excel(OUTPUT_XLSX, index=False)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    log("\n" + "=" * 80)
    log("Evaluation summary")
    log("=" * 80)
    for system_name, stats in report["systems"].items():
        correctness = stats["answer_correctness"]["pass_rate_ge_2"]
        hallucination = stats["hallucination_rate"]
        latency = stats["avg_total_latency_s"]
        log(
            f"{system_name:<12} correctness_pass@>=2={correctness} "
            f"hallucination_rate={hallucination} avg_latency_s={latency}"
        )
    log("\nRetrieval summary")
    for system_name, stats in report["retrieval"].items():
        log(
            f"{system_name:<12} recall@5={stats.get('recall@5')} "
            f"precision@5={stats.get('precision@5')} "
            f"mrr@5={stats.get('mrr@5')} ndcg@5={stats.get('ndcg@5')}"
        )
    log(f"\n[INFO] Details saved to: {OUTPUT_XLSX}")
    log(f"[INFO] JSON report saved to: {OUTPUT_JSON}")
    if report.get("charts"):
        log("[INFO] Charts saved:")
        for path in report["charts"]:
            log(f"  - {path}")


if __name__ == "__main__":
    main()
