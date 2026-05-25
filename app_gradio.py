import os
import gradio as gr

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

from rag_chat import (
    Qwen3Embedder,
    FaissRetriever,
    build_context,
    call_vllm,
)

from reranker import BGEReranker


# =========================
# 初始化模型和检索器
# =========================

print("[INFO] Initializing RAG system...")

embedder = Qwen3Embedder(
    model_path=EMBED_MODEL_PATH,
    device=EMBED_DEVICE,
    max_length=MAX_EMBED_LEN,
)

retriever = FaissRetriever(
    index_dir=INDEX_DIR,
    embedder=embedder,
)

reranker = None

RERANK_DEVICE = os.environ.get("RERANK_DEVICE", "cpu")

if USE_RERANKER:
    reranker = BGEReranker(
        model_path=RERANK_MODEL_PATH,
        device=RERANK_DEVICE,
        batch_size=RERANK_BATCH_SIZE,
    )

print("[INFO] RAG system ready.")


# =========================
# RAG 问答函数
# =========================

def rag_answer(message, history):
    message = message.strip()

    if not message:
        return "请输入问题。"

    try:
        # 1. FAISS 初筛
        faiss_results = retriever.search(
            message,
            top_k=TOP_K,
        )

        if not faiss_results:
            return "没有检索到相关资料。"

        # 2. BGE Reranker 重排
        if reranker is not None:
            final_results = reranker.rerank(
                query=message,
                candidates=faiss_results,
                top_n=RERANK_TOP_N,
            )
        else:
            final_results = faiss_results[:RERANK_TOP_N]

        # 3. 构造 context
        context = build_context(final_results)

        # 4. 调用 vLLM
        answer = call_vllm(message, context)

        # 5. 附加引用资料，方便展示
        sources = []
        for r in final_results:
            q = str(r.get("question", "")).strip()
            a = str(r.get("answer", "")).strip()
            faiss_score = r.get("faiss_score", r.get("score", 0.0))
            rerank_score = r.get("rerank_score", None)

            if len(q) > 80:
                q = q[:80] + "..."

            if rerank_score is not None:
                sources.append(
                    f"- [{r.get('rank')}] {q}\n"
                    f"  - FAISS: {faiss_score:.4f}, Rerank: {rerank_score:.4f}"
                )
            else:
                sources.append(
                    f"- [{r.get('rank')}] {q}\n"
                    f"  - FAISS: {faiss_score:.4f}"
                )

        source_text = "\n".join(sources)

        return f"{answer}\n\n---\n\n**参考资料 Top {len(final_results)}：**\n{source_text}"

    except Exception as e:
        return f"运行出错：{repr(e)}"


# =========================
# Gradio UI
# =========================

demo = gr.ChatInterface(
    fn=rag_answer,
    title="南方科技大学校园知识库 RAG 问答系统",
    description=(
        "本系统基于 QA 知识库、Qwen3 Embedding、FAISS、BGE Reranker 和 Qwen3-32B 构建。"
        "请输入关于校园卡、宿舍、选课、图书馆、学生事务等问题。"
    ),
    examples=[
        "南科大校园卡怎么补办？",
        "南科大图书馆怎么借书？",
        "南科大选课有什么规定？",
        "学生事务可以办理哪些事项？",
    ],
)

if __name__ == "__main__":
    demo.queue(
        default_concurrency_limit=2
    ).launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )