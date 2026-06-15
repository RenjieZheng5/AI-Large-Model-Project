# SUSTech Campus Knowledge RAG System

本项目实现了一个面向南方科技大学校园知识的 Retrieval-Augmented
Generation 问答系统。系统从校园网页和文档中构建 QA 知识库，使用
Qwen3-Embedding 和 FAISS 建立向量索引，并通过 OpenAI 兼容接口调用
Qwen3-32B 完成回答生成。项目同时包含 Gradio 演示界面、批量评测脚本、
延迟诊断实验和论文报告。
## 目录结构

```text
.
|-- README.md
|-- src/
|   |-- rag_config.py
|   |-- rag_chat.py
|   |-- reranker.py
|   `-- __init__.py
|-- scripts/
|   |-- app_gradio.py
|   |-- build_qa_index.py
|   |-- search_qa.py
|   |-- rag_chat_transformers.py
|   |-- crawling/
|   |   |-- ai_crawler_qa.py
|   |   `-- ai_crawler_qa_enhanced.py
|   `-- evaluation/
|       |-- batch_eval_full.py
|       |-- batch_eval_tag_template.py
|       `-- latency_diagnostic_experiment.py
|-- data/
|   |-- sustech_raw_documents.jsonl
|   |-- sustech_qa_pairs.jsonl
|   |-- sustech_rag_test_questions.xlsx
|   |-- data_processed.md
|   `-- rag_index_sustech_qwen3_embedding/
|       |-- index.faiss
|       |-- items.jsonl
|       `-- meta.json
|-- results/
|   |-- sustech_rag_batch_results_full.xlsx
|   |-- sustech_rag_batch_results_full.json
|   |-- latency_diagnostic_results.csv
|   |-- latency_diagnostic_results.json
|   `-- charts/
|       |-- rag_eval_quality_comparison.png
|       |-- rag_eval_retrieval_comparison.png
|       |-- rag_eval_latency_comparison.png
|       |-- rag_latency_diagnostic_breakdown.png
|       `-- rag_latency_vs_output_tokens.png
|-- report/
|   |-- example_paper.tex
|   |-- Large_AI_Model_RAG_Report.pdf
|   |-- report_overleaf.zip
|   `-- figures/
`-- docs/
    |-- RAG.pptx
    `-- projects.pdf
```

目录职责：

- `src/`：RAG 系统核心代码，包括配置、检索、上下文构造、LLM 调用和重排器。
- `scripts/`：可执行脚本，包括 Web UI、索引构建、检索调试、数据爬取和实验评测。
- `data/`：原始数据、QA 知识库、测试题集和 FAISS 索引。
- `results/`：实验结果表格、JSON 报告和评测图表。
- `report/`：论文 LaTeX 源文件、PDF 成稿和报告图片。
- `docs/`：课程说明、汇报 PPT 等辅助材料。

## 依赖

需要准备 Python 环境，并安装以下主要依赖：

```text
torch
transformers
sentence-transformers
faiss-cpu 或 faiss-gpu
vllm
pandas
openpyxl
requests
gradio
matplotlib
numpy
tqdm
beautifulsoup4
pymupdf
python-docx
ddgs
```

还需要准备以下本地模型，路径通过环境变量传入：

```bash
export EMBED_MODEL_PATH=/path/to/Qwen3-Embedding-0.6B
export RERANK_MODEL_PATH=/path/to/bge-reranker-v2-m3
export VLLM_MODEL=qwen3-32b
export VLLM_URL=http://127.0.0.1:8000/v1/chat/completions
```

如果只查看已有结果和报告，不需要重新运行模型服务。

## 基本配置

所有命令默认从项目根目录运行：

```bash
cd /path/to/AI-Large-Model-Project
```

常用环境变量：

```bash
export DATA_PATH="$PWD/data/sustech_qa_pairs.jsonl"
export INDEX_DIR="$PWD/data/rag_index_sustech_qwen3_embedding"

export TOP_K=30
export USE_RERANKER=1
export RERANK_TOP_N=5

export EMBED_DEVICE=cpu
export RERANK_DEVICE=cpu
```

`src/rag_config.py` 已经为 `DATA_PATH` 和 `INDEX_DIR` 设置了项目内默认值；
如果数据和索引保持在 `data/` 下，通常不需要手动设置这两个变量。

## 启动问答界面

先确保已经有一个 OpenAI 兼容的聊天补全服务，并且 `VLLM_URL` 指向该服务。
然后运行：

```bash
python scripts/app_gradio.py
```

默认访问地址：

```text
http://127.0.0.1:7860
```

界面会使用 `data/rag_index_sustech_qwen3_embedding/` 中的 FAISS 索引进行检索，
再调用 `VLLM_URL` 配置的模型接口生成答案。

## 重建 FAISS 索引

当 `data/sustech_qa_pairs.jsonl` 更新后，需要重新构建索引：

```bash
python scripts/build_qa_index.py
```

输出文件位于：

```text
data/rag_index_sustech_qwen3_embedding/index.faiss
data/rag_index_sustech_qwen3_embedding/items.jsonl
data/rag_index_sustech_qwen3_embedding/meta.json
```

## 命令行检索调试

如果只想检查检索结果，不调用完整 UI，可以运行：

```bash
python scripts/search_qa.py
```

该脚本会加载 FAISS 索引，并输出相似问题、来源和可选重排后的结果。

## 数据采集脚本

基础爬虫：

```bash
python scripts/crawling/ai_crawler_qa.py
```

增强版爬虫：

```bash
python scripts/crawling/ai_crawler_qa_enhanced.py
```

默认输出到：

```text
data/sustech_qa_pairs.jsonl
data/sustech_raw_documents.jsonl
data/sustech_failed_urls.jsonl
```

运行爬虫前需要配置用于生成 QA 的外部 API 密钥，例如：

```bash
export DEEPSEEK_API_KEY=your_api_key
```

## 完整评测

完整评测比较四种系统：

- `Direct`：不检索，直接让 LLM 回答。
- `Basic_RAG`：FAISS 检索后生成。
- `RAG_Rerank`：FAISS 检索、BGE 重排后生成。
- `RAG_Rewrite_Rerank`：查询改写、FAISS 检索、BGE 重排后生成。

运行：

```bash
export RUN_LLM_JUDGE=1
export ENABLE_QUERY_REWRITE=1

python scripts/evaluation/batch_eval_full.py
```

输入文件：

```text
data/sustech_rag_test_questions.xlsx
```

输出文件：

```text
results/sustech_rag_batch_results_full.xlsx
results/sustech_rag_batch_results_full.json
results/sustech_rag_batch_results_full.partial.xlsx
results/charts/rag_eval_quality_comparison.png
results/charts/rag_eval_retrieval_comparison.png
results/charts/rag_eval_latency_comparison.png
```

当前完整评测结果摘要：

```text
Direct correctness: 36.67%
Basic RAG correctness: 80.00%
RAG Rerank correctness: 86.67%
RAG Rewrite Rerank correctness: 86.67%

Direct hallucination rate: 66.67%
RAG hallucination rate: 0.00%
```

## 延迟诊断实验

该实验用于解释为什么主评测中 `Basic_RAG` 的平均响应时间可能低于
`Direct`：虽然 RAG 增加了检索耗时，但检索证据会显著缩短模型生成答案的长度，
从而减少生成阶段耗时。

运行：

```bash
python scripts/evaluation/latency_diagnostic_experiment.py --repetitions 1 --max-tokens 512
```

输出文件：

```text
results/latency_diagnostic_results.csv
results/latency_diagnostic_results.json
results/charts/rag_latency_diagnostic_breakdown.png
results/charts/rag_latency_vs_output_tokens.png
```

当前诊断结果摘要：

```text
Direct:
  average total latency: 8.757s
  average prompt tokens: 51.80
  average completion tokens: 180.53
  average generation throughput: 20.58 tokens/s

Basic RAG:
  average total latency: 3.583s
  average retrieval latency: 0.741s
  average generation latency: 2.842s
  average prompt tokens: 580.73
  average completion tokens: 56.13
  average generation throughput: 18.89 tokens/s
```

结论：`Basic_RAG` 平均更快的主要原因不是检索本身加速，而是检索证据和约束性
prompt 让模型输出更短。实验中 RAG 增加了约 `0.741s` 检索开销，但平均输出从
`180.53` tokens 降到 `56.13` tokens，节省的生成时间超过了检索开销。

## 报告

论文源文件：

```text
report/example_paper.tex
```

已编译 PDF：

```text
report/Large_AI_Model_RAG_Report.pdf
```

报告图片：

```text
report/figures/
```

Overleaf 压缩包：

```text
report/report_overleaf.zip
```