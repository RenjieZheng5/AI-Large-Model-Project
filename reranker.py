from typing import List, Dict, Any
from sentence_transformers import CrossEncoder


class BGEReranker:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        batch_size: int = 8
    ):
        self.model_path = model_path
        self.device = device
        self.batch_size = batch_size

        print(f"[INFO] Loading BGE reranker from: {model_path}", flush=True)
        self.model = CrossEncoder(
            model_path,
            device=device
        )

    def build_doc_text(self, item: Dict[str, Any]) -> str:
        """
        把候选 QA 拼成 reranker 判断相关性的文本。
        """
        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()
        index_text = str(item.get("index_text", "")).strip()
        raw = item.get("raw", {})

        category = ""
        source_url = ""
        source_title = ""

        if isinstance(raw, dict):
            category = str(raw.get("category", "")).strip()
            source_url = str(raw.get("source_url", "")).strip()
            source_title = str(raw.get("source_title", "")).strip()

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

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_n: int = 5
    ) -> List[Dict[str, Any]]:
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

        for old_rank, (item, score) in enumerate(zip(candidates, scores), start=1):
            new_item = dict(item)
            new_item["faiss_rank"] = item.get("rank", old_rank)
            new_item["faiss_score"] = item.get("score", 0.0)
            new_item["rerank_score"] = float(score)
            reranked.append(new_item)

        reranked.sort(
            key=lambda x: x["rerank_score"],
            reverse=True
        )

        for new_rank, item in enumerate(reranked[:top_n], start=1):
            item["rank"] = new_rank

        return reranked[:top_n]
