from typing import List
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, documents: List[Document], top_k: int = 5) -> List[Document]:
        if not documents:
            return []

        pairs = [(query, doc.page_content) for doc in documents]

        scores = self.model.predict(pairs)

        # attach scores
        doc_scores = list(zip(documents, scores))

        # sort by score (descending)
        ranked = sorted(doc_scores, key=lambda x: x[1], reverse=True)

        # return top-k documents only
        return [doc for doc, score in ranked[:top_k]]