from typing import List
from langchain_core.documents import Document


class HybridRetriever:
    def __init__(self, faiss_retriever, bm25_retriever):
        self.faiss_retriever = faiss_retriever
        self.bm25_retriever = bm25_retriever

    def retrieve(self, query: str) -> List[Document]:
        semantic_results = self.faiss_retriever.retrieve(query)
        keyword_results = self.bm25_retriever.retrieve(query)

        # simple merge (we’ll improve later)
        combined = semantic_results + keyword_results

        # remove duplicates
        seen = set()
        unique_docs = []

        for doc in combined:
            content = doc.page_content
            if content not in seen:
                seen.add(content)
                unique_docs.append(doc)

        return unique_docs