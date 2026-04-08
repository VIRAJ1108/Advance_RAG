from typing import List
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi


class BM25Retriever:
    def __init__(self, documents: List[Document], k: int = 5):
        self.k = k
        self.documents = documents

        # tokenize documents
        self.tokenized_docs = [doc.page_content.split() for doc in documents]

        # initialize BM25
        self.bm25 = BM25Okapi(self.tokenized_docs)

    def retrieve(self, query: str) -> List[Document]:
        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)

        # get top-k indices
        top_k_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:self.k]

        return [self.documents[i] for i in top_k_indices]