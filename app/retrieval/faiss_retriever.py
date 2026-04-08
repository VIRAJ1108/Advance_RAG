from typing import List
from langchain_core.documents import Document


class FAISSRetriever:
    def __init__(self, vectorstore, k: int = 5):
        self.vectorstore = vectorstore
        self.k = k

    def retrieve(self, query: str) -> List[Document]:
        results = self.vectorstore.similarity_search(query, k=self.k)
        return results