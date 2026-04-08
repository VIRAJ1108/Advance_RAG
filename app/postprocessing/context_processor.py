from typing import List
from langchain_core.documents import Document

class ContextProcessor:
    def __init__(self, all_chunks: List[Document]):
        self.all_chunks = all_chunks

    def is_noise(self, text: str) -> bool:
        text_lower = text.lower()

        # simple heuristics (we'll improve later)
        if "references" in text_lower:
            return True
        if text_lower.strip().startswith("["):
            return True
        if "arxiv" in text_lower:
            return True

        return False

    def expand_context(self, doc: Document, window: int = 1) -> List[Document]:
        chunk_id = doc.metadata.get("chunk_id", None)

        if chunk_id is None:
            return [doc]

        start = max(0, chunk_id - window)
        end = min(len(self.all_chunks), chunk_id + window + 1)

        return self.all_chunks[start:end]

    def process(self, documents: List[Document]) -> List[Document]:
        final_docs = []
        seen = set()

        for doc in documents:
            if self.is_noise(doc.page_content):
                continue

            expanded_docs = self.expand_context(doc)

            for d in expanded_docs:
                content = d.page_content
                if content not in seen:
                    seen.add(content)
                    final_docs.append(d)

        return final_docs