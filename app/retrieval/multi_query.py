from typing import List
from langchain_core.documents import Document

class MultiQueryGenerator:
    def __init__(self, llm):
        self.llm = llm

    def generate_queries(self, query: str) -> List[str]:
        prompt = f"""
You are an AI assistant.

Generate 3 different rephrased versions of the following query.
Each should capture a slightly different perspective.

Query: {query}

Return each query on a new line without numbering.
"""

        response = self.llm.invoke(prompt)

        raw_output = response.strip()

        # Split by new lines
        queries = raw_output.split("\n")

        # Clean results
        cleaned_queries = []
        for q in queries:
            q = q.strip().lstrip("-").strip()
            if q:
                cleaned_queries.append(q)

        # Fallback (important)
        if len(cleaned_queries) == 0:
            return [query]

        return cleaned_queries


class MultiQueryRetriever:
    def __init__(self, base_retriever, query_generator):
        self.base_retriever = base_retriever
        self.query_generator = query_generator

    def retrieve(self, query: str) -> List[Document]:
        queries = self.query_generator.generate_queries(query)

        # Always include original query (don’t trust LLM fully)
        if query not in queries:
            queries.append(query)

        all_docs = []

        for q in queries:
            docs = self.base_retriever.retrieve(q)
            all_docs.extend(docs)

        # Deduplicate based on content
        seen = set()
        unique_docs = []

        for doc in all_docs:
            content = doc.page_content
            if content not in seen:
                seen.add(content)
                unique_docs.append(doc)

        return unique_docs