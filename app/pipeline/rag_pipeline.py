# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS
# from langchain_huggingface import HuggingFaceEmbeddings
# from rank_bm25 import BM25Okapi
# from sentence_transformers import CrossEncoder

# from langchain_core.documents import Document

# import tempfile
# import numpy as np
# from groq import Groq
# import os

# from dotenv import load_dotenv
# load_dotenv()

# class RAGPipeline:
#     def __init__(self):
#         self.vectorstore = None
#         self.retriever = None
#         self.bm25 = None
#         self.documents = None
#         self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

#         self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

#     # -------------------------
#     # LOAD + PROCESS DOCUMENTS
#     # -------------------------
#     def load_documents(self, uploaded_files):
#         docs = []

#         for file in uploaded_files:
#             with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
#                 tmp.write(file.read())
#                 loader = PyPDFLoader(tmp.name)
#                 docs.extend(loader.load())

#         # Split
#         splitter = RecursiveCharacterTextSplitter(
#             chunk_size=500,
#             chunk_overlap=50
#         )
#         split_docs = splitter.split_documents(docs)

#         self.documents = split_docs

#         # -------------------------
#         # VECTOR STORE (FAISS)
#         # -------------------------
#         embeddings = HuggingFaceEmbeddings()
#         self.vectorstore = FAISS.from_documents(split_docs, embeddings)
#         self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})

#         # -------------------------
#         # BM25 (KEYWORD RETRIEVAL)
#         # -------------------------
#         tokenized_docs = [doc.page_content.split() for doc in split_docs]
#         self.bm25 = BM25Okapi(tokenized_docs)

#     # -------------------------
#     # HYBRID RETRIEVAL
#     # -------------------------
#     def hybrid_search(self, query, k=4):
#         if self.documents is None:
#             return []

#         # Dense retrieval
#         dense_docs = self.retriever.get_relevant_documents(query)

#         # BM25 retrieval
#         tokenized_query = query.split()
#         bm25_scores = self.bm25.get_scores(tokenized_query)

#         top_n = np.argsort(bm25_scores)[-k:]
#         bm25_docs = [self.documents[i] for i in top_n]

#         # Combine
#         combined = dense_docs + bm25_docs

#         # Remove duplicates
#         seen = set()
#         unique_docs = []
#         for doc in combined:
#             if doc.page_content not in seen:
#                 seen.add(doc.page_content)
#                 unique_docs.append(doc)

#         return unique_docs[:k]
    
#     def generate_queries(self, query):
#         prompt = f"""
#     Generate 4 different variations of the following question.
#     Each variation should capture a different way of asking the same thing.

#     Question: {query}

#     Output:
#     """

#         response = self.client.chat.completions.create(
#             model="llama-3.1-8b-instant",
#             messages=[{"role": "user", "content": prompt}],
#             temperature=0
#         )

#         queries = response.choices[0].message.content.split("\n")

#         # Clean
#         queries = [q.strip("- ").strip() for q in queries if q.strip()]

#         return list(set([query] + queries))
    
#     def rerank_documents(self, query, docs, top_k=3):
#         if not docs:
#             return []

#         pairs = [(query, doc.page_content) for doc in docs]

#         scores = self.reranker.predict(pairs)

#         scored_docs = list(zip(docs, scores))

#         # Sort by score (descending)
#         scored_docs.sort(key=lambda x: x[1], reverse=True)

#         top_docs = [doc for doc, score in scored_docs[:top_k]]

#         return top_docs

#     # -------------------------
#     # QUERY FUNCTION
#     # -------------------------
#     def run(self, query, chat_history =[]):
#         if self.vectorstore is None:
#             return {"answer": "⚠️ Upload documents first."}

#         all_queries = self.generate_queries(query)

#         all_docs = []

#         for q in all_queries:
#             docs = self.hybrid_search(q)
#             all_docs.extend(docs)

#         # Remove duplicates
#         seen = set()
#         unique_docs = []
#         for doc in all_docs:
#             if doc.page_content not in seen:
#                 seen.add(doc.page_content)
#                 unique_docs.append(doc)

#         docs = self.rerank_documents(query, unique_docs, top_k=3)

#         docs = self.rerank_documents(query, unique_docs, top_k=5)

#         # ✅ ADD HERE (exact location)
#         print("\n========== RETRIEVED CONTEXT ==========\n")
#         for i, doc in enumerate(docs):
#             print(f"\n--- DOC {i+1} ---\n")
#             print(doc.page_content[:500])

#         context = "\n\n".join([doc.page_content for doc in docs])

#         prompt = f"""
# You are a helpful AI assistant.

# Use the conversation history and the provided context to answer the question.

# Follow rules:
# - Use ONLY the context for factual answers
# - Use chat history for understanding follow-up questions
# - Do NOT hallucinate
# - If answer not in context → say "I don't know"

# Chat History:
# {chat_history}

# Context:
# {context}

# Question:
# {query}

# Answer:
# """

#         response = self.client.chat.completions.create(
#             model="llama-3.1-8b-instant",
#             messages=[{"role": "user", "content": prompt}],
#             temperature=0
#         )

#         answer = response.choices[0].message.content

#         return {"answer": answer}


from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from fastapi import UploadFile
import tempfile
import numpy as np
from groq import Groq
import os

from dotenv import load_dotenv
load_dotenv()


class RAGPipeline:
    def __init__(self):
        self.vectorstore = None
        self.retriever = None
        self.bm25 = None
        self.documents = None

        self.reranker = None    #LAZY LOADING THE MODELS
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    # -------------------------
    # LOAD DOCUMENTS
    # -------------------------
    def load_documents(self, uploaded_files):
        docs = []

        for file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file.read())   # Streamlit uses .read()
                loader = PyPDFLoader(tmp.name)
                docs.extend(loader.load())

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100
        )

        split_docs = splitter.split_documents(docs)
        self.documents = split_docs

        embeddings = HuggingFaceEmbeddings()
        self.vectorstore = FAISS.from_documents(split_docs, embeddings)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})

        tokenized_docs = [doc.page_content.split() for doc in split_docs]
        self.bm25 = BM25Okapi(tokenized_docs)

        
    def load_documents_api(self, files: list[UploadFile]):
        docs = []

        for file in files:
            content = file.file.read()

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(content)
                tmp_path = tmp.name

            loader = PyPDFLoader(tmp_path)
            docs.extend(loader.load())

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100
        )

        split_docs = splitter.split_documents(docs)
        self.documents = split_docs

        embeddings = HuggingFaceEmbeddings()
        self.vectorstore = FAISS.from_documents(split_docs, embeddings)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})

        tokenized_docs = [doc.page_content.split() for doc in split_docs]
        self.bm25 = BM25Okapi(tokenized_docs)

    # -------------------------
    # HYBRID SEARCH
    # -------------------------
    def hybrid_search(self, query, k=4):
        if self.documents is None:
            return []

        dense_docs = self.retriever.get_relevant_documents(query)

        tokenized_query = query.split()
        bm25_scores = self.bm25.get_scores(tokenized_query)

        top_n = np.argsort(bm25_scores)[-k:]
        bm25_docs = [self.documents[i] for i in top_n]

        combined = dense_docs + bm25_docs

        seen = set()
        unique_docs = []
        for doc in combined:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                unique_docs.append(doc)

        return unique_docs[:k]

    # -------------------------
    # QUERY REWRITING
    # -------------------------
    def generate_queries(self, query):
        prompt = f"""
Generate 4 different variations of this question:

{query}

Only return the questions.
"""

        response = self.client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        queries = response.choices[0].message.content.split("\n")
        queries = [q.strip("- ").strip() for q in queries if q.strip()]

        return list(set([query] + queries))

    # -------------------------
    # RERANKING
    # -------------------------
    def rerank_documents(self, query, docs, top_k=5):
        if self.reranker is None:
            self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        if not docs:
            return []

        pairs = [(query, doc.page_content) for doc in docs]
        scores = self.reranker.predict(pairs)

        scored_docs = list(zip(docs, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        return [doc for doc, _ in scored_docs[:top_k]]

    # -------------------------
    # MAIN RUN
    # -------------------------
    def run(self, query, chat_history=[]):
        if self.vectorstore is None:
            return {"answer": "⚠️ Upload documents first."}

        # -------------------------
        # MULTI-QUERY
        # -------------------------
        all_queries = self.generate_queries(query)

        # -------------------------
        # RETRIEVAL
        # -------------------------
        all_docs = []
        for q in all_queries:
            all_docs.extend(self.hybrid_search(q))

        # -------------------------
        # DEDUP DOCUMENTS
        # -------------------------
        seen = set()
        unique_docs = []
        for doc in all_docs:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                unique_docs.append(doc)

        # -------------------------
        # RERANK (IMPORTANT: uses documents)
        # -------------------------
        docs = self.rerank_documents(query, unique_docs, top_k=5)

        # -------------------------
        # SOURCE SNIPPETS (AFTER docs exist)
        # -------------------------
        source_snippets = []

        for doc in docs:
            page = doc.metadata.get("page", None)
            text = doc.page_content.strip().replace("\n", " ")

            snippet = text[:150]

            if page is not None:
                source_snippets.append((page + 1, snippet))

        # Deduplicate sources
        seen = set()
        unique_sources = []

        for page, snippet in source_snippets:
            key = (page, snippet[:50])
            if key not in seen:
                seen.add(key)
                unique_sources.append((page, snippet))

        # -------------------------
        # DEBUG
        # -------------------------
        print("\n========== RETRIEVED CONTEXT ==========\n")
        for i, doc in enumerate(docs):
            print(f"\n--- DOC {i+1} ---\n")
            print(doc.page_content[:500])

        # -------------------------
        # CONTEXT
        # -------------------------
        context = "\n\n".join([doc.page_content for doc in docs])

        user_prompt = f"""
    Context:
    {context}

    Question:
    {query}
    """

        system_prompt = """
    You are a helpful AI assistant.

    Your job is to explain answers clearly and naturally to a human.

    Rules:
    - Use ONLY the provided context
    - Do NOT copy sentences directly
    - Combine information from multiple parts of the context
    - Explain in a simple, flowing, conversational way
    - Avoid bullet points unless necessary
    - If answer not found → say "I don't know"
    """

        messages = [{"role": "system", "content": system_prompt}]

        for msg in chat_history:
            messages.append(msg)

        messages.append({"role": "user", "content": user_prompt})

        response = self.client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=0.2
        )

        answer = response.choices[0].message.content

        # -------------------------
        # FORMAT SOURCES
        # -------------------------
        formatted_sources = "\n".join(
            [f"- Page {p}: \"{s}...\"" for p, s in unique_sources]
        )

        final_answer = f"""{answer}

    ---

    📌 Sources:
    {formatted_sources}
    """

        return {"answer": final_answer}
    
# =========================
# GLOBAL PIPELINE INSTANCE
# =========================
pipeline = RAGPipeline()


def run_rag(query: str):
    return pipeline.run(query)