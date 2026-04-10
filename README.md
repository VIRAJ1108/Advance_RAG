# HybridRAG - Advanced RAG System for Context-Aware Question Answering

## Overview

 HybridRAG is an advanced Retrieval-Augmented Generation (RAG) system designed to provide accurate, context-grounded answers from user-provided documents. The system focuses on improving retrieval quality, reducing hallucinations, and enabling conversational interactions with source-backed responses.

---

## Key Features

- 🔍 **Multi-Query Retrieval** — Expands user queries into multiple variations to improve recall
- ⚖️ **Hybrid Search (FAISS + BM25)** — Combines semantic and keyword-based retrieval
- 🧠 **Cross-Encoder Reranking** — Improves precision by selecting the most relevant chunks
- 💬 **Conversational Memory** — Supports follow-up questions using chat history
- 📌 **Source Attribution** — Displays page-level citations with supporting snippets
- ⚡ **Groq LLM Integration** — Fast inference using LLaMA-based models
- 🖥️ **Streamlit UI** — Interactive interface for document upload and Q&A

---

## System Architecture
User Query
↓
Multi-Query Generation (LLM)
↓
Hybrid Retrieval (FAISS + BM25)
↓
Deduplication
↓
Reranking (CrossEncoder)
↓
Context Construction
↓
LLM Response (Groq)
↓
Answer + Source Snippets

## Tech Stack

- **LLM**: Groq (LLaMA-based models)
- **Embeddings**: HuggingFace Sentence Transformers
- **Vector Store**: FAISS
- **Keyword Search**: BM25 (rank-bm25)
- **Reranking**: Cross-Encoder (ms-marco-MiniLM-L-6-v2)
- **Framework**: LangChain (modular components)
- **Frontend**: Streamlit

## Why This Design?

### 1. Hybrid Retrieval (FAISS + BM25)
Semantic search alone misses keyword-heavy queries, while BM25 captures lexical matches. Combining both improves recall.

### 2. Multi-Query Expansion
Single queries often fail to retrieve all relevant information. Query rewriting increases coverage and improves retrieval quality.

### 3. Reranking
Initial retrieval returns noisy results. Cross-encoder reranking improves precision by scoring relevance more accurately.

### 4. Source Attribution
To reduce hallucination and improve trust, answers are backed with page-level references and snippets.

---

## Challenges & Solutions

### Problem: Generic / Hallucinated Responses
- **Cause**: LLM ignored retrieved context
- **Solution**:
  - Improved prompt design (strict grounding)
  - Increased chunk size for better semantic coherence
  - Added reranking for higher-quality context

---

### Problem: Poor Retrieval Quality
- **Cause**: Single-query retrieval missing relevant chunks
- **Solution**:
  - Multi-query generation
  - Hybrid retrieval (semantic + keyword)

---

### Problem: No Context Awareness in Conversations
- **Cause**: Stateless queries
- **Solution**:
  - Integrated chat history into LLM input

---

## How It Works

1. Upload PDF documents
2. System splits text into chunks
3. Generates embeddings and builds FAISS index
4. Converts user query into multiple variations
5. Retrieves relevant chunks using hybrid search
6. Reranks results using cross-encoder
7. Generates answer using LLM
8. Displays answer with source snippets

---

## Installation

```bash
git clone <your-repo-link>
cd Advance_rag
pip install -r requirements.txt

