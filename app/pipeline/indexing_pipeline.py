from app.ingestion.loaders import load_pdf
from app.ingestion.splitter import split_documents
from app.embeddings.embedder import get_embedding_model
from app.vectorstore.faiss_store import create_faiss_index

def run_indexing_pipeline(pdf_path: str):
    print("Loading documents...")
    docs = load_pdf(pdf_path)

    print("Splitting documents...")
    chunks = split_documents(docs)

    print("Loading embedding model...")
    embedding_model = get_embedding_model()

    print("Creating FAISS index...")
    vectorstore = create_faiss_index(chunks, embedding_model)

    return vectorstore, chunks