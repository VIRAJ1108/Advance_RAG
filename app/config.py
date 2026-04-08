import os
from dotenv import load_dotenv

load_dotenv()

DATA_PATH = "data/Attention is all you need.pdf"
FAISS_INDEX_PATH = "faiss_index"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100