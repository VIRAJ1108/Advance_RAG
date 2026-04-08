from langchain_community.vectorstores import FAISS

def create_faiss_index(documents, embedding_model):
    vectorstore = FAISS.from_documents(documents, embedding_model)
    return vectorstore