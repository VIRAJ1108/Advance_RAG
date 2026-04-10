import streamlit as st
from app.pipeline.rag_pipeline import RAGPipeline

@st.cache_resource
def load_pipeline():
    return RAGPipeline()

st.set_page_config(page_title="Advanced RAG", layout="wide")

st.title("📄 Advanced RAG Chatbot")

# -------------------------
# INIT PIPELINE
# -------------------------
if "pipeline" not in st.session_state:
    st.session_state.pipeline = load_pipeline()

if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------------
# FILE UPLOAD
# -------------------------
st.sidebar.header("Upload PDFs")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    with st.spinner("Processing documents..."):
        st.session_state.pipeline.load_documents(uploaded_files)

    st.sidebar.success("Documents processed ✅")

# -------------------------
# CHAT DISPLAY
# -------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -------------------------
# INPUT
# -------------------------
query = st.chat_input("Ask something...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    try:
        response = st.session_state.pipeline.run(query)
        answer = response["answer"]
    except Exception as e:
        answer = f"Error: {str(e)}"

    st.session_state.messages.append({"role": "assistant", "content": answer})

    with st.chat_message("assistant"):
        st.markdown(answer)

