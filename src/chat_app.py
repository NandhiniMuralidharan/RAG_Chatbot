import os
import streamlit as st

from loader import load_and_split_pdf
from embedder import get_embedding_model
from vectorstore import build_vector_store, load_vector_store
from rag_engine import build_rag_pipeline


# Make sure the data folder exists for saving uploaded files
os.makedirs("data", exist_ok=True)

st.set_page_config(page_title="RAG PDF Chatbot", page_icon="📘")
st.title("📘 RAG Chatbot - Ask Questions About Your PDF")


st.write("Upload a PDF, and then ask questions about its content. "
         "The app will use RAG (Retrieval-Augmented Generation) to answer based on the document.")


# --- File upload section ---
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    file_path = os.path.join("data", "temp.pdf")

    # Save the uploaded PDF to disk
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    st.info("✅ PDF uploaded. Processing...")

    # Load and split the PDF into chunks
    chunks = load_and_split_pdf(file_path)

    # Get embedding model
    embedding_model = get_embedding_model()

    # Build or rebuild vector store
    vector_store = build_vector_store(chunks, embedding_model)

    # Build RAG pipeline
    rag_chain = build_rag_pipeline(vector_store)

    st.success("🎉 Your RAG chatbot is ready! Ask a question below.")

    # --- Question input ---
    user_question = st.text_input("Type your question about the PDF:")

    if user_question:
        with st.spinner("Thinking..."):
            answer = rag_chain.run(user_question)

        st.write("### Answer:")
        st.write(answer)
else:
    st.warning("Please upload a PDF to get started.")