from langchain_community.vectorstores import Chroma
import os

def build_vector_store(chunks, embedding_model, persist_dir="chroma_store"):
    """
    Creates a Chroma vector store from the document chunks and saves it to disk.
    """
    # Make sure the folder for Chroma exists
    os.makedirs(persist_dir, exist_ok=True)

    # Build the vector store from the chunks
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_dir
    )

    # Save to disk so we can reuse it later
    vector_store.persist()
    return vector_store


def load_vector_store(embedding_model, persist_dir="chroma_store"):
    """
    Loads an existing Chroma vector store from disk.
    """
    if not os.path.isdir(persist_dir):
        raise ValueError(
            f"Vector store not found at '{persist_dir}'. Build it first."
        )

    vector_store = Chroma(
        embedding_function=embedding_model,
        persist_directory=persist_dir
    )

    return vector_store