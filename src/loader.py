from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_split_pdf(file_path):
    """
    Loads a PDF file and splits it into small overlapping text chunks.
    This helps the RAG model retrieve accurate context.
    """
    # Load the PDF
    loader = PyPDFLoader(file_path)
    documents = loader.load()   # Extract text from PDF pages

    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,      # each chunk is ~800 characters
        chunk_overlap=200    # overlaps for better context
    )

    chunks = splitter.split_documents(documents)
    return chunks