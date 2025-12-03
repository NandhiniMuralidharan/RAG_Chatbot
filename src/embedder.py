from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

def get_embedding_model():
    """
    Loads environment variables and returns an OpenAI embedding model.
    This model converts text into numeric vectors for similarity search.
    """
    # Load variables from .env (including OPENAI_API_KEY)
    load_dotenv()

    # Create the embedding model (uses OPENAI_API_KEY from environment)
    embedding_model = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    return embedding_model