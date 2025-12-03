from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


def build_rag_pipeline(vector_store):
    """
    Builds a simple RAG pipeline without using RetrievalQA classes.

    It returns a function `answer_question(question: str) -> str`
    that:
      1. Retrieves relevant chunks from the vector store
      2. Sends them + the question to the LLM
      3. Returns the answer text
    """

    # Load environment variables (OPENAI_API_KEY)
    load_dotenv()

    # LLM: this is the brain that writes answers
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.0  # 0 = more deterministic, less random
    )

    # Convert the Chroma vector store into a retriever
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},  # top 4 most relevant chunks
    )

    def answer_question(question: str) -> str:
        """
        Given a user question, retrieve context and ask the LLM
        to answer ONLY based on that context.
        """
        # 1. Get most relevant documents/chunks
        docs = retriever.get_relevant_documents(question)

        if not docs:
            return "I couldn't find any relevant information in the document."

        # 2. Build a context string from the chunks
        context = "\n\n".join(d.page_content for d in docs)

        # 3. Create a prompt for the LLM
        prompt = (
            "You are a helpful assistant that answers questions "
            "based ONLY on the provided context. "
            "If the answer is not in the context, say "
            "'I don't know based on this document.'\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )

        # 4. Ask the LLM
        response = llm.predict(prompt)
        return response

    # Return the function so other code can call it
    return answer_question