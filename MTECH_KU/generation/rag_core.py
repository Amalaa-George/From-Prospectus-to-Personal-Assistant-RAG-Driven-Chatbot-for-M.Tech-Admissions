from generation.prompt_utils import build_prompt
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def answer_query(user_query: str, embedding_model, vectorstore, llm) -> str:
    """
    Generates an answer to the user query using provided components.

    Parameters:
        user_query (str): The input query from the user.
        embedding_model: Embedding model instance with embed_query().
        vectorstore: Vector store retriever with retrieve_documents().
        llm: LLM model with generate() method.

    Returns:
        str: Final generated answer.
    """

    # Early validation: Ensure user query is not empty or only whitespace
    if user_query is None or not user_query.strip():
        logger.warning("Empty or whitespace-only query received.")
        return "Please enter a valid question. The query cannot be empty or just spaces."

    # Validate dependencies
    if not all([embedding_model, vectorstore, llm]):
        logger.error("One or more RAG components are missing (embedding model, vectorstore, or LLM).")
        return "Internal error: Missing RAG components."

    # Generate query embedding
    try:
        logger.info("Embedding user query...")
        query_embedding = embedding_model.embed_query(user_query)
        logger.info("Query embedding generated.")
    except Exception as e:
        logger.exception("Error embedding query.")
        return "Failed to process your question due to an embedding error."

    # Retrieve documents
    try:
        logger.info("Retrieving documents from vector store...")
        docs = vectorstore.retrieve_documents(query_embedding)
        logger.info(f"Retrieved {len(docs)} documents.")
    except Exception as e:
        logger.exception("Error retrieving documents from vectorstore.")
        return "Sorry, I couldn't access the knowledge base at the moment."

    if not docs:
        logger.warning("No documents found for the query.")
        return "Sorry, I couldn't find any relevant information."

    # Clean and deduplicate lines from retrieved documents while preserving order
    unique_lines = list(dict.fromkeys(
        line.strip() for doc in docs for line in doc.splitlines() if line.strip()
    ))
    context = "\n".join(unique_lines)

    # Generate prompt and get answer
    try:
        logger.info("Building prompt and generating response from LLM...")
        prompt = build_prompt(context, user_query)
        raw_answer = llm.generate(prompt)
        logger.info("LLM response generated successfully.")
        return raw_answer.strip()
    except AttributeError:
        logger.exception("LLM object is missing the required 'generate()' method.")
        return "Internal error: LLM configuration is invalid."
    except Exception as e:
        logger.exception("Error during LLM response generation.")
        return "Sorry, something went wrong while generating the response. Please try again later."