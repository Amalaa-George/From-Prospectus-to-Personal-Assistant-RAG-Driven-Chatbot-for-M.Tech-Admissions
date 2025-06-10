"""
This module provides a wrapper for generating text embeddings using Jina AI's embedding models.

It defines the EmbeddingModel class, which initializes a JinaEmbeddings model using an API key
and exposes a method to embed input queries into numerical vectors.
The embeddings generated can be used in downstream tasks such as semantic search, clustering,
or feeding into a retrieval-augmented generation (RAG) pipeline.
"""

import logging
from langchain_community.embeddings import JinaEmbeddings
from config import JINA_API_KEY, EMBEDDING_MODEL_NAME  # Allow model name config

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class EmbeddingModel:
    """
    Initializes the JinaEmbeddings model using a configured API key and model name,
    and generates vector embeddings for input queries.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        if not JINA_API_KEY or not JINA_API_KEY.strip():
            logger.error("JINA_API_KEY is missing or invalid.")
            raise ValueError("JINA_API_KEY is missing. Please set it in your .env or config.py.")

        try:
            logger.info(f"Initializing JinaEmbeddings with model: {model_name}")
            self.model = JinaEmbeddings(
                jina_api_key=JINA_API_KEY,
                model_name=model_name
            )
            logger.info("JinaEmbeddings initialized successfully.")
        except Exception as e:
            logger.exception(f"Failed to initialize JinaEmbeddings: {e}")
            raise

    def embed_query(self, text: str) -> list:
        """
        Generates an embedding for the input text using the Jina model.

        Args:
            text (str): The input string to embed.

        Returns:
            list: A list of floats representing the text embedding.
        """
        if not text or not text.strip():
            logger.warning("Empty or whitespace-only text received for embedding.")
            return []

        try:
            embedding = self.model.embed_query(text)
            logger.info("Query embedded successfully.")
            return embedding
        except Exception as e:
            logger.exception(f"Embedding failed for input: '{text[:30]}...' | Error: {e}")
            return []
