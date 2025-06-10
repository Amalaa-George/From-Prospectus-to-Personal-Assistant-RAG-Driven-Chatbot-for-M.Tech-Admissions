import logging
import chromadb
from config import CHROMA_DIR, COLLECTION_NAME, TOP_K

logger = logging.getLogger(__name__)

class ChromaRetriever:
    def __init__(self, collection_name: str = COLLECTION_NAME):
        if not CHROMA_DIR:
            raise ValueError("CHROMA_DIR is not set in config.")
        if not collection_name:
            raise ValueError("COLLECTION_NAME is not set in config.")
        
        self.client = chromadb.PersistentClient(path=CHROMA_DIR)
        try:
            self.collection = self.client.get_collection(name=collection_name)
        except Exception as e:
            logger.error(f"Error loading ChromaDB collection '{collection_name}': {e}")
            raise RuntimeError(f"Failed to load ChromaDB collection: {e}")

    def retrieve_documents(self, query_embedding: list, top_k: int = TOP_K) -> list:
        """
        Retrieve the top_k most similar documents for a given query embedding.

        Args:
            query_embedding (list): A list of floats representing the query embedding.
            top_k (int): Number of top documents to retrieve.

        Returns:
            list: A list of document strings, or an empty list if none found.
        """
        try:
            results = self.collection.query(query_embeddings=[query_embedding], n_results=top_k)
            documents = results.get("documents")
            if documents and isinstance(documents, list) and len(documents) > 0:
                return documents[0]
            else:
                logger.warning("No documents returned from query.")
                return []
        except Exception as e:
            logger.error(f"Error during document retrieval: {e}")
            return []