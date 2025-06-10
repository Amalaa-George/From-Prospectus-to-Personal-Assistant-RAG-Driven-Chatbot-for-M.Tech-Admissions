import sys
import os

# Dynamically add the project root (1 levels up from test_vectorestore.py) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
import pytest
from retrieval.chroma_vectorstore import ChromaRetriever
from unittest.mock import MagicMock, patch

# Sample mock embedding (float values for test)
DUMMY_EMBEDDING = [0.1] * 1024  # Ensure this matches your embedding dimension
DUMMY_DOCUMENTS = [["Doc1", "Doc2"]]

@patch("retrieval.chroma_vectorstore.chromadb.PersistentClient")
def test_retrieve_documents_success(mock_client):
    mock_collection = MagicMock()
    mock_collection.query.return_value = {"documents": DUMMY_DOCUMENTS}
    
    mock_instance = mock_client.return_value
    mock_instance.get_collection.return_value = mock_collection

    retriever = ChromaRetriever(collection_name="test_collection")
    results = retriever.retrieve_documents(DUMMY_EMBEDDING, top_k=2)
    
    assert isinstance(results, list)
    assert len(results) == 2
    assert results == ["Doc1", "Doc2"]

@patch("retrieval.chroma_vectorstore.chromadb.PersistentClient")
def test_retrieve_documents_empty(mock_client):
    mock_collection = MagicMock()
    mock_collection.query.return_value = {"documents": [[]]}  # No results

    mock_instance = mock_client.return_value
    mock_instance.get_collection.return_value = mock_collection

    retriever = ChromaRetriever(collection_name="test_collection")
    results = retriever.retrieve_documents(DUMMY_EMBEDDING)
    
    assert results == []

@patch("retrieval.chroma_vectorstore.chromadb.PersistentClient")
def test_retrieve_documents_failure(mock_client):
    mock_collection = MagicMock()
    mock_collection.query.side_effect = Exception("Query error")

    mock_instance = mock_client.return_value
    mock_instance.get_collection.return_value = mock_collection

    retriever = ChromaRetriever(collection_name="test_collection")
    results = retriever.retrieve_documents(DUMMY_EMBEDDING)

    assert results == []

def test_invalid_config():
    with patch("retrieval.chroma_vectorstore.CHROMA_DIR", new=""):
        with pytest.raises(ValueError):
            ChromaRetriever(collection_name="test_collection")