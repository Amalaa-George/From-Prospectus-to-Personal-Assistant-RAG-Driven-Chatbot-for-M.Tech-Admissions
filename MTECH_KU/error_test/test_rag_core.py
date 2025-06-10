import sys
import os

# Dynamically add the project root (1 levels up from test_rag_core.py) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from generation.rag_core import answer_query

class MockEmbeddingModel:
    def embed_query(self, text):
        if text == "error":
            raise Exception("Embedding error")
        return [0.1, 0.2, 0.3]

class MockVectorStore:
    def retrieve_documents(self, embedding):
        if embedding == [9.9]:
            raise Exception("Vectorstore error")
        if embedding == [0.1, 0.2, 0.3]:
            return ["Doc 1 line 1\nDoc 1 line 2", "Doc 2 line"]
        return []

class MockLLM:
    def generate(self, prompt):
        if prompt == "fail" or "cause error" in prompt:
            raise Exception("LLM error")
        return "This is the generated answer."

def test_empty_query():
    response = answer_query("   ", MockEmbeddingModel(), MockVectorStore(), MockLLM())
    assert "cannot be empty" in response

def test_missing_components():
    response = answer_query("What is AI?", None, None, None)
    assert "Missing RAG components" in response

def test_embedding_failure():
    response = answer_query("error", MockEmbeddingModel(), MockVectorStore(), MockLLM())
    assert "embedding error" in response.lower() or "embedding" in response.lower()

def test_vectorstore_failure():
    class FailingVectorStore:
        def retrieve_documents(self, _):
            raise Exception("Vectorstore retrieval failed")

    response = answer_query("What is AI?", MockEmbeddingModel(), FailingVectorStore(), MockLLM())
    assert "couldn't access" in response.lower()

def test_no_documents_found():
    class EmptyVectorStore:
        def retrieve_documents(self, _):
            return []

    response = answer_query("What is AI?", MockEmbeddingModel(), EmptyVectorStore(), MockLLM())
    assert "couldn't find" in response.lower()

def test_llm_failure():
    class FailingLLM:
        def generate(self, _):
            raise Exception("LLM failure")

    response = answer_query("What is AI?", MockEmbeddingModel(), MockVectorStore(), FailingLLM())
    assert "went wrong" in response.lower()

def test_successful_response():
    response = answer_query("What is AI?", MockEmbeddingModel(), MockVectorStore(), MockLLM())
    assert response == "This is the generated answer."