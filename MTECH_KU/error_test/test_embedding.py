import sys
import os

# Dynamically add the project root (1 levels up from test_embedding.py) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from retrieval.embedding import EmbeddingModel

VALID_QUERY = "What are the eligibility criteria for MTech admission?"   
def test_embed_valid_query():
    model = EmbeddingModel()
    embedding = model.embed_query(VALID_QUERY)

    assert isinstance(embedding, list), "Embedding output should be a list"
    assert all(isinstance(x, float) for x in embedding), "All elements must be floats"
    assert len(embedding) > 0, "Embedding should not be empty"

# Test empty query handling
def test_embed_empty_query():
    model = EmbeddingModel()
    embedding = model.embed_query("")

    assert embedding == [], "Embedding should be an empty list for empty query"