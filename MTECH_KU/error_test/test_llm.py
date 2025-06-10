import sys
import os

# Dynamically add the project root (1 levels up from test_llm.py) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import pytest
from unittest.mock import patch, MagicMock
from generation.llm import LLM

TEST_PROMPT = "What is the admission process for M.Tech?"

@patch("generation.llm.Groq")
def test_generate_success(mock_groq):
    # Mock response chunks with streamed tokens
    mock_chunk1 = MagicMock()
    mock_chunk1.choices[0].delta.content = "The "
    mock_chunk2 = MagicMock()
    mock_chunk2.choices[0].delta.content = "admission process "
    mock_chunk3 = MagicMock()
    mock_chunk3.choices[0].delta.content = "involves an entrance exam."

    mock_client_instance = mock_groq.return_value
    mock_client_instance.chat.completions.create.return_value = [mock_chunk1, mock_chunk2, mock_chunk3]

    llm = LLM(model_name="llama3-8b-8192", api_key="dummy_api_key")
    response = llm.generate(TEST_PROMPT)

    assert "The admission process involves an entrance exam." == response


@patch("generation.llm.Groq")
def test_generate_empty_response(mock_groq):
    # Simulate empty chunks
    mock_chunk = MagicMock()
    mock_chunk.choices[0].delta.content = ""

    mock_client_instance = mock_groq.return_value
    mock_client_instance.chat.completions.create.return_value = [mock_chunk]

    llm = LLM(model_name="llama3-8b-8192", api_key="dummy_api_key")
    response = llm.generate(TEST_PROMPT)

    assert response == ""


@patch("generation.llm.Groq")
def test_generate_exception_handling(mock_groq):
    # Simulate API exception on all retries
    mock_client_instance = mock_groq.return_value
    mock_client_instance.chat.completions.create.side_effect = Exception("API failure")

    llm = LLM(model_name="llama3-8b-8192", api_key="dummy_api_key")
    response = llm.generate(TEST_PROMPT, retries=3, delay=0.1)

    assert "Failed to generate a response after multiple attempts." in response


@patch("generation.llm.Groq")
def test_invalid_api_key(mock_groq):
    # Simulate invalid API key behavior
    with pytest.raises(ValueError):
        LLM(api_key=None)