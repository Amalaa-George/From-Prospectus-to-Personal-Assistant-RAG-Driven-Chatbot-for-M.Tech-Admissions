"""
llm.py

This module provides a wrapper around the Groq API to interact with LLaMA 3 models.
It supports prompt-based response generation with streaming and includes retry and timeout handling
for reliable LLM integration in downstream tasks like RAG-based question answering systems.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from groq import Groq
from config import GROQ_API_KEY, LLM_MODEL_NAME

logger = logging.getLogger(__name__)


class LLM:
    """
    Wrapper class to interface with Groq-hosted LLaMA 3 models via the Groq Python SDK.
    
    Features:
    - Prompt-based text generation using chat completions
    - Streaming response support
    - Retry mechanism for robustness
    - Timeout handling for long-running requests

    Attributes:
        model (str): Name of the Groq-hosted LLaMA model to use (e.g., 'llama3-8b-8192')
        client (Groq): Initialized Groq client with API key

    Methods:
        generate(prompt: str, retries: int = 3, delay: float = 1.5, timeout: int = 10) -> str
            Generates a response for a given prompt using the Groq API with retries and timeout.
    """

    def __init__(self, model_name: str = LLM_MODEL_NAME, api_key: str = GROQ_API_KEY):
        if not api_key:
            raise ValueError("GROQ_API_KEY is not set. Please check your environment configuration.")
        self.client = Groq(api_key=api_key)
        self.model = model_name

    def _call_llm(self, prompt: str):
        """
        Internal method to make a streaming LLM request to the Groq API.
        
        Args:
            prompt (str): Input prompt string for the LLM.

        Returns:
            Iterable: Streaming chunks from the LLM.
        """
        return self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            top_p=1,
            max_tokens=1024,
            stream=True
        )

    def generate(self, prompt: str, retries: int = 3, delay: float = 1.5, timeout: int = 10) -> str:
        """
        Generates a response from the LLM for a given prompt, with timeout and retry logic.

        Args:
            prompt (str): The user prompt to send to the language model.
            retries (int): Number of retry attempts on failure.
            delay (float): Delay in seconds between retries.
            timeout (int): Timeout duration in seconds for each LLM call.

        Returns:
            str: The generated response or an error fallback message.
        """
        for attempt in range(retries):
            try:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(self._call_llm, prompt)
                    response_chunks = future.result(timeout=timeout)

                return "".join(
                    chunk.choices[0].delta.content
                    for chunk in response_chunks
                    if chunk.choices[0].delta.content
                )

            except TimeoutError:
                logger.warning(f"[Attempt {attempt + 1}] LLM request timed out.")
            except Exception as e:
                logger.warning(f"[Attempt {attempt + 1}] LLM request failed: {e}")
            time.sleep(delay)

        return "Failed to generate a response after multiple attempts."
