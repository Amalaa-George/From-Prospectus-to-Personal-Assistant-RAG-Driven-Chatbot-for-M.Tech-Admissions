"""

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
        """
        Initializes the LLM client and sets up initial system message.

        Args:
            model_name (str): Name of the model to be used from Groq.
            api_key (str): Groq API key for authentication.

        Raises:
            ValueError: If the API key is not set.
        """

        if not api_key:
            raise ValueError("GROQ_API_KEY is not set. Please check your environment configuration.")
        self.client = Groq(api_key=api_key)
        self.model = model_name
        self.chat_history = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]

    def _call_llm(self, prompt: str):
        """
        Internal method to prepare and send a streaming chat completion request to the LLM.

        This method maintains a rolling window of the last 3 user-assistant message pairs
        (plus the system message) to provide short-term context during conversation.

        Args:
            prompt (str): The user input to include in the conversation history.

        Returns:
            Generator: A generator yielding streamed response chunks from the Groq API.
        """
        # Add latest user input to history
        self.chat_history.append({"role": "user", "content": prompt})
        logger.info("User input added to chat history.")
        # Keep only last 3 user-assistant pairs (6 messages) + system message
        filtered_history = [self.chat_history[0]] + self.chat_history[-6:]
        logger.debug(f"Filtered chat history (memory): {filtered_history}")

        return self.client.chat.completions.create(
            model=self.model,
            messages=filtered_history,
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

                # Collect streamed response
                response = "".join(
                    chunk.choices[0].delta.content
                    for chunk in response_chunks
                    if chunk.choices[0].delta.content
                )

                # Add assistant reply to history
                self.chat_history.append({"role": "assistant", "content": response})
                logger.info("Assistant response added to chat history.")
                logger.debug(f"Updated chat history: {self.chat_history}")

                return response

            except TimeoutError:
                logger.warning(f"[Attempt {attempt + 1}] LLM request timed out.")
            except Exception as e:
                logger.warning(f"[Attempt {attempt + 1}] LLM request failed: {e}")
            time.sleep(delay)

        return "Failed to generate a response after multiple attempts."
