import sys
import os

# Dynamically add the project root (2 levels up from actions.py) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import logging
from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

from generation.rag_core import answer_query
from retrieval.embedding import EmbeddingModel
from retrieval.chroma_vectorstore import ChromaRetriever
from generation.llm import LLM


logger = logging.getLogger(__name__)

class ActionQueryRag(Action):
    # The name() method is required by Rasa to register this custom action.
    def name(self) -> Text:
        return "action_rag_query"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        user_query = tracker.latest_message.get("text")
        logger.info(f"Received user query: {user_query}")

        # Check both for None (not user_query) and for strings containing only whitespace (user_query.strip() == "").
        # This ensures we ignore invalid input such as empty strings, spaces, tabs, or newlines and only proceed if the user actually provided a meaningful query.
        if not user_query or user_query.strip() == "":
            logger.warning("Empty or missing user query detected.")
            dispatcher.utter_message(text="I didn't receive a valid question. Please try again.")
            return []

        try:
            # Create fresh instances of the embedding model, vector store, and LLM.
            embedding_model = EmbeddingModel()
            vectorstore = ChromaRetriever()
            llm = LLM()
            # answer_query() performs a multi-step RAG process:it calls the embedding model, queries the vector store, and then invokes the LLM.
            # This can be a slow operation depending on system load and model size.
            answer = answer_query(user_query.strip(), embedding_model, vectorstore, llm) # answer_query now takes explicit component instances, making it easier to unit test and reuse.

            if not answer or answer.strip() == "":
                answer = "I'm sorry, I couldn't find any information for that query."
            logger.info("Successfully retrieved answer from RAG logic.")
        except Exception as e:  # We use a general try-except here to catch unexpected issues during the RAG process,but rely on more specific error handling.
            logger.error(f"Error in RAG logic: {str(e)}")
            answer = "There was an error while retrieving the information. Please try again."

        cleaned_answer = ' '.join(answer.split())
        logger.info(f"Final response to user: {cleaned_answer}")
        dispatcher.utter_message(text=cleaned_answer)

        return []

