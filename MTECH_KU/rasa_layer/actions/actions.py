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
from fuzzywuzzy import process
from rasa_layer.faq_data import faq_list



logger = logging.getLogger(__name__)


class ActionSmartRouter(Action):
    def name(self) -> Text:
        return "action_smart_router"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        user_query = tracker.latest_message.get("text")
        logger.info(f"[SmartRouter] Received user query: '{user_query}'")

        if not user_query or user_query.strip() == "":
            logger.warning("[SmartRouter] Empty or missing user query.")
            dispatcher.utter_message(text="I didn't receive a valid question. Please try again.")
            return []

        # Fuzzy FAQ Handling
        try:
            questions = [faq["question"] for faq in faq_list]
            best_match, score = process.extractOne(user_query, questions)
            logger.debug(f"[SmartRouter] Best FAQ match: '{best_match}' with score {score}")

            if score >= 90:
                logger.info(f"[SmartRouter] High-confidence FAQ match found (Score: {score})")
                matched_faq = next(item for item in faq_list if item["question"] == best_match)
                answer = matched_faq["answer"]
                logger.info(f"[SmartRouter] Responding with FAQ answer: '{answer}'")
                dispatcher.utter_message(text=answer)
                return []

            logger.info(f"[SmartRouter] No strong FAQ match (score: {score}). Proceeding to RAG.")
        except Exception as e:
            logger.error(f"[SmartRouter] Error during FAQ matching: {str(e)}")

        # RAG Pipeline Fallback
        try:
            logger.debug("[SmartRouter] Initializing RAG components...")
            embedding_model = EmbeddingModel()
            vectorstore = ChromaRetriever()
            llm = LLM()

            logger.debug("[SmartRouter] Calling RAG pipeline...")
            answer = answer_query(user_query.strip(), embedding_model, vectorstore, llm)

            if not answer or answer.strip() == "":
                logger.warning("[SmartRouter] Empty response from RAG. Sending fallback message.")
                answer = "I'm sorry, I couldn't find any information for that query."

            cleaned_answer = ' '.join(answer.split())
            logger.info(f"[SmartRouter] Final RAG response: '{cleaned_answer}'")
            dispatcher.utter_message(text=cleaned_answer)

        except Exception as e:
            logger.error(f"[SmartRouter] Error in RAG logic: {str(e)}")
            dispatcher.utter_message(text="There was an error while retrieving the information. Please try again.")

        return []

