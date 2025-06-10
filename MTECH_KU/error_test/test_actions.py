import sys
import os

# Dynamically add the project root (1 levels up from test_actions.py) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from unittest.mock import patch
from rasa_layer.actions.actions import ActionQueryRag
from rasa_sdk import Tracker
from rasa_sdk.executor import CollectingDispatcher

def test_action_rag_query_success():
    dispatcher = CollectingDispatcher()
    tracker = Tracker(sender_id="test_user", slots={}, latest_message={"text": "What is the M.Tech eligibility?"}, events=[], paused=False, followup_action=None, active_loop={}, latest_action_name=None)
    domain = {}

    with patch("rasa_layer.actions.actions.EmbeddingModel") as MockEmbedding, \
         patch("rasa_layer.actions.actions.ChromaRetriever") as MockRetriever, \
         patch("rasa_layer.actions.actions.LLM") as MockLLM, \
         patch("rasa_layer.actions.actions.answer_query") as mock_answer_query:
        
        mock_answer_query.return_value = "To be eligible for M.Tech, you must have a valid GATE score."

        action = ActionQueryRag()
        events = action.run(dispatcher, tracker, domain)

        # Check that the dispatcher was called with expected message
        assert any("valid GATE score" in m["text"] for m in dispatcher.messages)
        assert events == []

def test_action_rag_query_empty_input():
    dispatcher = CollectingDispatcher()
    tracker = Tracker(sender_id="test_user", slots={}, latest_message={"text": "   "}, events=[], paused=False, followup_action=None, active_loop={}, latest_action_name=None)
    domain = {}

    action = ActionQueryRag()
    events = action.run(dispatcher, tracker, domain)

    # Expecting fallback message for empty input
    assert any("didn't receive a valid question" in m["text"] for m in dispatcher.messages)
    assert events == []

def test_action_rag_query_exception_handling():
    dispatcher = CollectingDispatcher()
    tracker = Tracker(sender_id="test_user", slots={}, latest_message={"text": "What is the admission process?"}, events=[], paused=False, followup_action=None, active_loop={}, latest_action_name=None)
    domain = {}

    with patch("rasa_layer.actions.actions.EmbeddingModel") as MockEmbedding, \
         patch("rasa_layer.actions.actions.ChromaRetriever") as MockRetriever, \
         patch("rasa_layer.actions.actions.LLM") as MockLLM, \
         patch("rasa_layer.actions.actions.answer_query", side_effect=Exception("RAG failed")):

        action = ActionQueryRag()
        events = action.run(dispatcher, tracker, domain)

        assert any("error while retrieving the information" in m["text"] for m in dispatcher.messages)
        assert events == []

