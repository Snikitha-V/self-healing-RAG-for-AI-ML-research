import json
from unittest.mock import patch

import pytest

from src.critic.critic import check_grounding


class TestCheckGrounding:
    """Standalone critic module — test all code paths through check_grounding()."""

    def test_empty_documents_returns_insufficient(self):
        result = check_grounding("Q", "A", [])
        assert result == {"verdict": "insufficient", "verdict_reason": "No documents retrieved"}

    @patch("src.critic.critic.ollama.chat")
    def test_grounded_verdict(self, mock_chat):
        mock_chat.return_value = {
            "message": {"content": json.dumps({"verdict": "grounded", "reason": "All claims supported"})}
        }
        docs = [{"page_content": "Paris is the capital of France."}]
        result = check_grounding("What is the capital of France?", "Paris.", docs)
        assert result["verdict"] == "grounded"
        assert result["verdict_reason"] == "All claims supported"

    @patch("src.critic.critic.ollama.chat")
    def test_hallucinated_verdict(self, mock_chat):
        mock_chat.return_value = {
            "message": {"content": json.dumps({"verdict": "hallucinated", "reason": "London not in docs"})}
        }
        docs = [{"page_content": "Paris is in France."}]
        result = check_grounding("Capital of France?", "London.", docs)
        assert result["verdict"] == "hallucinated"
        assert result["verdict_reason"] == "London not in docs"

    @patch("src.critic.critic.ollama.chat")
    def test_insufficient_verdict(self, mock_chat):
        mock_chat.return_value = {
            "message": {"content": json.dumps({"verdict": "insufficient", "reason": "Docs don't contain answer"})}
        }
        docs = [{"page_content": "Some unrelated text."}]
        result = check_grounding("Q?", "A.", docs)
        assert result["verdict"] == "insufficient"
        assert result["verdict_reason"] == "Docs don't contain answer"

    @patch("src.critic.critic.ollama.chat")
    def test_json_parse_failure_falls_back_to_insufficient(self, mock_chat):
        mock_chat.return_value = {"message": {"content": "not valid json at all"}}
        docs = [{"page_content": "Something"}]
        result = check_grounding("Q?", "A.", docs)
        assert result["verdict"] == "insufficient"
        assert result["verdict_reason"] == "Failed to parse critic JSON"

    @patch("src.critic.critic.ollama.chat")
    def test_llm_error_returns_insufficient(self, mock_chat):
        mock_chat.side_effect = Exception("Connection refused")
        docs = [{"page_content": "Something"}]
        result = check_grounding("Q?", "A.", docs)
        assert result["verdict"] == "insufficient"
        assert "LLM error" in result["verdict_reason"]

    @patch("src.critic.critic.ollama.chat")
    def test_missing_verdict_key_defaults_to_insufficient(self, mock_chat):
        mock_chat.return_value = {
            "message": {"content": json.dumps({"reason": "no verdict key"})}
        }
        docs = [{"page_content": "Something"}]
        result = check_grounding("Q?", "A.", docs)
        assert result["verdict"] == "insufficient"

    @patch("src.critic.critic.ollama.chat")
    def test_missing_reason_key_defaults_to_empty_string(self, mock_chat):
        mock_chat.return_value = {
            "message": {"content": json.dumps({"verdict": "grounded"})}
        }
        docs = [{"page_content": "Something"}]
        result = check_grounding("Q?", "A.", docs)
        assert result["verdict"] == "grounded"
        assert result["verdict_reason"] == ""

    @patch("src.critic.critic.ollama.chat")
    def test_prompt_includes_question_answer_and_docs(self, mock_chat):
        mock_chat.return_value = {
            "message": {"content": json.dumps({"verdict": "grounded", "reason": "OK"})}
        }
        docs = [{"page_content": "Doc content here."}]
        check_grounding("My question?", "My answer.", docs)
        call_args = mock_chat.call_args
        messages = call_args[1]["messages"]
        user_msg = messages[1]["content"]
        assert "My question?" in user_msg
        assert "My answer." in user_msg
        assert "Doc content here." in user_msg
