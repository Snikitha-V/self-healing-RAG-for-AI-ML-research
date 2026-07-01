from unittest.mock import patch

from src.graph.nodes import critique, finalize, generate, retrieve, rewrite_query


def _state(**overrides):
    base = {
        "question": "What is RAG?",
        "query": "What is RAG?",
        "documents": [],
        "answer": "",
        "verdict": "",
        "verdict_reason": "",
        "retry_count": 0,
        "final_answer": "",
    }
    base.update(overrides)
    return base


class TestRetrieve:
    @patch("src.graph.nodes.similarity_search")
    def test_calls_similarity_search_with_query(self, mock_search):
        mock_search.return_value = [{"page_content": "doc1", "metadata": {}}]
        state = _state(query="custom query")
        result = retrieve(state)
        mock_search.assert_called_once_with("custom query", k=4)
        assert result == {"documents": [{"page_content": "doc1", "metadata": {}}]}

    @patch("src.graph.nodes.similarity_search")
    def test_falls_back_to_question_when_query_empty(self, mock_search):
        mock_search.return_value = []
        state = _state(query="", question="fallback question")
        retrieve(state)
        mock_search.assert_called_once_with("fallback question", k=4)


class TestGenerate:
    def test_no_documents_returns_fallback(self):
        state = _state(documents=[])
        result = generate(state)
        assert "No relevant documents" in result["answer"]

    @patch("src.graph.nodes.ollama.chat")
    def test_calls_ollama_with_documents_and_question(self, mock_chat):
        mock_chat.return_value = {"message": {"content": "RAG means retrieval-augmented generation."}}
        docs = [{"page_content": "RAG is a technique.", "metadata": {}}]
        state = _state(question="What is RAG?", documents=docs)
        result = generate(state)
        mock_chat.assert_called_once()
        messages = mock_chat.call_args[1]["messages"]
        assert any("RAG is a technique." in m["content"] for m in messages)
        assert any("What is RAG?" in m["content"] for m in messages)
        assert result["answer"] == "RAG means retrieval-augmented generation."

    @patch("src.graph.nodes.ollama.chat")
    def test_llm_error_returns_error_message(self, mock_chat):
        mock_chat.side_effect = Exception("LLM timeout")
        docs = [{"page_content": "Some content.", "metadata": {}}]
        state = _state(documents=docs)
        result = generate(state)
        assert "LLM error" in result["answer"]


class TestCritique:
    @patch("src.graph.nodes.check_grounding")
    def test_delegates_to_check_grounding(self, mock_check):
        mock_check.return_value = {"verdict": "grounded", "verdict_reason": "OK"}
        docs = [{"page_content": "Content.", "metadata": {}}]
        state = _state(question="Q?", answer="A.", documents=docs)
        result = critique(state)
        mock_check.assert_called_once_with("Q?", "A.", docs)
        assert result == {"verdict": "grounded", "verdict_reason": "OK"}

    @patch("src.graph.nodes.check_grounding")
    def test_passes_empty_list_when_no_documents(self, mock_check):
        mock_check.return_value = {"verdict": "insufficient", "verdict_reason": "No docs"}
        state = _state(documents=[])
        critique(state)
        mock_check.assert_called_once_with("What is RAG?", "", [])


class TestRewriteQuery:
    @patch("src.graph.nodes.ollama.chat")
    def test_rewrites_query_and_increments_retry(self, mock_chat):
        mock_chat.return_value = {"message": {"content": "What is retrieval augmented generation?"}}
        state = _state(query="What is RAG?", verdict_reason="Too vague")
        result = rewrite_query(state)
        assert result["query"] == "What is retrieval augmented generation?"
        assert result["retry_count"] == 1

    @patch("src.graph.nodes.ollama.chat")
    def test_llm_error_falls_back_to_original_query(self, mock_chat):
        mock_chat.side_effect = Exception("LLM down")
        state = _state(query="original", verdict_reason="bad")
        result = rewrite_query(state)
        assert result["query"] == "original"
        assert result["retry_count"] == 1

    @patch("src.graph.nodes.ollama.chat")
    def test_uses_question_when_query_is_empty(self, mock_chat):
        mock_chat.side_effect = Exception("Error")
        state = _state(query="", question="fallback question", verdict_reason="bad")
        result = rewrite_query(state)
        assert result["query"] == "fallback question"
        assert result["retry_count"] == 1

    @patch("src.graph.nodes.ollama.chat")
    def test_includes_critic_feedback_in_prompt(self, mock_chat):
        mock_chat.return_value = {"message": {"content": "new query"}}
        state = _state(query="old", verdict_reason="Answer contains hallucinated details")
        rewrite_query(state)
        messages = mock_chat.call_args[1]["messages"]
        user_msg = messages[1]["content"]
        assert "Answer contains hallucinated details" in user_msg


class TestFinalize:
    def test_grounded_returns_answer(self):
        state = _state(verdict="grounded", answer="Paris is the capital.")
        result = finalize(state)
        assert result["final_answer"] == "Paris is the capital."

    def test_hallucinated_returns_fallback(self):
        state = _state(verdict="hallucinated", answer="Made up answer.")
        result = finalize(state)
        assert "couldn't find sufficient information" in result["final_answer"]

    def test_insufficient_returns_fallback(self):
        state = _state(verdict="insufficient", answer="Partial answer.")
        result = finalize(state)
        assert "couldn't find sufficient information" in result["final_answer"]
