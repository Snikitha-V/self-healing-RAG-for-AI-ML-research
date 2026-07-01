from unittest.mock import patch

from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


class TestHealth:
    def test_returns_ok(self):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestQuery:
    @patch("src.api.main.get_graph")
    def test_returns_answer_and_retry_count(self, mock_get_graph):
        mock_graph = mock_get_graph.return_value
        mock_graph.invoke.return_value = {
            "final_answer": "Paris is the capital of France.",
            "retry_count": 0,
        }

        response = client.post("/query", json={"question": "What is the capital of France?"})

        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "Paris is the capital of France."
        assert data["retry_count"] == 0

    @patch("src.api.main.get_graph")
    def test_passes_correct_initial_state(self, mock_get_graph):
        mock_graph = mock_get_graph.return_value
        mock_graph.invoke.return_value = {
            "final_answer": "Some answer.",
            "retry_count": 1,
        }

        client.post("/query", json={"question": "What is RAG?"})

        initial_state = mock_graph.invoke.call_args[0][0]
        assert initial_state["question"] == "What is RAG?"
        assert initial_state["query"] == "What is RAG?"
        assert initial_state["retry_count"] == 0
        assert initial_state["documents"] == []
        assert initial_state["final_answer"] == ""

    @patch("src.api.main.get_graph")
    def test_handles_retry_loop_in_result(self, mock_get_graph):
        mock_graph = mock_get_graph.return_value
        mock_graph.invoke.return_value = {
            "final_answer": "I couldn't find sufficient information...",
            "retry_count": 2,
        }

        response = client.post("/query", json={"question": "Obscure topic?"})

        assert response.status_code == 200
        data = response.json()
        assert "couldn't find sufficient" in data["answer"]
        assert data["retry_count"] == 2

    def test_query_fails_without_question(self):
        response = client.post("/query", json={})
        assert response.status_code == 422
