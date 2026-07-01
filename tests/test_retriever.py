from unittest.mock import MagicMock, patch

from src.retriever.vectorstore import add_documents, similarity_search


class TestSimilaritySearch:
    @patch("src.retriever.vectorstore.embed_texts")
    @patch("src.retriever.vectorstore.get_embedder")
    @patch("src.retriever.vectorstore.get_client")
    def test_returns_formatted_results(self, mock_get_client, mock_get_embedder, mock_embed_texts):
        mock_embed_texts.return_value = [[0.1] * 384]

        mock_point = MagicMock()
        mock_point.payload = {"text": "Document content", "metadata": {"source": "arxiv"}}

        mock_response = MagicMock()
        mock_response.points = [mock_point]

        mock_client = mock_get_client.return_value
        mock_client.query_points.return_value = mock_response

        results = similarity_search("test query", k=4)

        assert len(results) == 1
        assert results[0]["page_content"] == "Document content"
        assert results[0]["metadata"]["source"] == "arxiv"

    @patch("src.retriever.vectorstore.embed_texts")
    @patch("src.retriever.vectorstore.get_embedder")
    @patch("src.retriever.vectorstore.get_client")
    def test_passes_correct_args_to_qdrant(self, mock_get_client, mock_get_embedder, mock_embed_texts):
        mock_embed_texts.return_value = [[0.2] * 384]

        mock_response = MagicMock()
        mock_response.points = []

        mock_client = mock_get_client.return_value
        mock_client.query_points.return_value = mock_response

        similarity_search("my query", k=3)

        mock_client.query_points.assert_called_once_with(
            collection_name="documents",
            query=[0.2] * 384,
            limit=3,
            with_payload=True,
        )


class TestAddDocuments:
    @patch("src.retriever.vectorstore._existing_ids")
    @patch("src.retriever.vectorstore.embed_texts")
    @patch("src.retriever.vectorstore.get_embedder")
    @patch("src.retriever.vectorstore.get_client")
    def test_upserts_new_documents(self, mock_get_client, mock_get_embedder, mock_embed_texts, mock_existing_ids):
        mock_embed_texts.return_value = [[0.1] * 384]
        mock_existing_ids.return_value = set()

        docs = [{"page_content": "New doc", "metadata": {"source": "test"}}]
        add_documents(docs)

        mock_client = mock_get_client.return_value
        mock_client.upsert.assert_called_once()
        points = mock_client.upsert.call_args[1]["points"]
        assert len(points) == 1
        assert points[0].payload["text"] == "New doc"

    @patch("src.retriever.vectorstore._existing_ids")
    @patch("src.retriever.vectorstore.embed_texts")
    @patch("src.retriever.vectorstore.get_embedder")
    @patch("src.retriever.vectorstore.get_client")
    def test_skips_duplicates(self, mock_get_client, mock_get_embedder, mock_embed_texts, mock_existing_ids):
        from src.retriever.vectorstore import _make_id

        doc_text = "Existing doc"
        mock_embed_texts.return_value = [[0.1] * 384]
        mock_existing_ids.return_value = {_make_id(doc_text)}

        docs = [{"page_content": doc_text, "metadata": {}}]
        add_documents(docs)

        mock_client = mock_get_client.return_value
        mock_client.upsert.assert_not_called()
