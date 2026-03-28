"""Tests for RAG indexer and retriever."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestRAGIndexer:
    def test_chunk_text_basic(self):
        from src.rag.indexer import _chunk_text
        text = "This is a sentence. " * 50
        chunks = _chunk_text(text, chunk_size=30, overlap=5)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk.split()) > 10

    def test_chunk_text_short_text(self):
        from src.rag.indexer import _chunk_text
        text = "This is a short text with a few sentences. It should be one chunk."
        chunks = _chunk_text(text, chunk_size=100, overlap=10)
        assert len(chunks) == 1

    def test_chunk_text_overlap(self):
        from src.rag.indexer import _chunk_text
        text = ". ".join([f"Sentence number {i} with some content words here" for i in range(30)]) + "."
        chunks = _chunk_text(text, chunk_size=20, overlap=5)
        assert len(chunks) >= 2

    @patch("src.rag.indexer.get_settings")
    def test_load_documents_empty_dir(self, mock_settings):
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_settings.return_value = MagicMock(
                rag=MagicMock(chunk_size=512, chunk_overlap=64),
                embeddings=MagicMock(model="all-MiniLM-L6-v2"),
                rag_index_path=Path(tmpdir) / "index",
                knowledge_base_path=Path(tmpdir) / "kb",
            )
            from src.rag.indexer import RAGIndexer
            indexer = RAGIndexer()
            docs = indexer._load_documents()
            assert docs == []

    @patch("src.rag.indexer.get_settings")
    def test_load_documents_with_txt_file(self, mock_settings):
        with tempfile.TemporaryDirectory() as tmpdir:
            kb_path = Path(tmpdir) / "kb"
            kb_path.mkdir()
            test_file = kb_path / "test_article.txt"
            test_file.write_text("This is test financial content about investing.")

            mock_settings.return_value = MagicMock(
                rag=MagicMock(chunk_size=512, chunk_overlap=64),
                embeddings=MagicMock(model="all-MiniLM-L6-v2"),
                rag_index_path=Path(tmpdir) / "index",
                knowledge_base_path=kb_path,
            )

            from src.rag.indexer import RAGIndexer
            indexer = RAGIndexer()
            docs = indexer._load_documents()

            assert len(docs) == 1
            assert "test financial content" in docs[0]["text"]


class TestRAGRetriever:
    @patch("src.rag.retriever.get_settings")
    def test_search_returns_empty_when_no_index(self, mock_settings):
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_settings.return_value = MagicMock(
                rag=MagicMock(top_k=5),
                embeddings=MagicMock(model="all-MiniLM-L6-v2"),
                rag_index_path=Path(tmpdir) / "index",
                knowledge_base_path=Path(tmpdir) / "kb",
            )

            from src.rag.retriever import RAGRetriever
            retriever = RAGRetriever()

            # Should not raise; just returns empty
            with patch.object(retriever, "_ensure_index", return_value=False):
                results = retriever.search("what is a P/E ratio")
            assert results == []

    @patch("src.rag.retriever.get_settings")
    def test_get_context_empty(self, mock_settings):
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_settings.return_value = MagicMock(
                rag=MagicMock(top_k=5),
                embeddings=MagicMock(model="all-MiniLM-L6-v2"),
                rag_index_path=Path(tmpdir) / "index",
                knowledge_base_path=Path(tmpdir) / "kb",
            )

            from src.rag.retriever import RAGRetriever
            retriever = RAGRetriever()

            with patch.object(retriever, "search", return_value=[]):
                context = retriever.get_context("test query")
            assert context == ""
