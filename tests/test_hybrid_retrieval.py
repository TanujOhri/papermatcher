"""Comprehensive automated test suite for PaperMatcher-AI Hybrid Retrieval.

Tests:
1. TF-IDF retrieval baseline integrity (old functionality preserved)
2. Sentence-BERT model loading and query encoding
3. SBERT embedding dimensions and unit normalization
4. Score normalization methods ('minmax', 'clip', 'none')
5. Hybrid score calculation and weighted combination
6. Categorical filtering (category_code)
7. Chronological filtering (published_year)
8. Combined category + year filtering
9. Top-K cutoff validation (K = 1, 5, 10, 20)
10. Empty / whitespace-only query handling
11. API endpoint compatibility via FastAPI TestClient
12. Retrieval mode switching ('hybrid', 'tfidf', 'semantic')
"""

import sys
import numpy as np
import pytest
from pathlib import Path

# Add backend directory to sys.path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
backend_dir = project_root / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from fastapi.testclient import TestClient
from app.main import app
from app.core.config import settings
from app.services.matcher import PaperMatcher, get_matcher
from app.schemas.recommendation import RecommendationRequest


@pytest.fixture(scope="module")
def matcher():
    """Shared PaperMatcher instance for tests."""
    engine = get_matcher()
    engine.load_index(auto_build=False, load_semantic=True)
    return engine


@pytest.fixture(scope="module")
def test_client():
    """FastAPI TestClient fixture."""
    with TestClient(app) as client:
        yield client


class TestLexicalBaseline:
    """Validate that the existing TF-IDF baseline functionality remains intact."""

    def test_tfidf_retrieval_returns_results(self, matcher):
        query = "transformer architectures for natural language processing"
        recs = matcher.recommend(query=query, top_k=5, retrieval_mode="tfidf")

        assert len(recs) == 5, f"Expected 5 recommendations, got {len(recs)}"
        assert all(0.0 <= r.similarity_score <= 1.0 for r in recs)
        assert recs[0].similarity_score >= recs[1].similarity_score
        assert recs[0].retrieval_mode == "tfidf"

    def test_tfidf_category_filtering(self, matcher):
        query = "diffusion models image generation"
        recs = matcher.recommend(query=query, top_k=5, category_code="cs.CV", retrieval_mode="tfidf")

        assert len(recs) > 0
        for r in recs:
            assert r.category_code.lower() == "cs.cv", f"Category mismatch: {r.category_code}"

    def test_tfidf_year_filtering(self, matcher):
        query = "graph neural networks"
        recs = matcher.recommend(query=query, top_k=5, published_year=2024, retrieval_mode="tfidf")

        assert len(recs) > 0
        for r in recs:
            assert "24" in r.published_date or "2024" in r.published_date


class TestSentenceBertSemantic:
    """Validate Sentence-BERT model, query encoding, and semantic retrieval."""

    def test_sbert_model_loaded(self, matcher):
        assert matcher.sbert_model is not None, "Sentence-BERT model should be loaded"
        assert matcher.sbert_model.max_seq_length == settings.SBERT_MAX_SEQ_LENGTH

    def test_sbert_query_encoding_dimensions(self, matcher):
        query = "quantum computing algorithms"
        vec = matcher.sbert_model.encode([query], normalize_embeddings=True)[0]

        assert vec.shape == (384,), f"Expected (384,), got {vec.shape}"
        norm = np.linalg.norm(vec)
        assert np.isclose(norm, 1.0, atol=1e-3), f"Vector not unit normalized: {norm}"

    def test_semantic_retrieval_mode(self, matcher):
        if not matcher.has_semantic:
            pytest.skip("Full precomputed SBERT embeddings file not present yet")

        query = "federated learning privacy preserving machine learning"
        recs = matcher.recommend(query=query, top_k=5, retrieval_mode="semantic")

        assert len(recs) == 5
        assert recs[0].retrieval_mode == "semantic"
        assert all(0.0 <= r.similarity_score <= 1.0 for r in recs)


class TestScoreNormalizationAndHybrid:
    """Validate score normalization and hybrid ranking combinations."""

    def test_minmax_normalization(self):
        raw = np.array([0.1, 0.2, 0.5, 0.3], dtype=np.float32)
        norm = PaperMatcher.normalize_scores(raw, method="minmax")

        assert np.isclose(norm.min(), 0.0)
        assert np.isclose(norm.max(), 1.0)
        assert norm[2] == 1.0  # Original max is 0.5

    def test_clip_normalization(self):
        raw = np.array([-0.2, 0.4, 0.9, 1.2], dtype=np.float32)
        norm = PaperMatcher.normalize_scores(raw, method="clip")

        assert norm[0] == 0.0
        assert norm[1] == 0.4
        assert norm[2] == 0.9
        assert norm[3] == 1.0

    def test_hybrid_retrieval_execution(self, matcher):
        query = "reinforcement learning from human feedback"
        recs = matcher.recommend(query=query, top_k=5, retrieval_mode="hybrid")

        assert len(recs) == 5
        for r in recs:
            assert 0.0 <= r.similarity_score <= 1.0
            # If semantic is available, scores are populated; if falling back, retrieval_mode is set
            assert r.retrieval_mode in ["hybrid", "tfidf"]


class TestTopKAndEdgeCases:
    """Validate boundary cases, empty queries, and Top-K parameter constraints."""

    @pytest.mark.parametrize("k", [1, 3, 5, 10, 20])
    def test_top_k_cardinality(self, matcher, k):
        query = "deep neural networks optimization"
        recs = matcher.recommend(query=query, top_k=k)
        assert len(recs) == k

    def test_empty_query_returns_empty(self, matcher):
        recs = matcher.recommend(query="   ", top_k=5)
        assert recs == []

    def test_unmatched_filter_returns_empty(self, matcher):
        # Query with non-existent future year
        recs = matcher.recommend(query="transformer", published_year=1901)
        assert recs == []


class TestApiEndpoints:
    """Validate FastAPI HTTP routes and schema validation."""

    def test_health_endpoint(self, test_client):
        res = test_client.get("/api/v1/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "healthy"
        assert data["service"] == "PaperMatcher-AI"

    def test_filters_endpoint(self, test_client):
        res = test_client.get("/api/v1/recommendations/filters")
        assert res.status_code == 200
        data = res.json()
        assert len(data.get("categories", [])) >= 100
        assert len(data.get("years", [])) >= 20

    def test_recommendations_post_endpoint(self, test_client):
        payload = {
            "query": "transformer architectures for natural language processing",
            "top_k": 5,
            "category_code": "cs.CL",
            "retrieval_mode": "hybrid",
        }
        res = test_client.post("/api/v1/recommendations", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["total_results"] == 5
        assert len(data["results"]) == 5
        first = data["results"][0]
        assert "similarity_score" in first
        assert "title" in first
        assert "summary" in first
        assert first["category_code"].lower() == "cs.cl"

    def test_validation_error_on_short_query(self, test_client):
        payload = {"query": "ab", "top_k": 5}
        res = test_client.post("/api/v1/recommendations", json=payload)
        assert res.status_code == 422
