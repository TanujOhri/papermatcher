"""Paper Matcher Service.

Performs research paper recommendation using a hybrid architecture combining:
1. Lexical Retrieval: Sublinear TF-IDF + Cosine Similarity.
2. Semantic Retrieval: Sentence-BERT (all-MiniLM-L6-v2) dense embeddings.
3. Hybrid Retrieval: Configurable weighted combination of normalized scores.

Architecture:
    User Query
        |
        +-----------------------+
        |                       |
        v                       v
    TF-IDF Vectorizer       Sentence-BERT
    (Sparse 50k features)   (Dense 384 dimensions)
        |                       |
        v                       v
    TF-IDF Cosine Score     SBERT Cosine Score
        |                       |
        +-----------+-----------+
                    |
                    v
            Score Normalization
            & Weighted Hybrid Scoring
            final = w_tfidf * tfidf_norm + w_sbert * sbert_norm
                    |
                    v
            Boolean Mask Filtering (Category / Year)
                    |
                    v
            Linear-Time Top-K Selection (np.argpartition)
                    |
                    v
            Hydrated Paper Recommendations
"""

import os
import time
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

import joblib
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from app.core.config import settings
from app.core.exceptions import EngineNotInitializedException
from app.schemas.recommendation import PaperRecommendation
from app.services.dataset import get_dataset_loader
from app.services.sbert_model import get_sentence_transformer

logger = logging.getLogger("papermatcher.matcher")


def _extract_year(date_str: Optional[str]) -> int:
    """Extract 4-digit calendar year from date string (e.g. '8/1/93', '5/1/24', '2024-05-10')."""
    if not date_str or not isinstance(date_str, str):
        return -1
    parts = date_str.strip().split("/")
    if parts:
        try:
            y = int(parts[-1])
            if y < 100:
                return 1900 + y if y >= 90 else 2000 + y
            return y
        except ValueError:
            pass
    # Fallback to prefix if hyphenated (e.g. '2024-05-10')
    if "-" in date_str:
        try:
            return int(date_str.split("-")[0])
        except ValueError:
            pass
    return -1


class PaperMatcher:
    """Core NLP recommendation engine supporting TF-IDF, Sentence-BERT, and Hybrid retrieval."""

    def __init__(self, models_dir: Optional[Path] = None):
        if models_dir is None:
            self.models_dir = settings.get_models_dir()
        else:
            self.models_dir = Path(models_dir)

        # Lexical (TF-IDF) components
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.paper_matrix: Optional[sp.csr_matrix] = None

        # Semantic (Sentence-BERT) components
        self.sbert_model: Optional[SentenceTransformer] = None
        self.sbert_embeddings: Optional[np.ndarray] = None
        self.has_semantic: bool = False

        # Metadata and pre-extracted filter indices
        self.paper_metadata: Optional[List[Dict[str, Any]]] = None
        self.category_codes: Optional[np.ndarray] = None
        self.published_years: Optional[np.ndarray] = None
        self.filter_options: Optional[Dict[str, Any]] = None

        self.is_initialized: bool = False

    def load_index(self, auto_build: bool = False, load_semantic: bool = True) -> bool:
        """Load persisted TF-IDF vectorizer, sparse matrix, metadata, and SBERT embeddings.

        Args:
            auto_build: If True, builds the index if artifacts are missing.
            load_semantic: If True, also attempts to load Sentence-BERT model and embeddings.

        Returns:
            bool: True if base index loaded successfully, False otherwise.
        """
        vectorizer_path = self.models_dir / "tfidf_vectorizer.joblib"
        matrix_path = self.models_dir / "tfidf_matrix.npz"
        metadata_path = self.models_dir / "paper_metadata.joblib"
        sbert_path = settings.get_sbert_embeddings_file_path()

        artifacts_exist = (
            vectorizer_path.exists()
            and matrix_path.exists()
            and metadata_path.exists()
        )

        if not artifacts_exist:
            if auto_build:
                logger.info("Base index artifacts not found. Initiating TF-IDF build...")
                from app.services.build_index import build_index
                build_index()
            else:
                logger.warning(
                    "TF-IDF index artifacts missing in %s. Base engine remains uninitialized.",
                    self.models_dir,
                )
                self.is_initialized = False
                return False

        logger.info("Loading TF-IDF artifacts from %s...", self.models_dir)
        t0 = time.time()

        self.vectorizer = joblib.load(vectorizer_path)
        self.paper_matrix = sp.load_npz(matrix_path)
        self.paper_metadata = joblib.load(metadata_path)

        total_docs = len(self.paper_metadata)

        # Pre-extract category codes and years into numpy arrays for fast vectorized filtering
        self.category_codes = np.array(
            [str(m.get("category_code", "")).strip().lower() for m in self.paper_metadata]
        )
        self.published_years = np.array(
            [_extract_year(m.get("published_date", "")) for m in self.paper_metadata],
            dtype=np.int32,
        )

        # Precompute dynamic filter options directly from metadata
        from collections import Counter
        cat_map = {}
        cat_counts = Counter()
        for m in self.paper_metadata:
            code = str(m.get("category_code", "")).strip()
            name = str(m.get("category", "")).strip()
            if code:
                cat_counts[code] += 1
                if code not in cat_map or len(name) > len(cat_map[code]):
                    cat_map[code] = name

        unique_years = sorted(
            [int(y) for y in set(self.published_years) if 1990 <= y <= 2030],
            reverse=True,
        )

        self.filter_options = {
            "categories": [
                {
                    "code": code,
                    "name": cat_map.get(code, code),
                    "count": count,
                }
                for code, count in cat_counts.most_common()
            ],
            "years": unique_years,
        }

        # Load Sentence-BERT components if requested
        self.has_semantic = False
        if load_semantic:
            self._load_semantic_components(sbert_path, total_docs)

        self.is_initialized = True

        load_time = time.time() - t0
        logger.info(
            "PaperMatcher initialized in %.2f seconds (Documents: %d, Lexical Features: %d, Semantic Ready: %s).",
            load_time,
            self.paper_matrix.shape[0],
            self.paper_matrix.shape[1],
            self.has_semantic,
        )
        return True

    def _load_semantic_components(self, sbert_path: Path, expected_docs: int) -> None:
        """Load Sentence-BERT model and precomputed embeddings."""
        try:
            # 1. Load Sentence-BERT model once
            logger.info("Initializing Sentence-BERT model '%s'...", settings.SBERT_MODEL_NAME)
            t_sbert_load = time.time()
            self.sbert_model = get_sentence_transformer(
                model_name=settings.SBERT_MODEL_NAME,
                max_seq_length=settings.SBERT_MAX_SEQ_LENGTH,
            )
            logger.info("Sentence-BERT model loaded in %.2f seconds.", time.time() - t_sbert_load)

            # 2. Load precomputed embeddings
            if not sbert_path.exists():
                logger.warning(
                    "Sentence-BERT embeddings file not found at %s. "
                    "Run 'python -m app.services.build_semantic_index' to compile dense embeddings. "
                    "Hybrid retrieval will fall back to TF-IDF baseline.",
                    sbert_path,
                )
                self.has_semantic = False
                return

            logger.info("Loading precomputed SBERT embeddings from %s...", sbert_path)
            t_emb_load = time.time()
            embs = np.load(sbert_path, mmap_mode="r")

            # Validate dimensions and count
            if embs.shape[0] != expected_docs:
                logger.error(
                    "SBERT embedding count mismatch! Found %d vectors, expected %d matching paper_metadata. "
                    "Semantic search disabled for safety until re-indexed.",
                    embs.shape[0],
                    expected_docs,
                )
                self.has_semantic = False
                return

            self.sbert_embeddings = embs
            self.has_semantic = True
            logger.info(
                "SBERT embeddings loaded in %.2fs (Shape: %s, Dimensions: %d). Semantic retrieval active.",
                time.time() - t_emb_load,
                self.sbert_embeddings.shape,
                self.sbert_embeddings.shape[1],
            )
        except Exception:
            logger.exception("Failed to load Sentence-BERT model or embeddings:")
            self.has_semantic = False

    def get_filter_options(self) -> Dict[str, Any]:
        """Return available categories and years dynamically computed from the dataset."""
        if not self.is_initialized or self.filter_options is None:
            raise EngineNotInitializedException(
                "Recommendation engine is not initialized. Model and dataset must be loaded first."
            )
        return self.filter_options

    @staticmethod
    def normalize_scores(
        scores: np.ndarray,
        method: str = "minmax",
    ) -> np.ndarray:
        """Normalize raw similarity scores to ensure compatible ranges before hybrid combination.

        Score Range Analysis:
        - TF-IDF Cosine Scores: Non-negative, sparse, typically range in [0.0, 0.45].
        - SBERT Cosine Scores: Dense unit dot products, range in [-1.0, 1.0], typically [0.0, 0.85] for queries.

        Normalization Methods:
        1. 'minmax':
           Scaled relative to min and max of candidate distribution:
           s_norm = (s - min) / (max - min + 1e-9)
           Ensures both lexical and semantic features span [0.0, 1.0], preserving proportional 40/60 weighting.
        2. 'clip':
           Clamps scores to [0.0, 1.0], preserving absolute cosine values.
        3. 'none':
           Leaves raw scores unscaled.

        Args:
            scores: 1D NumPy array of raw similarity scores.
            method: Normalization technique ('minmax', 'clip', or 'none').

        Returns:
            Normalized 1D NumPy array.
        """
        if method == "none" or len(scores) == 0:
            return scores

        if method == "clip":
            return np.clip(scores, 0.0, 1.0)

        # Default: 'minmax'
        s_min = float(np.min(scores))
        s_max = float(np.max(scores))
        diff = s_max - s_min
        if diff <= 1e-9:
            return np.ones_like(scores, dtype=np.float32) if s_max > 0 else np.zeros_like(scores, dtype=np.float32)

        return ((scores - s_min) / diff).astype(np.float32)

    def recommend(
        self,
        query: str,
        top_k: int = 5,
        category_code: Optional[str] = None,
        published_year: Optional[int] = None,
        retrieval_mode: Optional[str] = None,
    ) -> List[PaperRecommendation]:
        """Generate top-K paper recommendations for a given query text.

        Supports retrieval modes:
        - 'hybrid' (default): 0.40 * TF-IDF + 0.60 * SBERT
        - 'tfidf': Pure lexical TF-IDF baseline
        - 'semantic': Pure dense Sentence-BERT semantic retrieval

        Pipeline:
        1. Clean and normalize query.
        2. Resolve effective retrieval mode (falls back gracefully to TF-IDF if semantic index missing).
        3. Vectorize query with TF-IDF and/or Sentence-BERT.
        4. Calculate cosine similarity via sparse or dense dot products.
        5. Apply pre-ranking boolean masks for category_code and published_year.
        6. Compute normalized hybrid scores for candidate papers.
        7. Select Top-K most similar documents using linear-time argpartition.
        8. Map to metadata and return List[PaperRecommendation].

        Raises:
            EngineNotInitializedException: If index artifacts are not loaded.
        """
        if not self.is_initialized or self.vectorizer is None or self.paper_matrix is None or self.paper_metadata is None:
            raise EngineNotInitializedException(
                "Recommendation engine is not initialized. Model and dataset must be loaded "
                "before generating recommendations."
            )

        # 1. Clean query
        loader = get_dataset_loader()
        clean_query = loader.preprocess_text(query)
        if not clean_query:
            return []

        # 2. Determine effective retrieval mode
        req_mode = (retrieval_mode or settings.RETRIEVAL_MODE).strip().lower()
        if req_mode not in ["hybrid", "tfidf", "semantic"]:
            req_mode = "hybrid"

        effective_mode = req_mode
        if effective_mode in ["hybrid", "semantic"] and not self.has_semantic:
            if effective_mode == "semantic":
                raise EngineNotInitializedException(
                    "Sentence-BERT semantic index is not initialized or embedding count mismatch. "
                    "Run 'python -m app.services.build_semantic_index' to compile dense embeddings."
                )
            logger.warning("Semantic index unavailable; falling back to TF-IDF baseline for query.")
            effective_mode = "tfidf"

        num_docs = len(self.paper_metadata)

        # 3. Calculate TF-IDF similarities if needed
        tfidf_sims: Optional[np.ndarray] = None
        if effective_mode in ["tfidf", "hybrid"]:
            query_tfidf_vec = self.vectorizer.transform([clean_query])
            # Sparse matrix-vector dot product: A * v_q^T
            tfidf_sims = self.paper_matrix.dot(query_tfidf_vec.T).toarray().ravel()

        # 4. Calculate Sentence-BERT similarities if needed
        sbert_sims: Optional[np.ndarray] = None
        if effective_mode in ["semantic", "hybrid"] and self.sbert_model is not None and self.sbert_embeddings is not None:
            # Encode user query once (normalized dense vector of size 384)
            query_raw = self.sbert_model.encode(
                [clean_query],
                batch_size=1,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            if hasattr(query_raw, "ndim") and query_raw.ndim > 1:
                query_sbert_vec = np.asarray(query_raw[0], dtype=np.float32)
            else:
                query_sbert_vec = np.asarray(query_raw, dtype=np.float32)

            # Dense matrix-vector dot product: M * v_q (shape: num_docs)
            sbert_sims = self.sbert_embeddings.dot(query_sbert_vec)

        # 5. Handle optional filtering (category and year)
        if category_code or published_year is not None:
            mask = np.ones(num_docs, dtype=bool)
            if category_code:
                target_code = category_code.strip().lower()
                mask = mask & (self.category_codes == target_code)
            if published_year is not None:
                mask = mask & (self.published_years == published_year)

            candidate_indices = np.where(mask)[0]
            if len(candidate_indices) == 0:
                return []
        else:
            candidate_indices = np.arange(num_docs)

        # 6. Compute candidate similarity scores based on effective mode
        if effective_mode == "hybrid" and tfidf_sims is not None and sbert_sims is not None:
            sub_tfidf = tfidf_sims[candidate_indices]
            sub_sbert = sbert_sims[candidate_indices]

            # Normalize distributions before weighted combination
            norm_method = settings.SCORE_NORMALIZATION
            norm_tfidf = self.normalize_scores(sub_tfidf, method=norm_method)
            norm_sbert = self.normalize_scores(sub_sbert, method=norm_method)

            w_tfidf = settings.TFIDF_WEIGHT
            w_sbert = settings.SEMANTIC_WEIGHT

            # Hybrid score formulation
            sub_final = (w_tfidf * norm_tfidf) + (w_sbert * norm_sbert)
        elif effective_mode == "semantic" and sbert_sims is not None:
            sub_final = np.clip(sbert_sims[candidate_indices], 0.0, 1.0)
        else:
            # TF-IDF baseline
            sub_final = tfidf_sims[candidate_indices] if tfidf_sims is not None else np.zeros(len(candidate_indices))

        # 7. Select Top-K candidates using linear-time argpartition
        k = min(top_k, len(candidate_indices))
        if k <= 0:
            return []

        if k >= len(candidate_indices):
            sub_top = np.argsort(-sub_final)
        else:
            partition_idx = np.argpartition(sub_final, -k)[-k:]
            sub_top = partition_idx[np.argsort(-sub_final[partition_idx])]

        top_indices = candidate_indices[sub_top]

        # 8. Hydrate recommendation models
        recommendations: List[PaperRecommendation] = []
        for rank_pos, idx in enumerate(top_indices):
            final_score = float(sub_final[sub_top[rank_pos]])
            t_score = float(tfidf_sims[idx]) if tfidf_sims is not None else None
            s_score = float(sbert_sims[idx]) if sbert_sims is not None else None
            meta = self.paper_metadata[idx]

            recommendations.append(
                PaperRecommendation(
                    id=str(meta["id"]),
                    title=str(meta["title"]),
                    summary=str(meta["summary"]),
                    category=str(meta.get("category", "")),
                    category_code=str(meta.get("category_code", "")),
                    authors=str(meta.get("authors", "")),
                    first_author=str(meta.get("first_author", "")),
                    published_date=str(meta.get("published_date", "")),
                    similarity_score=round(min(max(final_score, 0.0), 1.0), 4),
                    tfidf_score=round(t_score, 4) if t_score is not None else None,
                    semantic_score=round(s_score, 4) if s_score is not None else None,
                    retrieval_mode=effective_mode,
                )
            )

        return recommendations


# Shared singleton instance
_matcher_instance: Optional[PaperMatcher] = None


def get_matcher() -> PaperMatcher:
    """FastAPI dependency to provide shared PaperMatcher instance."""
    global _matcher_instance
    if _matcher_instance is None:
        _matcher_instance = PaperMatcher()
    return _matcher_instance
