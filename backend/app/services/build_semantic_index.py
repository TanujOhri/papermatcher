"""Sentence-BERT semantic embedding index builder for PaperMatcher-AI.

Generates and persists dense normalized vector representations for all verified
arXiv preprints using sentence-transformers (all-MiniLM-L6-v2).

Features:
- Incremental chunk-based encoding with resume capability.
- Exact 1-to-1 ordering alignment with models/paper_metadata.joblib.
- Unit L2 normalization for fast dot-product cosine similarity.
- Strict post-build validation of embedding count, dimensions, and norms.
- Metadata manifest written to models/sbert_meta.json.
"""

import os
import sys
import time
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List

import joblib
import numpy as np

# Ensure backend directory is on sys.path
current_dir = Path(__file__).resolve().parent
backend_dir = current_dir.parent.parent
project_root = backend_dir.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.core.config import settings
from app.services.dataset import DatasetLoader
from app.services.sbert_model import get_sentence_transformer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("papermatcher.semantic_indexer")


def get_models_dir() -> Path:
    """Return models directory path."""
    return settings.get_models_dir()


def load_corpus_metadata() -> List[Dict[str, Any]]:
    """Load or generate verified paper metadata matching TF-IDF index."""
    models_dir = get_models_dir()
    meta_path = models_dir / "paper_metadata.joblib"

    if meta_path.exists():
        logger.info("Loading existing paper metadata from %s...", meta_path)
        metadata = joblib.load(meta_path)
        logger.info("Loaded %d paper metadata records.", len(metadata))
        return metadata

    logger.info("paper_metadata.joblib not found; ingesting via DatasetLoader...")
    loader = DatasetLoader()
    df = loader.load_dataset()
    metadata_cols = [
        "id",
        "title",
        "summary",
        "category",
        "category_code",
        "authors",
        "first_author",
        "published_date",
    ]
    metadata = df[metadata_cols].to_dict(orient="records")
    joblib.dump(metadata, meta_path, compress=3)
    logger.info("Saved %d metadata records to %s.", len(metadata), meta_path)
    return metadata


def build_semantic_index(
    model_name: Optional[str] = None,
    batch_size: int = 128,
    max_seq_length: int = 128,
    chunk_size: int = 5000,
    max_docs: Optional[int] = None,
    force_rebuild: bool = False,
    output_filename: str = "sbert_embeddings.npy",
    backend: str = "auto",
) -> Dict[str, Any]:
    """Encode all corpus documents using Sentence-BERT and save normalized embeddings.

    Args:
        model_name: Hugging Face model identifier (default: all-MiniLM-L6-v2).
        batch_size: Number of texts per forward pass.
        max_seq_length: Maximum token sequence length for truncation.
        chunk_size: Checkpoint frequency (number of texts per saved chunk).
        max_docs: Optional document limit (for fast testing / verification).
        force_rebuild: If True, ignores existing chunks and re-encodes.
        output_filename: Filename for the final consolidated .npy file.

    Returns:
        Dict with build statistics and validation metrics.
    """
    model_name = model_name or settings.SBERT_MODEL_NAME
    models_dir = get_models_dir()
    final_output_path = models_dir / output_filename
    chunks_dir = models_dir / "sbert_chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    start_total_time = time.time()
    logger.info("=== Starting PaperMatcher-AI Semantic Index Build ===")
    logger.info("Model: %s | Max Sequence Length: %d | Batch Size: %d", model_name, max_seq_length, batch_size)

    # 1. Load ground truth metadata
    metadata = load_corpus_metadata()
    if max_docs is not None and max_docs > 0:
        logger.warning("Restricting build to first %d documents (--max-docs specified)", max_docs)
        metadata = metadata[:max_docs]

    total_docs = len(metadata)
    logger.info("Total documents to encode: %d", total_docs)

    # 2. Prepare text representations (title + " " + summary)
    # Strictly matches TF-IDF text representation
    logger.info("Synthesizing document text representations (title + summary)...")
    texts = [f"{m.get('title', '')} {m.get('summary', '')}".strip() for m in metadata]

    # 3. Initialize Sentence-BERT model
    logger.info("Loading Sentence-BERT model '%s' (backend: %s)...", model_name, backend)
    t_model_load = time.time()
    model = get_sentence_transformer(
        model_name=model_name,
        max_seq_length=max_seq_length,
        backend=backend,
    )
    logger.info("Model loaded in %.2f seconds.", time.time() - t_model_load)

    # 4. Chunked Encoding with Resumption
    chunk_files: List[Path] = []
    num_chunks = (total_docs + chunk_size - 1) // chunk_size

    for chunk_idx in range(num_chunks):
        c_start = chunk_idx * chunk_size
        c_end = min(c_start + chunk_size, total_docs)
        chunk_file = chunks_dir / f"chunk_{c_start:06d}_{c_end:06d}.npy"
        chunk_files.append(chunk_file)

        if chunk_file.exists() and not force_rebuild:
            logger.info(
                "Chunk %d/%d (%d to %d) already exists. Skipping encoding.",
                chunk_idx + 1,
                num_chunks,
                c_start,
                c_end,
            )
            continue

        chunk_texts = texts[c_start:c_end]
        logger.info(
            "Encoding chunk %d/%d (docs %d to %d, count: %d)...",
            chunk_idx + 1,
            num_chunks,
            c_start,
            c_end,
            len(chunk_texts),
        )
        t_chunk_start = time.time()

        chunk_embs = model.encode(
            chunk_texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        np.save(chunk_file, chunk_embs)
        dur = time.time() - t_chunk_start
        throughput = len(chunk_texts) / max(dur, 0.001)
        logger.info(
            "Chunk %d/%d encoded and saved in %.2fs (%.1f docs/sec).",
            chunk_idx + 1,
            num_chunks,
            dur,
            throughput,
        )

    # 5. Consolidate Chunks
    logger.info("Consolidating %d chunks into final matrix...", len(chunk_files))
    all_chunks = []
    for cf in chunk_files:
        if not cf.exists():
            raise FileNotFoundError(f"Missing expected chunk file: {cf}")
        all_chunks.append(np.load(cf))

    full_embeddings = np.vstack(all_chunks).astype(np.float32)

    # 6. Strict Validation
    logger.info("Validating final embedding matrix...")
    if full_embeddings.shape[0] != total_docs:
        raise ValueError(
            f"Embedding count mismatch: generated {full_embeddings.shape[0]}, "
            f"expected {total_docs} matching metadata records."
        )

    embedding_dim = full_embeddings.shape[1]
    if embedding_dim != 384:
        logger.warning("Unexpected embedding dimension: %d (expected 384 for all-MiniLM-L6-v2)", embedding_dim)

    # Check normalization: each row should have Euclidean norm ~ 1.0
    sample_norms = np.linalg.norm(full_embeddings[: min(1000, total_docs)], axis=1)
    if not np.allclose(sample_norms, 1.0, atol=1e-3):
        logger.warning("Embeddings are not unit normalized! Re-normalizing...")
        norms = np.linalg.norm(full_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        full_embeddings = full_embeddings / norms

    # 7. Persist Final Artifact
    logger.info("Saving full embedding matrix to %s...", final_output_path)
    np.save(final_output_path, full_embeddings)
    file_size_mb = os.path.getsize(final_output_path) / (1024 * 1024)

    # 8. Write Manifest Metadata
    manifest_path = models_dir / "sbert_meta.json"
    manifest = {
        "model_name": model_name,
        "max_seq_length": max_seq_length,
        "num_embeddings": int(full_embeddings.shape[0]),
        "dimensions": int(embedding_dim),
        "dtype": str(full_embeddings.dtype),
        "normalized": True,
        "file_size_mb": round(file_size_mb, 2),
        "output_filename": output_filename,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    total_time = time.time() - start_total_time
    logger.info("=== Semantic Index Build Complete ===")
    logger.info("Total Time: %.2f seconds", total_time)
    logger.info("Embeddings Shape: %s", full_embeddings.shape)
    logger.info("Storage Size: %.2f MB (%s)", file_size_mb, final_output_path.name)
    logger.info("Manifest: %s", manifest_path.name)

    return {
        "total_time_seconds": total_time,
        "num_embeddings": full_embeddings.shape[0],
        "dimensions": embedding_dim,
        "file_size_mb": file_size_mb,
        "manifest": manifest,
    }


def main():
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Generate Sentence-BERT embeddings for PaperMatcher-AI.")
    parser.add_argument("--model-name", type=str, default="all-MiniLM-L6-v2", help="HuggingFace model name.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for model encoding.")
    parser.add_argument("--max-seq-length", type=int, default=128, help="Max token sequence length.")
    parser.add_argument("--chunk-size", type=int, default=5000, help="Number of documents per saved checkpoint.")
    parser.add_argument("--max-docs", type=int, default=None, help="Optional limit for rapid testing.")
    parser.add_argument("--force", action="store_true", help="Force rebuild existing chunks.")
    parser.add_argument("--output-filename", type=str, default="sbert_embeddings.npy", help="Output .npy filename.")
    parser.add_argument("--backend", type=str, default="auto", choices=["auto", "onnx", "torch"], help="Inference backend engine.")

    args = parser.parse_args()
    build_semantic_index(
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        chunk_size=args.chunk_size,
        max_docs=args.max_docs,
        force_rebuild=args.force,
        output_filename=args.output_filename,
        backend=args.backend,
    )


if __name__ == "__main__":
    main()
