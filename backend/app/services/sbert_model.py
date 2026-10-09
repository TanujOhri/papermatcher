"""Sentence-BERT Model Service with Dual Backend (PyTorch & ONNX Runtime).

Provides seamless Sentence-BERT inference using 'sentence-transformers/all-MiniLM-L6-v2'.
Includes automatic fallback to Microsoft ONNX Runtime when PyTorch is unavailable
or restricted by Windows Application Control / Smart App Control policies.
"""

import os
import logging
from typing import List, Union, Optional
import numpy as np

logger = logging.getLogger("papermatcher.sbert")

_ONNX_MODEL_REPO = "sentence-transformers/all-MiniLM-L6-v2"


class ONNXSentenceTransformer:
    """High-performance ONNX Runtime implementation of Sentence-BERT.
    
    Provides 100% compatible API with SentenceTransformer for feature extraction,
    with zero PyTorch binary dependencies and faster CPU inference.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        max_seq_length: int = 128,
        num_threads: Optional[int] = None,
    ):
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self._session = None
        self._tokenizer = None
        self._num_threads = num_threads or min(os.cpu_count() or 4, 8)
        self._init_model()

    def _init_model(self):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        from huggingface_hub import hf_hub_download

        logger.info("Initializing ONNX SentenceTransformer for %s...", self.model_name)
        
        # Download / resolve cached tokenizer and onnx model
        repo_id = self.model_name
        if "/" not in repo_id:
            repo_id = f"sentence-transformers/{self.model_name}"

        tok_path = hf_hub_download(repo_id=repo_id, filename="tokenizer.json")
        onnx_path = hf_hub_download(repo_id=repo_id, filename="onnx/model.onnx")

        self._tokenizer = Tokenizer.from_file(tok_path)
        self._tokenizer.enable_truncation(max_length=self.max_seq_length)
        self._tokenizer.enable_padding(length=self.max_seq_length)

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = self._num_threads
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._session = ort.InferenceSession(onnx_path, sess_options)
        logger.info(
            "ONNX SentenceTransformer initialized successfully (dim: 384, threads: %d).",
            self._num_threads,
        )

    def get_sentence_embedding_dimension(self) -> int:
        return 384

    def encode(
        self,
        sentences: Union[str, List[str]],
        batch_size: int = 128,
        show_progress_bar: bool = False,
        normalize_embeddings: bool = True,
        convert_to_numpy: bool = True,
    ) -> np.ndarray:
        """Encode sentences into normalized 384-dimensional dense vectors."""
        is_single = isinstance(sentences, str)
        text_list = [sentences] if is_single else list(sentences)

        if len(text_list) == 0:
            return np.empty((0, 384), dtype=np.float32)

        # Update tokenizer max_length if changed on instance
        self._tokenizer.enable_truncation(max_length=self.max_seq_length)
        self._tokenizer.enable_padding(length=self.max_seq_length)

        all_embeddings = []
        total = len(text_list)

        for i in range(0, total, batch_size):
            batch_texts = text_list[i : i + batch_size]
            encoded_batch = self._tokenizer.encode_batch(batch_texts)

            input_ids = np.array([e.ids for e in encoded_batch], dtype=np.int64)
            attention_mask = np.array([e.attention_mask for e in encoded_batch], dtype=np.int64)
            token_type_ids = np.array([e.type_ids for e in encoded_batch], dtype=np.int64)

            outputs = self._session.run(
                None,
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "token_type_ids": token_type_ids,
                },
            )
            token_embeddings = outputs[0]  # shape: (batch_size, seq_len, 384)

            # Mean Pooling with attention mask
            input_mask_expanded = np.expand_dims(attention_mask, -1).astype(np.float32)
            sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
            sum_mask = np.clip(input_mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
            pooled = sum_embeddings / sum_mask

            if normalize_embeddings:
                norms = np.linalg.norm(pooled, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                pooled = pooled / norms

            all_embeddings.append(pooled.astype(np.float32))

        result = np.vstack(all_embeddings)
        return result[0] if is_single else result


def get_sentence_transformer(
    model_name: str = "all-MiniLM-L6-v2",
    max_seq_length: int = 128,
    device: Optional[str] = None,
    backend: str = "auto",
):
    """Factory function returning a SentenceTransformer instance with automatic fallback.

    Supports:
    - 'auto': Attempts PyTorch sentence-transformers first, falls back to ONNX Runtime.
    - 'onnx': Uses high-performance ONNX Runtime with multithreaded CPU acceleration.
    - 'torch': Uses PyTorch sentence-transformers.
    """
    if backend == "onnx":
        logger.info("Explicit ONNX backend requested for Sentence-BERT ('%s').", model_name)
        return ONNXSentenceTransformer(
            model_name=model_name,
            max_seq_length=max_seq_length,
        )

    try:
        from sentence_transformers import SentenceTransformer
        logger.info("Attempting to load native PyTorch SentenceTransformer('%s')...", model_name)
        model = SentenceTransformer(model_name, device=device)
        model.max_seq_length = max_seq_length
        return model
    except (ImportError, OSError, Exception) as exc:
        if backend == "torch":
            raise
        logger.warning(
            "Native PyTorch SentenceTransformer unavailable (%s). "
            "Falling back to high-performance ONNX Runtime backend for Sentence-BERT.",
            exc,
        )
        return ONNXSentenceTransformer(
            model_name=model_name,
            max_seq_length=max_seq_length,
        )
