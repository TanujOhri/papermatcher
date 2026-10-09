"""Academic Verification Script for PaperMatcher-AI.

Generates precise metrics, scores (raw and normalized), latency breakdowns,
and artifact inspection for the 7 canonical benchmark queries.
"""

import sys
import time
import json
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import joblib

# Add backend directory
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
backend_dir = project_root / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.services.sbert_model import get_sentence_transformer
from app.services.dataset import DatasetLoader
from app.services.matcher import PaperMatcher

BENCHMARK_QUERIES = [
    "transformer architectures for natural language processing",
    "large language models hallucination",
    "diffusion models image generation",
    "graph neural networks",
    "reinforcement learning from human feedback",
    "federated learning privacy",
    "computer vision image classification",
]

def run_academic_verification():
    models_dir = project_root / "models"
    
    # 1. Inspect Artifacts
    meta_path = models_dir / "paper_metadata.joblib"
    tfidf_matrix_path = models_dir / "tfidf_matrix.npz"
    tfidf_vec_path = models_dir / "tfidf_vectorizer.joblib"
    prod_sbert_path = models_dir / "sbert_embeddings.npy"
    test_sbert_path = models_dir / "test_sbert_5000.npy"
    
    metadata = joblib.load(meta_path)
    tfidf_matrix = sp.load_npz(tfidf_matrix_path)
    vectorizer = joblib.load(tfidf_vec_path)
    
    total_metadata_count = len(metadata)
    prod_sbert_exists = prod_sbert_path.exists()
    prod_sbert_count = 0
    if prod_sbert_exists:
        prod_embs = np.load(prod_sbert_path, mmap_mode="r")
        prod_sbert_count = prod_embs.shape[0]

    test_sbert_exists = test_sbert_path.exists()
    test_sbert_count = 0
    if test_sbert_exists:
        test_embs = np.load(test_sbert_path)
        test_sbert_count = test_embs.shape[0]
        
    print(f"Metadata Records: {total_metadata_count}")
    print(f"TF-IDF CSR Shape: {tfidf_matrix.shape}")
    print(f"Production SBERT Exists: {prod_sbert_exists} (Count: {prod_sbert_count})")
    print(f"Test SBERT Exists: {test_sbert_exists} (Count: {test_sbert_count})")
    
    # 2. Load Model & Prepare Engine
    loader = DatasetLoader()
    sbert_model = get_sentence_transformer(model_name="all-MiniLM-L6-v2", max_seq_length=128)
    
    # Use test_embs for dense calculations if prod not present
    active_embs = test_embs if test_sbert_exists else None
    active_meta = metadata[:len(active_embs)] if active_embs is not None else metadata
    active_tfidf = tfidf_matrix[:len(active_meta)] if active_embs is not None else tfidf_matrix

    print("\n" + "="*90)
    print("  DETAILED QUERY-BY-QUERY VERIFICATION (Top-5 Results & Score Breakdown)")
    print("="*90)
    
    query_results = []
    
    latency_records = {
        "tfidf": [],
        "sbert_encode": [],
        "sbert_retrieve": [],
        "hybrid": []
    }
    
    w_tfidf = 0.40
    w_sbert = 0.60
    
    for q_idx, query in enumerate(BENCHMARK_QUERIES, 1):
        clean_q = loader.preprocess_text(query)
        
        # 1. TF-IDF timing & retrieval (on full 287k corpus)
        t_tfidf_start = time.time()
        q_tfidf_vec = vectorizer.transform([clean_q])
        tfidf_scores_full = tfidf_matrix.dot(q_tfidf_vec.T).toarray().ravel()
        t_tfidf_lat = (time.time() - t_tfidf_start) * 1000
        latency_records["tfidf"].append(t_tfidf_lat)
        
        # Top-5 TF-IDF on full corpus
        top5_tfidf_idx = np.argsort(-tfidf_scores_full)[:5]
        
        # 2. SBERT query encoding timing
        t_enc_start = time.time()
        q_sbert_vec = sbert_model.encode([clean_q], normalize_embeddings=True)
        if hasattr(q_sbert_vec, 'ndim') and q_sbert_vec.ndim > 1:
            q_sbert_vec = q_sbert_vec[0]
        q_sbert_vec = np.asarray(q_sbert_vec, dtype=np.float32)
        t_sbert_encode = (time.time() - t_enc_start) * 1000
        latency_records["sbert_encode"].append(t_sbert_encode)
        
        # 3. SBERT dense retrieval timing (on active indexed dense subset)
        t_sbert_ret_start = time.time()
        if active_embs is not None:
            sbert_scores = active_embs.dot(q_sbert_vec)
        else:
            sbert_scores = np.zeros(len(active_meta), dtype=np.float32)
        t_sbert_retrieve = (time.time() - t_sbert_ret_start) * 1000
        latency_records["sbert_retrieve"].append(t_sbert_retrieve)
        
        # Top-5 SBERT on active corpus
        top5_sbert_idx = np.argsort(-sbert_scores)[:5]
        
        # 4. Hybrid Scoring & timing
        t_hybrid_start = time.time()
        # Transform active TF-IDF
        sub_tfidf = active_tfidf.dot(q_tfidf_vec.T).toarray().ravel()
        sub_sbert = sbert_scores
        
        # Min-max normalization
        norm_tfidf = PaperMatcher.normalize_scores(sub_tfidf, method="minmax")
        norm_sbert = PaperMatcher.normalize_scores(sub_sbert, method="minmax")
        hybrid_scores = (w_tfidf * norm_tfidf) + (w_sbert * norm_sbert)
        top5_hybrid_idx = np.argsort(-hybrid_scores)[:5]
        t_hybrid_lat = (time.time() - t_hybrid_start) * 1000 + t_sbert_encode
        latency_records["hybrid"].append(t_hybrid_lat)
        
        print(f"\n[QUERY {q_idx}] \"{query}\"")
        print(f"Latency -> TF-IDF: {t_tfidf_lat:.2f}ms | SBERT Encode: {t_sbert_encode:.2f}ms | SBERT Retrieval: {t_sbert_retrieve:.2f}ms | Hybrid: {t_hybrid_lat:.2f}ms")
        
        print("\n  --- Top-5 TF-IDF Results (Full 287,383 Corpus) ---")
        for rank, idx in enumerate(top5_tfidf_idx, 1):
            m = metadata[idx]
            raw_t = tfidf_scores_full[idx]
            print(f"  {rank}. [{m.get('id')}] (Score: {raw_t:.4f}) {m.get('title')[:60]}... ({m.get('category_code')})")
            
        print(f"\n  --- Top-5 SBERT Results (Indexed Subset: {len(active_meta)}) ---")
        for rank, idx in enumerate(top5_sbert_idx, 1):
            m = active_meta[idx]
            raw_s = sbert_scores[idx]
            print(f"  {rank}. [{m.get('id')}] (Cosine: {raw_s:.4f}) {m.get('title')[:60]}... ({m.get('category_code')})")
            
        print(f"\n  --- Top-5 Hybrid Results (Fused Subset: {len(active_meta)}) ---")
        for rank, idx in enumerate(top5_hybrid_idx, 1):
            m = active_meta[idx]
            raw_t = sub_tfidf[idx]
            raw_s = sub_sbert[idx]
            n_t = norm_tfidf[idx]
            n_s = norm_sbert[idx]
            h_score = hybrid_scores[idx]
            print(f"  {rank}. [{m.get('id')}] Hybrid: {h_score:.4f} | Raw TF-IDF: {raw_t:.4f} (Norm: {n_t:.4f}) | Raw SBERT: {raw_s:.4f} (Norm: {n_s:.4f})")
            print(f"      Title: {m.get('title')[:65]}...")
            
        query_results.append({
            "query": query,
            "top5_tfidf_full": [
                {"id": metadata[idx]["id"], "title": metadata[idx]["title"], "score": float(tfidf_scores_full[idx])}
                for idx in top5_tfidf_idx
            ],
            "top5_sbert": [
                {"id": active_meta[idx]["id"], "title": active_meta[idx]["title"], "score": float(sbert_scores[idx])}
                for idx in top5_sbert_idx
            ],
            "top5_hybrid": [
                {
                    "id": active_meta[idx]["id"],
                    "title": active_meta[idx]["title"],
                    "raw_tfidf": float(sub_tfidf[idx]),
                    "raw_sbert": float(sub_sbert[idx]),
                    "norm_tfidf": float(norm_tfidf[idx]),
                    "norm_sbert": float(norm_sbert[idx]),
                    "hybrid_score": float(hybrid_scores[idx]),
                }
                for idx in top5_hybrid_idx
            ]
        })

    # Summary table
    print("\n" + "="*90)
    print("  LATENCY SUMMARY (Averaged over 7 Canonical Queries)")
    print("="*90)
    avg_tfidf = np.mean(latency_records["tfidf"])
    avg_encode = np.mean(latency_records["sbert_encode"])
    avg_sbert_ret = np.mean(latency_records["sbert_retrieve"])
    avg_hybrid = np.mean(latency_records["hybrid"])
    
    print(f"TF-IDF Full Search Latency:      {avg_tfidf:.2f} ms")
    print(f"SBERT Query Encoding Latency:    {avg_encode:.2f} ms")
    print(f"SBERT Retrieval Dot Product:     {avg_sbert_ret:.2f} ms")
    print(f"Hybrid Total Pipeline Latency:   {avg_hybrid:.2f} ms")
    
    # Save verification JSON
    res_path = project_root / "scratch" / "academic_verification_report.json"
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_metadata_records": total_metadata_count,
            "production_sbert_index_complete": prod_sbert_exists and prod_sbert_count == total_metadata_count,
            "production_sbert_count": prod_sbert_count,
            "test_sbert_count": test_sbert_count,
            "avg_latencies_ms": {
                "tfidf": round(avg_tfidf, 2),
                "sbert_query_encoding": round(avg_encode, 2),
                "sbert_retrieval_dot_product": round(avg_sbert_ret, 2),
                "hybrid_total": round(avg_hybrid, 2),
            },
            "queries": query_results
        }, f, indent=2)
    print(f"\nVerification report written to {res_path}")

if __name__ == "__main__":
    run_academic_verification()
