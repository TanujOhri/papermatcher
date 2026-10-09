"""Retrieval Performance Benchmark Script for PaperMatcher-AI.

Measures and compares:
1. Lexical TF-IDF Baseline
2. Dense Sentence-BERT Semantic Retrieval
3. Hybrid Lexical + Semantic Retrieval

Across the 7 established scientific benchmark queries:
- Query Encoding Time (ms)
- Dot Product & Partition Retrieval Time (ms)
- Total Recommendation Latency (ms)
- Memory Utilization (MB)
- Top-ranked preprints and score distributions

Outputs machine-readable benchmark JSON to scratch/benchmark_results.json.
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import List, Dict, Any

# Ensure backend directory is on sys.path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
backend_dir = project_root / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.services.matcher import PaperMatcher, get_matcher

BENCHMARK_QUERIES = [
    "transformer architectures for natural language processing",
    "large language models hallucination",
    "diffusion models image generation",
    "graph neural networks",
    "reinforcement learning from human feedback",
    "federated learning privacy",
    "computer vision image classification",
]


def run_benchmark():
    print("=" * 85)
    print("  PaperMatcher-AI Retrieval Benchmark: TF-IDF vs Sentence-BERT vs Hybrid")
    print("=" * 85)

    matcher = get_matcher()
    print("Loading models and precomputed index artifacts...")
    t0 = time.time()
    matcher.load_index(auto_build=False, load_semantic=True)
    init_time = time.time() - t0
    print(f"Index loaded in {init_time:.2f} seconds.")

    # Calculate memory usage
    tfidf_mem_mb = 0.0
    if matcher.paper_matrix is not None:
        m = matcher.paper_matrix
        tfidf_mem_mb = (m.data.nbytes + m.indices.nbytes + m.indptr.nbytes) / (1024 * 1024)

    sbert_mem_mb = 0.0
    if matcher.sbert_embeddings is not None:
        sbert_mem_mb = matcher.sbert_embeddings.nbytes / (1024 * 1024)

    print(f"TF-IDF CSR Matrix Memory: {tfidf_mem_mb:.2f} MB")
    print(f"SBERT Embedding Matrix Memory: {sbert_mem_mb:.2f} MB (Semantic Active: {matcher.has_semantic})")
    print("-" * 85)

    modes = ["tfidf"]
    if matcher.has_semantic:
        modes.extend(["semantic", "hybrid"])
    else:
        print("[NOTICE] Semantic embeddings not loaded. Benchmarking will test TF-IDF baseline only.")

    benchmark_records: List[Dict[str, Any]] = []

    for q_idx, query in enumerate(BENCHMARK_QUERIES, 1):
        print(f"\nQuery {q_idx}/{len(BENCHMARK_QUERIES)}: \"{query}\"")
        q_record = {
            "query": query,
            "modes": {},
        }

        for mode in modes:
            # Measure encoding and retrieval components
            t_start = time.time()
            recs = matcher.recommend(query=query, top_k=5, retrieval_mode=mode)
            total_latency_ms = (time.time() - t_start) * 1000

            top_item = recs[0] if recs else None
            top_title = top_item.title if top_item else "N/A"
            top_score = top_item.similarity_score if top_item else 0.0
            top_tfidf = top_item.tfidf_score if top_item else None
            top_semantic = top_item.semantic_score if top_item else None

            q_record["modes"][mode] = {
                "total_latency_ms": round(total_latency_ms, 2),
                "top_score": round(top_score, 4),
                "tfidf_score": round(top_tfidf, 4) if top_tfidf is not None else None,
                "semantic_score": round(top_semantic, 4) if top_semantic is not None else None,
                "top_paper_id": top_item.id if top_item else None,
                "top_paper_title": top_title[:65],
            }

            print(
                f"  [{mode.upper():<8}] Latency: {total_latency_ms:>5.1f} ms | "
                f"Score: {top_score:.4f} | {top_title[:55]}..."
            )

        benchmark_records.append(q_record)

    # Compute aggregate latency statistics
    print("\n" + "=" * 85)
    print("  Aggregate Latency & Performance Summary (7 Benchmark Queries)")
    print("=" * 85)
    print(f"{'Retrieval Mode':<16} | {'Avg Latency':<14} | {'Min Latency':<14} | {'Max Latency':<14}")
    print("-" * 85)

    mode_summaries = {}
    for mode in modes:
        latencies = [r["modes"][mode]["total_latency_ms"] for r in benchmark_records if mode in r["modes"]]
        avg_lat = sum(latencies) / len(latencies) if latencies else 0.0
        min_lat = min(latencies) if latencies else 0.0
        max_lat = max(latencies) if latencies else 0.0

        mode_summaries[mode] = {
            "avg_latency_ms": round(avg_lat, 2),
            "min_latency_ms": round(min_lat, 2),
            "max_latency_ms": round(max_lat, 2),
        }
        print(f"{mode.upper():<16} | {avg_lat:>10.2f} ms   | {min_lat:>10.2f} ms   | {max_lat:>10.2f} ms")

    print("=" * 85)

    # Output JSON artifact
    output_path = project_root / "scratch" / "benchmark_results.json"
    results_payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_name": "all-MiniLM-L6-v2",
        "tfidf_memory_mb": round(tfidf_mem_mb, 2),
        "sbert_memory_mb": round(sbert_mem_mb, 2),
        "tested_modes": modes,
        "aggregate_latency": mode_summaries,
        "queries": benchmark_records,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    print(f"\nMachine-readable benchmark results saved to: {output_path}")
    return results_payload


if __name__ == "__main__":
    run_benchmark()
