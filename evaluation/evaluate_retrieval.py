"""Information Retrieval Evaluation Framework for PaperMatcher-AI.

Evaluates retrieval quality across:
1. TF-IDF Lexical Retrieval
2. Sentence-BERT Semantic Retrieval
3. Hybrid Retrieval

Standard IR Metrics:
- Precision@K
- Recall@K
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (nDCG@K)

Zero-Hallucination Compliance:
If queries in queries.json lack human ground-truth relevance annotations,
the script explicitly reports that metrics cannot be computed without verified labels.
"""

import os
import sys
import json
import math
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

# Ensure backend is on sys.path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
backend_dir = project_root / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.services.matcher import PaperMatcher, get_matcher


def compute_precision_at_k(retrieved_ids: List[str], relevant_ids: set, k: int) -> float:
    """Compute Precision@K."""
    if k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for pid in top_k if pid in relevant_ids)
    return hits / k


def compute_recall_at_k(retrieved_ids: List[str], relevant_ids: set, k: int) -> float:
    """Compute Recall@K."""
    if not relevant_ids or k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for pid in top_k if pid in relevant_ids)
    return hits / len(relevant_ids)


def compute_reciprocal_rank(retrieved_ids: List[str], relevant_ids: set) -> float:
    """Compute Reciprocal Rank (1 / rank of first relevant result)."""
    for rank, pid in enumerate(retrieved_ids, 1):
        if pid in relevant_ids:
            return 1.0 / rank
    return 0.0


def compute_ndcg_at_k(retrieved_ids: List[str], relevant_ids: set, k: int) -> float:
    """Compute nDCG@K with binary relevance."""
    if not relevant_ids or k <= 0:
        return 0.0

    top_k = retrieved_ids[:k]
    dcg = 0.0
    for rank, pid in enumerate(top_k, 1):
        rel = 1 if pid in relevant_ids else 0
        if rel > 0:
            dcg += rel / math.log2(rank + 1)

    # Ideal DCG
    ideal_hits = min(k, len(relevant_ids))
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    if idcg <= 0.0:
        return 0.0

    return dcg / idcg


def evaluate_queries(
    queries_file: Path,
    top_k: int = 5,
    output_json: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run evaluation across all annotated queries in queries_file."""
    print("=" * 70)
    print("  PaperMatcher-AI Information Retrieval Evaluation")
    print("=" * 70)

    if not queries_file.exists():
        print(f"Error: queries file not found at {queries_file}")
        sys.exit(1)

    with open(queries_file, "r", encoding="utf-8") as f:
        queries_data = json.load(f)

    annotated_queries = [q for q in queries_data if len(q.get("relevant_paper_ids", [])) > 0]
    total_queries = len(queries_data)
    annotated_count = len(annotated_queries)

    print(f"Total queries in dataset: {total_queries}")
    print(f"Annotated queries with relevance labels: {annotated_count}")

    if annotated_count == 0:
        print("\n[NOTICE: ZERO-HALLUCINATION ENFORCEMENT]")
        print("No queries in queries.json currently contain verified ground-truth relevance labels.")
        print("To compute empirical IR metrics (Precision@K, Recall@K, MRR, nDCG@K):")
        print("1. Open evaluation/queries.json.")
        print("2. Add known relevant paper IDs to 'relevant_paper_ids' for your evaluation queries.")
        print("3. Re-run this script.\n")
        return {
            "status": "unannotated",
            "message": "No labeled ground-truth queries found. Evaluation cannot be performed without labels.",
            "annotated_queries": 0,
        }

    # Initialize matcher
    print("\nInitializing PaperMatcher engine...")
    matcher = get_matcher()
    matcher.load_index(auto_build=False, load_semantic=True)

    available_modes = ["tfidf"]
    if matcher.has_semantic:
        available_modes.extend(["semantic", "hybrid"])
    else:
        print("Warning: Semantic index not loaded. Only TF-IDF baseline can be evaluated.")

    results_by_mode: Dict[str, Dict[str, float]] = {}

    for mode in available_modes:
        precisions, recalls, mrrs, ndcgs = [], [], [], []

        for q_entry in annotated_queries:
            q_text = q_entry["query"]
            cat_code = q_entry.get("category_code")
            relevant = set(q_entry["relevant_paper_ids"])

            try:
                recs = matcher.recommend(
                    query=q_text,
                    top_k=top_k,
                    category_code=cat_code,
                    retrieval_mode=mode,
                )
                retrieved_ids = [r.id for r in recs]
            except Exception as e:
                print(f"Error executing query '{q_text}' in mode '{mode}': {e}")
                retrieved_ids = []

            precisions.append(compute_precision_at_k(retrieved_ids, relevant, top_k))
            recalls.append(compute_recall_at_k(retrieved_ids, relevant, top_k))
            mrrs.append(compute_reciprocal_rank(retrieved_ids, relevant))
            ndcgs.append(compute_ndcg_at_k(retrieved_ids, relevant, top_k))

        results_by_mode[mode] = {
            f"Precision@{top_k}": sum(precisions) / len(precisions),
            f"Recall@{top_k}": sum(recalls) / len(recalls),
            "MRR": sum(mrrs) / len(mrrs),
            f"nDCG@{top_k}": sum(ndcgs) / len(ndcgs),
        }

    # Print summary table
    print("\n" + "=" * 70)
    print(f"  Evaluation Results (K={top_k}, Annotated Queries={annotated_count})")
    print("=" * 70)
    print(f"{'Metric':<18} | " + " | ".join(f"{m.upper():<12}" for m in available_modes))
    print("-" * 70)

    metric_keys = [f"Precision@{top_k}", f"Recall@{top_k}", "MRR", f"nDCG@{top_k}"]
    for m_key in metric_keys:
        row = f"{m_key:<18} | "
        for mode in available_modes:
            val = results_by_mode[mode][m_key]
            row += f"{val:<12.4f} | "
        print(row)
    print("=" * 70)

    if output_json:
        output_payload = {
            "top_k": top_k,
            "annotated_queries": annotated_count,
            "total_queries": total_queries,
            "results": results_by_mode,
        }
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(output_payload, f, indent=2)
        print(f"\nSaved evaluation metrics to {output_json}")

    return results_by_mode


def main():
    parser = argparse.ArgumentParser(description="Evaluate PaperMatcher-AI retrieval models.")
    parser.add_argument(
        "--queries-file",
        type=Path,
        default=project_root / "evaluation" / "queries.json",
        help="Path to queries JSON file.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Top-K cutoff.")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=project_root / "evaluation" / "evaluation_results.json",
        help="Optional output JSON path.",
    )
    args = parser.parse_args()

    evaluate_queries(
        queries_file=args.queries_file,
        top_k=args.top_k,
        output_json=args.output_json,
    )


if __name__ == "__main__":
    main()
