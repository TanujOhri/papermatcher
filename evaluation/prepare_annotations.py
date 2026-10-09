"""Generate unjudged retrieval candidates for human relevance annotation."""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
QUERIES_PATH = CURRENT_DIR / "queries.json"
OUTPUT_PATH = CURRENT_DIR / "annotation_candidates.json"
TOP_K = 5

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.services.matcher import get_matcher


ANNOTATION_INSTRUCTIONS = [
    "Review each paper's title and summary against the query's information need and mark relevance true only when the paper is topically relevant.",
    "Mark relevance false when the paper is not topically relevant; leave it null if you have not judged it or are unsure.",
    "Do not infer relevance from ranking position, retrieval mode, or score.",
    "Use the same relevance judgment for a paper ID across all retrieval modes when comparing their rankings.",
]


def generate_candidates() -> Dict[str, Any]:
    if OUTPUT_PATH.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing annotation file: {OUTPUT_PATH}"
        )

    with QUERIES_PATH.open("r", encoding="utf-8") as queries_file:
        queries = json.load(queries_file)
    if not isinstance(queries, list):
        raise ValueError("The queries file must contain a JSON array.")

    matcher = get_matcher()
    if not matcher.load_index(auto_build=False, load_semantic=True):
        raise RuntimeError("TF-IDF index is unavailable; no candidates were generated.")

    included_modes = ["tfidf"]
    unavailable_modes: Dict[str, str] = {}
    if matcher.has_semantic:
        included_modes.extend(["semantic", "hybrid"])
    else:
        unavailable_modes["semantic"] = "Semantic model or embedding index did not load."
        unavailable_modes["hybrid"] = (
            "Hybrid requires the semantic index; it was skipped to avoid the matcher's TF-IDF fallback."
        )

    errors: List[Dict[str, Any]] = []
    query_results: List[Dict[str, Any]] = []

    for query_index, query_entry in enumerate(queries, start=1):
        query_text = query_entry["query"]
        category_code = query_entry.get("category_code")
        candidates_by_id: Dict[str, Dict[str, Any]] = {}

        for mode in included_modes:
            try:
                recommendations = matcher.recommend(
                    query=query_text,
                    top_k=TOP_K,
                    category_code=category_code,
                    retrieval_mode=mode,
                )
            except Exception as error:
                error_entry = {
                    "query_index": query_index,
                    "query_text": query_text,
                    "retrieval_mode": mode,
                    "error": str(error),
                }
                errors.append(error_entry)
                print(
                    f"Error for query {query_index} in {mode}: {error}",
                    file=sys.stderr,
                )
                continue

            for rank, recommendation in enumerate(recommendations, start=1):
                paper_id = str(recommendation.id)
                candidate = candidates_by_id.get(paper_id)
                if candidate is None:
                    candidate = {
                        "paper_id": paper_id,
                        "title": recommendation.title,
                        "summary": recommendation.summary,
                        "query_text": query_text,
                        "query_category_code": category_code,
                        "paper_category": recommendation.category,
                        "paper_category_code": recommendation.category_code,
                        "relevance": None,
                        "annotation_status": "unjudged",
                        "rankings": {},
                    }
                    candidates_by_id[paper_id] = candidate

                candidate["rankings"][mode] = {
                    "rank": rank,
                    "similarity_score": recommendation.similarity_score,
                    "tfidf_score": recommendation.tfidf_score,
                    "semantic_score": recommendation.semantic_score,
                }

        query_results.append(
            {
                "query_index": query_index,
                "query_text": query_text,
                "category_code": category_code,
                "candidates": list(candidates_by_id.values()),
            }
        )

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "top_k_per_mode": TOP_K,
        "included_modes": included_modes,
        "unavailable_modes": unavailable_modes,
        "annotation_instructions": ANNOTATION_INSTRUCTIONS,
        "queries": query_results,
        "errors": errors,
    }


def main() -> int:
    try:
        payload = generate_candidates()
        with OUTPUT_PATH.open("x", encoding="utf-8") as output_file:
            json.dump(payload, output_file, indent=2, ensure_ascii=False)
            output_file.write("\n")
    except Exception as error:
        print(f"Failed to prepare annotation candidates: {error}", file=sys.stderr)
        return 1

    candidate_count = sum(len(query["candidates"]) for query in payload["queries"])
    print(f"Wrote {candidate_count} unique candidates to {OUTPUT_PATH}")
    print(f"Queries: {len(payload['queries'])}")
    print(f"Modes included: {', '.join(payload['included_modes'])}")
    if payload["unavailable_modes"]:
        print(f"Modes unavailable: {', '.join(payload['unavailable_modes'])}")
    if payload["errors"]:
        print(f"Retrieval errors: {len(payload['errors'])}; see output JSON.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())