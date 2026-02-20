from __future__ import annotations

import argparse
import importlib
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from services.catalog_loader import ensure_catalog
import services.text_index as text_index_module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate text retrieval quality for the commerce agent.")
    parser.add_argument(
        "--queries",
        type=Path,
        default=Path(__file__).resolve().with_name("eval_queries.json"),
        help="Path to labeled query file (JSON list).",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Top-K retrieval depth for evaluation.")
    parser.add_argument(
        "--regenerate-catalog",
        action="store_true",
        help="Regenerate catalog from backend/data before indexing.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional path to write full evaluation report as JSON.",
    )
    parser.add_argument(
        "--force-tfidf",
        action="store_true",
        help="Force TF-IDF backend for evaluation (offline-safe).",
    )
    return parser.parse_args()


def load_queries(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Query file not found: {path}")
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError("Query file must be a JSON list.")
    return data


def normalize_set(values: List[str]) -> set[str]:
    return {str(v).strip().lower() for v in values if str(v).strip()}


def is_relevant(item: Dict[str, Any], expected_ids: set[str], expected_categories: set[str]) -> bool:
    item_id = str(item.get("id", "")).strip().lower()
    item_category = str(item.get("category", "")).strip().lower()
    if expected_ids:
        return item_id in expected_ids
    if expected_categories:
        return item_category in expected_categories
    return False


def evaluate_query(index: Any, query_def: Dict[str, Any], top_k: int) -> Dict[str, Any]:
    name = str(query_def.get("name", query_def.get("query", "unnamed-query")))
    query_text = str(query_def.get("query", "")).strip()
    filters = query_def.get("filters") or {}
    if not isinstance(filters, dict):
        filters = {}

    expected_ids = normalize_set(list(query_def.get("expected_ids", [])))
    expected_categories = normalize_set(list(query_def.get("expected_categories", [])))

    results = index.search_with_filters(
        query=query_text,
        category=filters.get("category"),
        color=filters.get("color"),
        min_price=filters.get("min_price"),
        max_price=filters.get("max_price"),
        top_k=top_k,
    )

    relevant_mask = [is_relevant(item, expected_ids, expected_categories) for item in results]
    hits = sum(1 for x in relevant_mask if x)
    precision_at_k = hits / float(top_k)

    if expected_ids:
        recall_den = max(len(expected_ids), 1)
        recall_at_k = hits / float(recall_den)
    else:
        # category-level labels are coarse; use hit/no-hit as a proxy.
        recall_at_k = 1.0 if hits > 0 else 0.0

    reciprocal_rank = 0.0
    for idx, rel in enumerate(relevant_mask, start=1):
        if rel:
            reciprocal_rank = 1.0 / float(idx)
            break

    return {
        "name": name,
        "query": query_text,
        "top_k": top_k,
        "hits": hits,
        "precision_at_k": precision_at_k,
        "recall_at_k": recall_at_k,
        "reciprocal_rank": reciprocal_rank,
        "returned_ids": [item.get("id") for item in results],
    }


def build_text_index(catalog_path: Path, cache_dir: Path, force_tfidf: bool) -> Any:
    if force_tfidf:
        text_index_module.USE_ST = False
        from sklearn.feature_extraction.text import TfidfVectorizer

        text_index_module.TfidfVectorizer = TfidfVectorizer

    try:
        return text_index_module.TextIndex(catalog_path=catalog_path, cache_dir=cache_dir, force_rebuild=False)
    except Exception as exc:
        if force_tfidf:
            raise

        # Automatic offline fallback when ST model cannot be fetched.
        text_index_module.USE_ST = False
        from sklearn.feature_extraction.text import TfidfVectorizer

        text_index_module.TfidfVectorizer = TfidfVectorizer
        importlib.reload(text_index_module)
        text_index_module.USE_ST = False
        text_index_module.TfidfVectorizer = TfidfVectorizer
        print(f"[warn] Sentence-transformer path failed ({exc}). Falling back to TF-IDF.")
        return text_index_module.TextIndex(catalog_path=catalog_path, cache_dir=cache_dir, force_rebuild=False)


def main() -> int:
    args = parse_args()
    queries = load_queries(args.queries)

    data_dir = ROOT_DIR / "data"
    cache_dir = ROOT_DIR / ".cache"
    catalog_path = ensure_catalog(data_dir=data_dir, cache_dir=cache_dir, regenerate=args.regenerate_catalog)
    index = build_text_index(catalog_path=catalog_path, cache_dir=cache_dir, force_tfidf=args.force_tfidf)

    per_query = [evaluate_query(index, q, top_k=args.top_k) for q in queries]

    precisions = [x["precision_at_k"] for x in per_query]
    recalls = [x["recall_at_k"] for x in per_query]
    reciprocal_ranks = [x["reciprocal_rank"] for x in per_query]

    summary = {
        "num_queries": len(per_query),
        "top_k": args.top_k,
        "avg_precision_at_k": statistics.mean(precisions) if precisions else 0.0,
        "avg_recall_at_k": statistics.mean(recalls) if recalls else 0.0,
        "mrr": statistics.mean(reciprocal_ranks) if reciprocal_ranks else 0.0,
    }

    print("=== Retrieval Evaluation ===")
    print(f"queries={summary['num_queries']} top_k={summary['top_k']}")
    print(
        "avg_precision@k={:.3f} avg_recall@k={:.3f} mrr={:.3f}".format(
            summary["avg_precision_at_k"],
            summary["avg_recall_at_k"],
            summary["mrr"],
        )
    )
    print("")
    for row in per_query:
        print(
            "[{}] p@k={:.3f} r@k={:.3f} rr={:.3f} hits={} query={!r}".format(
                row["name"],
                row["precision_at_k"],
                row["recall_at_k"],
                row["reciprocal_rank"],
                row["hits"],
                row["query"],
            )
        )

    if args.json_output:
        report = {"summary": summary, "queries": per_query}
        args.json_output.write_text(json.dumps(report, indent=2))
        print(f"\nWrote JSON report to: {args.json_output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
