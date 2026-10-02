"""Main evaluation runner."""
from typing import List, Dict
from src.metrics.exact_match import exact_match, token_f1
from src.metrics.judge import LLMJudge


def run_evaluation(
    items: List[Dict],
    metrics: List[str] | None = None,
    use_judge: bool = False,
) -> Dict:
    """Evaluate a list of {prompt, reference, prediction} items."""
    metrics = metrics or ["exact_match", "f1"]
    n = len(items)
    results = {"n_items": n, "per_item": []}

    # Exact match and F1
    if "exact_match" in metrics:
        em_scores = [exact_match(it["prediction"], it["reference"]) for it in items]
        results["exact_match"] = sum(em_scores) / n

    if "f1" in metrics:
        f1_scores = [token_f1(it["prediction"], it["reference"]) for it in items]
        results["f1"] = sum(f1_scores) / n

    # Per-item detail
    for it in items:
        row = {
            "prompt": it["prompt"],
            "reference": it["reference"],
            "prediction": it["prediction"],
            "exact_match": exact_match(it["prediction"], it["reference"]),
            "f1": token_f1(it["prediction"], it["reference"]),
        }
        results["per_item"].append(row)

    # LLM-as-judge
    if use_judge:
        judge = LLMJudge()
        scores = judge.score_batch(items)
        results["judge_scores"] = scores
        results["judge_mean"] = sum(scores) / len(scores)
        for row, s in zip(results["per_item"], scores):
            row["judge_score"] = s

    return results
