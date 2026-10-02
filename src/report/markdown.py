"""Generate Markdown report from evaluation results."""
from typing import Dict


def to_markdown(results: Dict, title: str = "Evaluation Report") -> str:
    lines = [f"# {title}", ""]

    lines.append(f"**Items evaluated:** {results['n_items']}")
    lines.append("")

    lines.append("## Summary metrics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    if "exact_match" in results:
        lines.append(f"| Exact Match | {results['exact_match']:.4f} |")
    if "f1" in results:
        lines.append(f"| Token F1 | {results['f1']:.4f} |")
    if "judge_mean" in results:
        lines.append(f"| Judge (0-10) | {results['judge_mean']:.2f} |")
    lines.append("")

    lines.append("## Per-item results")
    lines.append("")
    lines.append("| Prompt | Reference | Prediction | EM | F1 |")
    lines.append("|--------|-----------|------------|-----|-----|")
    for row in results["per_item"][:20]:
        prompt = row["prompt"][:40].replace("|", "\\|")
        ref = row["reference"][:20].replace("|", "\\|")
        pred = row["prediction"][:20].replace("|", "\\|")
        lines.append(f"| {prompt} | {ref} | {pred} | {row['exact_match']:.0f} | {row['f1']:.2f} |")
    lines.append("")

    return "\n".join(lines)
