"""Run evaluation on the built-in sample dataset."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.datasets.loaders import load_sample_dataset
from src.eval.runner import run_evaluation
from src.report.markdown import to_markdown
from src.config import RESULTS_DIR


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    print("Loading sample dataset...")
    items = load_sample_dataset()
    print(f"  {len(items)} items")

    print("Running evaluation...")
    results = run_evaluation(items, metrics=["exact_match", "f1"], use_judge=True)

    print(f"\nResults:")
    print(f"  Exact Match: {results['exact_match']:.4f}")
    print(f"  Token F1:    {results['f1']:.4f}")
    print(f"  Judge mean:  {results['judge_mean']:.2f}")

    # Save report
    report = to_markdown(results, title="Sample Evaluation")
    out = RESULTS_DIR / "sample_report.md"
    out.write_text(report, encoding="utf-8")
    print(f"\nReport saved to {out}")


if __name__ == "__main__":
    main()
