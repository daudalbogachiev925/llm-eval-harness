"""Configuration."""
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

SEED = 42

# Models for LLM-as-judge
JUDGE_MODEL = "gpt-4o-mini"
LOCAL_JUDGE_MODEL = "cross-encoder/nli-deberta-v3-base"

# Metrics
DEFAULT_METRICS = ["exact_match", "f1", "bertscore"]
