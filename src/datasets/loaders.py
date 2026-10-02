"""Load evaluation datasets."""
import json
from pathlib import Path
from typing import List, Dict
from src.config import DATA_DIR


def load_jsonl(path: str | Path) -> List[Dict]:
    """Load JSONL file with {prompt, reference, prediction} fields."""
    path = Path(path)
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def save_jsonl(items: List[Dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_sample_dataset() -> List[Dict]:
    """Built-in small dataset for smoke testing."""
    return [
        {
            "prompt": "What is the capital of France?",
            "reference": "Paris",
            "prediction": "Paris",
        },
        {
            "prompt": "What is 2+2?",
            "reference": "4",
            "prediction": "four",
        },
        {
            "prompt": "Who wrote Hamlet?",
            "reference": "William Shakespeare",
            "prediction": "Shakespeare",
        },
        {
            "prompt": "What color is the sky?",
            "reference": "blue",
            "prediction": "green",
        },
        {
            "prompt": "What is the largest planet?",
            "reference": "Jupiter",
            "prediction": "Jupiter",
        },
    ]
