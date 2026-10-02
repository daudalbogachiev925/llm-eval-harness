"""Exact match and token F1 metrics."""
import re
import string
from collections import Counter


def normalize(text: str) -> str:
    """Lowercase, strip punctuation and articles."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, reference: str) -> float:
    """1.0 if normalized strings match, else 0.0."""
    return float(normalize(prediction) == normalize(reference))


def token_f1(prediction: str, reference: str) -> float:
    """Token-level F1 (SQuAD-style)."""
    pred_tokens = normalize(prediction).split()
    ref_tokens = normalize(reference).split()

    if not pred_tokens or not ref_tokens:
        return float(pred_tokens == ref_tokens)

    common = Counter(pred_tokens) & Counter(ref_tokens)
    n_common = sum(common.values())

    if n_common == 0:
        return 0.0

    precision = n_common / len(pred_tokens)
    recall = n_common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)
