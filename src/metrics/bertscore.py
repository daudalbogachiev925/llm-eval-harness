"""BERTScore wrapper. Requires bert-score package and downloads a model on first use."""
from typing import List


class BERTScorer:
    """Lazy-loaded BERTScore to avoid downloading model in tests."""

    def __init__(self, model_type: str = "distilbert-base-uncased", lang: str = "en"):
        self.model_type = model_type
        self.lang = lang
        self._scorer = None

    def _load(self):
        if self._scorer is None:
            from bert_score import BERTScorer as _B
            self._scorer = _B(model_type=self.model_type, lang=self.lang)
        return self._scorer

    def score(self, predictions: List[str], references: List[str]) -> dict:
        """Return dict with precision, recall, f1 arrays."""
        scorer = self._load()
        P, R, F1 = scorer.score(predictions, references)
        return {
            "precision": P.tolist(),
            "recall": R.tolist(),
            "f1": F1.tolist(),
            "mean_f1": float(F1.mean()),
        }
