"""LLM-as-judge metric. Uses OpenAI API if available, else rule-based fallback."""
import os
from typing import List


JUDGE_PROMPT = """You are evaluating an AI answer against a reference.

Question: {prompt}
Reference answer: {reference}
AI answer: {prediction}

Rate the AI answer on a scale of 0-10 for correctness.
Return ONLY the integer score, nothing else.
"""


class LLMJudge:
    """LLM-as-judge. Falls back to token overlap if no API key."""

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self.client = None
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)

    def score_one(self, prompt: str, reference: str, prediction: str) -> int:
        if self.client is None:
            return self._fallback_score(prediction, reference)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": JUDGE_PROMPT.format(
                        prompt=prompt, reference=reference, prediction=prediction,
                    ),
                }],
                temperature=0,
                max_tokens=5,
            )
            text = response.choices[0].message.content.strip()
            return int(text)
        except Exception:
            return self._fallback_score(prediction, reference)

    def score_batch(self, items: List[dict]) -> List[int]:
        return [
            self.score_one(it["prompt"], it["reference"], it["prediction"])
            for it in items
        ]

    @staticmethod
    def _fallback_score(prediction: str, reference: str) -> int:
        """Simple token overlap as fallback when no API."""
        p = set(prediction.lower().split())
        r = set(reference.lower().split())
        if not p or not r:
            return 0
        jaccard = len(p & r) / len(p | r)
        return int(round(jaccard * 10))
