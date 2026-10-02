"""Tests for metrics."""
from src.metrics.exact_match import exact_match, token_f1, normalize


def test_normalize():
    assert normalize("The Quick Brown Fox!") == "quick brown fox"
    assert normalize("A cat") == "cat"


def test_exact_match():
    assert exact_match("Paris", "paris") == 1.0
    assert exact_match("Paris", "London") == 0.0
    assert exact_match("The Paris", "paris") == 1.0


def test_token_f1():
    assert token_f1("Paris", "Paris") == 1.0
    f1 = token_f1("the capital Paris", "Paris")
    assert 0 < f1 < 1
    assert token_f1("London", "Paris") == 0.0
