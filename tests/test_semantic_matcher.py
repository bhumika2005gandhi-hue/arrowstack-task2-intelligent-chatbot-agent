"""
tests/test_semantic_matcher.py
==============================
Tests for the from-scratch TF-IDF + cosine similarity matcher.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest

from chatbot.intents import CATALOGUE
from chatbot.semantic_matcher import SemanticMatcher


@pytest.fixture(scope="module")
def matcher():
    return SemanticMatcher(CATALOGUE)


def test_matcher_initialises(matcher):
    assert matcher.vocabulary_size() > 0
    assert matcher._tfidf_matrix.shape[0] > 0


def test_score_returns_dict(matcher):
    scores = matcher.score("hello")
    assert isinstance(scores, dict)
    assert len(scores) > 0


def test_score_gives_high_score_to_greeting_for_hello(matcher):
    scores = matcher.score("hello")
    assert "greeting" in scores
    # "hello" appears verbatim in greeting patterns, so similarity should
    # be high.
    assert scores["greeting"] > 0.5


def test_score_zero_for_unknown_words(matcher):
    scores = matcher.score("xyzzy flummox")
    # No tokens match the vocab; expect either an empty dict or all zeros.
    if scores:
        assert all(v == 0.0 for v in scores.values())


def test_score_bounded_0_to_1(matcher):
    scores = matcher.score("offer letter certificate whatsapp")
    for v in scores.values():
        assert 0.0 <= v <= 1.0


def test_explain_returns_top_k(matcher):
    explanations = matcher.explain("hello", top_k=3)
    assert len(explanations) <= 3
    assert len(explanations) >= 1
    for item in explanations:
        assert "intent" in item
        assert "document" in item
        assert "score" in item


def test_explain_empty_query(matcher):
    assert matcher.explain("", top_k=3) == []


def test_documents_built_from_patterns(matcher):
    # Number of documents should equal sum of patterns across intents.
    expected = sum(len(i.patterns) for i in CATALOGUE if i.patterns)
    assert len(matcher._documents) == expected


def test_tfidf_matrix_normalised(matcher):
    # Each row should have L2 norm ~ 1 (or 0 for empty rows, but we have none).
    norms = np.linalg.norm(matcher._tfidf_matrix, axis=1)
    for n in norms:
        assert n == pytest.approx(1.0, abs=1e-6) or n == pytest.approx(0.0, abs=1e-6)


def testidf_positive(matcher):
    # IDF values should all be positive (since we use log((1+N)/(1+df)) + 1).
    assert (matcher._idf > 0).all()
