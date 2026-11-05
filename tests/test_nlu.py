"""
tests/test_nlu.py
=================
Tests for the NLU layer: text cleaning, rule scorer, blended intent
detection on a comprehensive set of queries.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from chatbot.nlu import NLU, clean_text, rule_score


# ----------------------------------------------------------------------
# clean_text
# ----------------------------------------------------------------------
def test_clean_text_lowercases():
    assert clean_text("HELLO") == "hello"


def test_clean_text_strips_punctuation():
    assert clean_text("hello, world!") == "hello world"


def test_clean_text_removes_stopwords():
    out = clean_text("please can you help me with my certificate")
    # "please", "can", "you", "me", "with", "my" are stopwords
    assert "please" not in out.split()
    assert "help" in out
    assert "certificate" in out


def test_clean_text_handles_empty():
    assert clean_text("") == ""
    assert clean_text("   ") == ""


def test_clean_text_collapses_whitespace():
    assert clean_text("hello    world") == "hello world"


# ----------------------------------------------------------------------
# rule_score
# ----------------------------------------------------------------------
def test_rule_score_picks_up_keywords():
    scores = rule_score("please help me")
    assert "help" in scores
    assert scores["help"] > 0


def test_rule_score_empty_text():
    assert rule_score("") == {}
    assert rule_score("   ") == {}


def test_rule_score_normalises_to_one():
    scores = rule_score("github github github")
    # All tokens hit the same intent; score should be normalised to 1.0.
    if scores:
        assert max(scores.values()) == pytest.approx(1.0)


# ----------------------------------------------------------------------
# NLU.parse
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def nlu():
    return NLU()


@pytest.mark.parametrize("query,expected", [
    ("hi", "greeting"),
    ("hello there", "greeting"),
    ("namaste", "greeting"),
    ("good morning", "greeting"),
    ("what can you do", "help"),
    ("help me", "help"),
    ("options", "help"),
    ("what internships do you offer", "internship_list"),
    ("list of domains", "internship_list"),
    ("available tracks", "internship_list"),
    ("how do i get my offer letter", "offer_letter"),
    ("where is my offer letter pdf", "offer_letter"),
    ("download offer letter", "offer_letter"),
    ("how to submit tasks", "task_submission"),
    ("where do i submit my project", "task_submission"),
    ("submit github link", "task_submission"),
    ("when will i get my certificate", "certificate"),
    ("certificate release date", "certificate"),
    ("how to get verified certificate", "certificate"),
    ("verification fee", "verification_fee"),
    ("how much is the verification fee", "verification_fee"),
    ("pay 99 rupees", "verification_fee"),
    ("easypaisa 400 pkr", "verification_fee"),
    ("whatsapp group link", "whatsapp_group"),
    ("join cohort group", "whatsapp_group"),
    ("telegram channel", "whatsapp_group"),
    ("task deadline", "deadline"),
    ("when is task due", "deadline"),
    ("last date", "deadline"),
    ("refund policy", "refund_policy"),
    ("can i cancel payment", "refund_policy"),
    ("thank you", "thanks"),
    ("thanks", "thanks"),
    ("bye", "goodbye"),
    ("goodbye", "goodbye"),
    ("quit", "goodbye"),
])
def test_nlu_classifies_correctly(nlu, query, expected):
    result = nlu.parse(query)
    assert result.intent == expected, (
        f"query={query!r} expected={expected} got={result.intent} "
        f"(rule={result.rule_score} sem={result.semantic_score})"
    )


def test_nlu_returns_fallback_for_garbage(nlu):
    result = nlu.parse("xyzzy flummox bleh")
    assert result.intent == "fallback"
    assert result.confidence == 0.0


def test_nlu_returns_fallback_for_empty(nlu):
    result = nlu.parse("")
    assert result.intent == "fallback"


def test_nlu_result_has_all_fields(nlu):
    result = nlu.parse("hello")
    assert hasattr(result, "intent")
    assert hasattr(result, "confidence")
    assert hasattr(result, "rule_score")
    assert hasattr(result, "semantic_score")
    assert hasattr(result, "cleaned_text")
    assert isinstance(result.to_dict(), dict)
