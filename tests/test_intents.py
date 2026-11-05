"""
tests/test_intents.py
=====================
Tests for the intent catalogue module.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from chatbot.intents import CATALOGUE, get_catalogue, get_intent, get_intent_map


def test_catalogue_non_empty():
    assert len(CATALOGUE) >= 10


def test_get_catalogue_returns_copy():
    a = get_catalogue()
    a.append("spoiled")
    assert "spoiled" not in get_catalogue()


def test_get_intent_by_name():
    intent = get_intent("greeting")
    assert intent is not None
    assert intent.name == "greeting"
    assert len(intent.patterns) >= 3
    assert len(intent.responses) >= 1


def test_get_intent_missing_returns_none():
    assert get_intent("does_not_exist") is None


def test_get_intent_map_keys():
    m = get_intent_map()
    assert "greeting" in m
    assert "fallback" in m
    assert "task_submission" in m
    assert all(v.name == k for k, v in m.items())


def test_each_intent_has_at_least_one_response():
    for intent in CATALOGUE:
        assert len(intent.responses) >= 1, f"{intent.name} has no responses"


def test_fallback_has_empty_patterns():
    fb = get_intent("fallback")
    assert fb is not None
    assert fb.patterns == []


def test_intent_to_dict_roundtrip():
    intent = get_intent("certificate")
    d = intent.to_dict()
    assert d["name"] == "certificate"
    assert "patterns" in d and "responses" in d
