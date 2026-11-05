"""
tests/test_response.py
======================
Tests for the response generator and conversation state.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from chatbot.response import ConversationState, ResponseGenerator


@pytest.fixture
def gen():
    return ResponseGenerator()


@pytest.fixture
def state():
    return ConversationState(session_id="test")


def test_respond_returns_string(gen, state):
    out = gen.respond("greeting", state)
    assert isinstance(out, str)
    assert len(out) > 0


def test_respond_fallback(gen, state):
    out = gen.respond("nonexistent_intent", state)
    # Should fall through to fallback responses.
    assert isinstance(out, str)


def test_respond_is_deterministic_for_same_input(gen, state):
    state.add_turn("user", "hello")
    a = gen.respond("greeting", state)
    b = gen.respond("greeting", state)
    assert a == b


def test_respond_with_variables(gen):
    state = ConversationState(session_id="t", variables={"name": "Bhoomika"})
    # The greeting responses don't use {name}, but we should at least not
    # crash when variables are provided.
    out = gen.respond("greeting", state)
    assert isinstance(out, str)


def test_next_expected_intent_followup(gen):
    # greeting -> help (followup chain)
    assert gen.next_expected_intent("greeting") == "help"


def test_next_expected_intent_none(gen):
    # help has no followup
    assert gen.next_expected_intent("help") is None


def test_conversation_state_add_turn(state):
    state.add_turn("user", "hi")
    state.add_turn("bot", "hello")
    assert len(state.turns) == 2
    assert state.turns[0].role == "user"
    assert state.turns[1].role == "bot"


def test_conversation_state_caps_history(state):
    for i in range(30):
        state.add_turn("user", f"msg {i}")
    assert len(state.turns) <= 20


def test_conversation_state_last_user_text(state):
    state.add_turn("user", "first")
    state.add_turn("bot", "reply")
    state.add_turn("user", "second")
    assert state.last_user_text() == "second"
