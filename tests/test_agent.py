"""
tests/test_agent.py
===================
End-to-end tests for the ChatbotAgent: multi-turn dialogue, followups,
fallback behaviour, and session management.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from chatbot.agent import ChatbotAgent


@pytest.fixture
def agent():
    return ChatbotAgent()


# ----------------------------------------------------------------------
# Sessions
# ----------------------------------------------------------------------
def test_new_session_returns_id(agent):
    sid = agent.new_session()
    assert isinstance(sid, str)
    assert len(sid) > 0


def test_get_session_creates_if_missing(agent):
    state = agent.get_session("never-seen-before")
    assert state is not None
    assert state.session_id == "never-seen-before"


def test_sessions_are_isolated(agent):
    sid1 = agent.new_session("alice")
    sid2 = agent.new_session("bob")
    agent.respond("hi", session_id=sid1)
    assert len(agent.get_session(sid1).turns) > 0
    # bob's session should still be empty
    assert len(agent.get_session(sid2).turns) == 0


# ----------------------------------------------------------------------
# End-to-end classification
# ----------------------------------------------------------------------
def test_respond_returns_full_dict(agent):
    sid = agent.new_session()
    result = agent.respond("hello", session_id=sid)
    assert "session_id" in result
    assert "user_text" in result
    assert "intent" in result
    assert "confidence" in result
    assert "response" in result


def test_respond_greeting_flow(agent):
    sid = agent.new_session()
    result = agent.respond("hello", session_id=sid)
    assert result["intent"] == "greeting"
    assert "Hello" in result["response"] or "Hi" in result["response"]
    # After greeting, the agent expects help next.
    assert result["expected_intent"] == "help"


def test_respond_fallback_for_garbage(agent):
    sid = agent.new_session()
    result = agent.respond("xyzzy flummox bleh", session_id=sid)
    assert result["intent"] == "fallback"


def test_respond_thanks(agent):
    sid = agent.new_session()
    result = agent.respond("thank you", session_id=sid)
    assert result["intent"] == "thanks"


def test_respond_quit_intent(agent):
    sid = agent.new_session()
    result = agent.respond("quit", session_id=sid)
    assert result["intent"] == "goodbye"


# ----------------------------------------------------------------------
# Multi-turn / followups
# ----------------------------------------------------------------------
def test_followup_after_greeting(agent):
    sid = agent.new_session()
    # 1. User says hi -> bot greets and expects help next.
    r1 = agent.respond("hi", session_id=sid)
    assert r1["expected_intent"] == "help"
    # 2. User says something unparseable - agent should nudge toward help.
    r2 = agent.respond("asdf jkl", session_id=sid)
    # The fallback in agent.py re-routes to expected_intent ("help").
    assert r2["intent"] in {"help", "fallback"}


def test_expose_intents(agent):
    intents = agent.available_intents()
    assert "greeting" in intents
    assert "fallback" in intents
    assert "task_submission" in intents


def test_explain_returns_top_k(agent):
    explanation = agent.explain("where is my offer letter", top_k=3)
    assert isinstance(explanation, list)
    assert len(explanation) <= 3
