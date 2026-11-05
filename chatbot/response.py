"""
chatbot/response.py
===================
Response generation and conversation state for the chatbot.

Responsibilities:
* Pick a deterministic response from an intent's candidates (so tests
  are reproducible).
* Apply template substitution for variables such as ``{name}`` or
  ``{domain}``.
* Drive multi-turn conversations via the ``followup`` field on each
  intent.
* Keep a tiny in-memory conversation log per session id.

Author: Bhoomika Gandhi (Arrowstack Batch 2026, AI domain)
Ref ID: AS-2026-26671
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .intents import Intent, get_intent_map


@dataclass
class ConversationTurn:
    role: str  # "user" or "bot"
    text: str
    intent: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class ConversationState:
    """Per-session state."""

    session_id: str
    turns: List[ConversationTurn] = field(default_factory=list)
    expected_intent: Optional[str] = None  # for followups
    variables: Dict[str, str] = field(default_factory=dict)

    def add_turn(self, role: str, text: str, intent: Optional[str] = None) -> None:
        self.turns.append(ConversationTurn(role=role, text=text, intent=intent))
        # Keep only the last 20 turns to bound memory.
        if len(self.turns) > 20:
            self.turns = self.turns[-20:]

    def last_user_text(self) -> str:
        for turn in reversed(self.turns):
            if turn.role == "user":
                return turn.text
        return ""


class ResponseGenerator:
    """Pick a response for an intent and apply template substitutions."""

    def __init__(self) -> None:
        self._intent_map = get_intent_map()

    # ------------------------------------------------------------------
    def respond(
        self,
        intent_name: str,
        state: ConversationState,
        variables: Optional[Dict[str, str]] = None,
    ) -> str:
        intent = self._intent_map.get(intent_name) or self._intent_map["fallback"]
        # Deterministic selection: hash the last user text so the same
        # query always gets the same response, but different queries get
        # variety.
        last_text = state.last_user_text()
        idx = (hash(last_text) % len(intent.responses)) if intent.responses else 0
        template = intent.responses[idx]

        merged = dict(state.variables)
        if variables:
            merged.update(variables)

        try:
            return template.format(**merged)
        except (KeyError, IndexError):
            # If the template references a variable we don't have, return it raw.
            return template

    # ------------------------------------------------------------------
    def next_expected_intent(self, intent_name: str) -> Optional[str]:
        intent = self._intent_map.get(intent_name)
        if intent is None:
            return None
        return intent.followup
