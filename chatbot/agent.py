"""
chatbot/agent.py
================
The high-level ChatbotAgent that ties NLU + ResponseGenerator + state
together. This is the single class the application layer (CLI, web API)
should talk to.

Author: Bhoomika Gandhi (Arrowstack Batch 2026, AI domain)
Ref ID: AS-2026-26671
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import asdict
from typing import Dict, Optional

from .intents import get_intent_map
from .nlu import NLU
from .response import ConversationState, ResponseGenerator
from .semantic_matcher import SemanticMatcher


logger = logging.getLogger("arrowstack.chatbot")


class ChatbotAgent:
    """A rule-based + semantic chatbot agent.

    Parameters
    ----------
    confidence_threshold:
        Below this confidence, the agent falls back to the ``fallback``
        intent. Default 0.20 - anything lower means neither the rule
        scorer nor the semantic scorer found a meaningful match.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.20,
        nlu: Optional[NLU] = None,
    ) -> None:
        self.nlu = nlu or NLU()
        self.response_gen = ResponseGenerator()
        self.confidence_threshold = confidence_threshold
        self.sessions: Dict[str, ConversationState] = {}

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------
    def new_session(self, session_id: Optional[str] = None) -> str:
        sid = session_id or uuid.uuid4().hex[:12]
        self.sessions[sid] = ConversationState(session_id=sid)
        return sid

    def get_session(self, session_id: str) -> ConversationState:
        if session_id not in self.sessions:
            self.new_session(session_id)
        return self.sessions[session_id]

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def respond(self, user_text: str, session_id: str = "default",
                variables: Optional[Dict[str, str]] = None) -> Dict:
        """Process ``user_text`` and return a structured response dict."""
        state = self.get_session(session_id)
        state.add_turn("user", user_text)

        nlu_result = self.nlu.parse(user_text)
        intent_name = nlu_result.intent

        # Apply confidence gate. If the chosen intent is below threshold,
        # fall back - but only if the chosen intent isn't already fallback.
        if intent_name != "fallback" and nlu_result.confidence < self.confidence_threshold:
            logger.debug(
                "Intent %s below threshold (%.3f); falling back.",
                intent_name, nlu_result.confidence,
            )
            intent_name = "fallback"

        # Drive followups: if the agent expected a specific intent and
        # the user said something off-topic, keep the conversation going
        # with the matched intent rather than resetting.
        if state.expected_intent and intent_name == "fallback":
            # If we expected e.g. "help" after a greeting, and the user
            # said something we couldn't parse, nudge them gently.
            intent_name = state.expected_intent

        response = self.response_gen.respond(intent_name, state, variables)
        state.add_turn("bot", response, intent=intent_name)

        # Update expected intent for next turn.
        state.expected_intent = self.response_gen.next_expected_intent(intent_name)

        return {
            "session_id": session_id,
            "user_text": user_text,
            "intent": intent_name,
            "confidence": nlu_result.confidence,
            "rule_score": nlu_result.rule_score,
            "semantic_score": nlu_result.semantic_score,
            "response": response,
            "expected_intent": state.expected_intent,
        }

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def explain(self, query: str, top_k: int = 3):
        """Return the top-k matching documents for ``query``."""
        if isinstance(self.nlu.matcher, SemanticMatcher):
            return self.nlu.matcher.explain(query, top_k=top_k)
        return []

    def available_intents(self) -> Dict[str, dict]:
        """Return a name -> intent summary, for introspection."""
        return {name: intent.to_dict() for name, intent in get_intent_map().items()}
