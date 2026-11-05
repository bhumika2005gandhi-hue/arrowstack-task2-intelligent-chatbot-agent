"""
chatbot/nlu.py
==============
Natural Language Understanding layer for the chatbot.

Combines two complementary strategies:

1. **Rule-based keyword matching** - fast, deterministic, never misses an
   exact keyword. Good for short commands such as ``"help"``, ``"bye"``.
2. **Semantic similarity** - a TF-IDF + cosine-similarity matcher that
   scores the user's query against every intent's example patterns. Good
   for paraphrased queries such as ``"how do I download my pdf"``.

The final score is a weighted blend: ``final = rule_score * 0.4 +
semantic_score * 0.6``. The blend was tuned by running the test suite
of sample queries - it consistently picked the right intent across
short commands and long paraphrases.

Author: Bhoomika Gandhi (Arrowstack Batch 2026, AI domain)
Ref ID: AS-2026-26671
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .intents import CATALOGUE, Intent, get_intent_map
from .semantic_matcher import SemanticMatcher


@dataclass
class NLUResult:
    """The output of one NLU pass."""

    intent: str
    confidence: float
    rule_score: float
    semantic_score: float
    cleaned_text: str

    def to_dict(self) -> dict:
        return {
            "intent": self.intent,
            "confidence": float(self.confidence),
            "rule_score": float(self.rule_score),
            "semantic_score": float(self.semantic_score),
            "cleaned_text": self.cleaned_text,
        }


# ----------------------------------------------------------------------
# Text cleaning
# ----------------------------------------------------------------------
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WHITESPACE_RE = re.compile(r"\s+")
_STOPWORDS = {
    # A small, hand-curated English stopword list. Keeping it small
    # means we still score domain words like "certificate" or "github"
    # highly without dropping them as noise.
    "a", "an", "the", "is", "are", "of", "to", "in", "on", "for", "and",
    "or", "i", "you", "my", "me", "we", "please", "can", "could", "would",
    "do", "does", "with", "this", "that", "it", "be",
}


def clean_text(text: str) -> str:
    """Lower-case, strip punctuation, collapse whitespace, drop stopwords."""
    if not text:
        return ""
    text = text.lower().strip()
    text = _PUNCT_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    tokens = [t for t in text.split() if t and t not in _STOPWORDS]
    return " ".join(tokens)


# ----------------------------------------------------------------------
# Rule-based scorer
# ----------------------------------------------------------------------
# A small keyword -> intent mapping for high-precision short commands.
# Each keyword maps to the intent name it indicates.
KEYWORD_MAP: Dict[str, str] = {
    "hi": "greeting", "hello": "greeting", "hey": "greeting",
    "namaste": "greeting", "hola": "greeting",
    "help": "help", "menu": "help", "options": "help",
    "bye": "goodbye", "exit": "goodbye", "quit": "goodbye",
    "thanks": "thanks", "thank": "thanks", "thx": "thanks",
    # Use the more specific "letter" rather than "offer" because "offer"
    # also appears in queries like "what internships do you offer" and
    # would incorrectly route them to offer_letter.
    "letter": "offer_letter",
    "certificate": "certificate", "cert": "certificate",
    "submit": "task_submission", "submission": "task_submission",
    "github": "task_submission",
    "whatsapp": "whatsapp_group", "telegram": "whatsapp_group",
    "deadline": "deadline", "due": "deadline", "last": "deadline",
    "refund": "refund_policy", "cancel": "refund_policy",
    "fee": "verification_fee", "payment": "verification_fee",
    "pay": "verification_fee", "rs": "verification_fee", "₹": "verification_fee",
    "rupees": "verification_fee", "pkr": "verification_fee",
    "easypaisa": "verification_fee", "upi": "verification_fee",
    "internship": "internship_list", "internships": "internship_list",
    "domains": "internship_list", "domain": "internship_list",
    "tracks": "internship_list", "track": "internship_list",
}


def rule_score(text: str) -> Dict[str, float]:
    """Return a dict of intent_name -> rule score (0..1)."""
    cleaned = clean_text(text)
    if not cleaned:
        return {}
    tokens = cleaned.split()
    scores: Dict[str, float] = {}
    for token in tokens:
        if token in KEYWORD_MAP:
            intent_name = KEYWORD_MAP[token]
            scores[intent_name] = scores.get(intent_name, 0.0) + 1.0
    # Normalise to 0..1: a single keyword hit is enough.
    if scores:
        max_score = max(scores.values())
        return {k: min(v / max_score, 1.0) for k, v in scores.items()}
    return {}


# ----------------------------------------------------------------------
# NLU engine
# ----------------------------------------------------------------------
class NLU:
    """Top-level NLU engine that blends rule-based and semantic scores."""

    RULE_WEIGHT = 0.4
    SEMANTIC_WEIGHT = 0.6

    def __init__(self, matcher: Optional[SemanticMatcher] = None) -> None:
        # Build the semantic matcher once and reuse it.
        self.matcher = matcher or SemanticMatcher(CATALOGUE)
        self._intent_map = get_intent_map()

    def parse(self, text: str) -> NLUResult:
        cleaned = clean_text(text)

        # Empty / whitespace-only input -> fallback
        if not cleaned:
            return NLUResult(
                intent="fallback",
                confidence=0.0,
                rule_score=0.0,
                semantic_score=0.0,
                cleaned_text="",
            )

        rule_scores = rule_score(text)
        sem_scores = self.matcher.score(cleaned)

        # Blend
        all_intents = set(rule_scores) | set(sem_scores)
        if not all_intents:
            return NLUResult(
                intent="fallback",
                confidence=0.0,
                rule_score=0.0,
                semantic_score=0.0,
                cleaned_text=cleaned,
            )

        blended: Dict[str, float] = {}
        for name in all_intents:
            r = rule_scores.get(name, 0.0)
            s = sem_scores.get(name, 0.0)
            blended[name] = r * self.RULE_WEIGHT + s * self.SEMANTIC_WEIGHT

        # Pick the top intent; use priority as a tiebreaker.
        best_name, best_score = self._pick_best(blended)
        intent_obj = self._intent_map.get(best_name)
        if intent_obj is None:
            best_name = "fallback"
            best_score = 0.0

        return NLUResult(
            intent=best_name,
            confidence=round(best_score, 4),
            rule_score=round(rule_scores.get(best_name, 0.0), 4),
            semantic_score=round(sem_scores.get(best_name, 0.0), 4),
            cleaned_text=cleaned,
        )

    # ------------------------------------------------------------------
    def _pick_best(self, scores: Dict[str, float]) -> Tuple[str, float]:
        """Return (intent_name, score) for the highest blended score.

        Ties are broken by intent priority (higher priority wins).
        """
        ranked = sorted(
            scores.items(),
            key=lambda kv: (kv[1], self._priority_of(kv[0])),
            reverse=True,
        )
        name, score = ranked[0]
        return name, score

    @staticmethod
    def _priority_of(name: str) -> int:
        intent = get_intent_map().get(name)
        return intent.priority if intent else 0
