"""
chatbot/semantic_matcher.py
===========================
Semantic similarity scorer for the chatbot NLU.

The matcher is a tiny TF-IDF + cosine-similarity engine implemented from
scratch on top of NumPy. It does not depend on scikit-learn so the
project stays lightweight and reproducible.

Why TF-IDF?
-----------
* It rewards rare domain words ("certificate", "github", "whatsapp") and
  penalises common ones, which matches how users actually phrase
  questions.
* Cosine similarity is the textbook baseline for short-text semantic
  similarity. It is fast (O(n_intents) per query after the matrix is
  built) and easy to debug.
* We compare the user query against each intent's *pattern set* (not
  against a single concatenated pattern), which gives a smoother score
  distribution.

Limitations
-----------
* No word-order awareness ("when can I submit" vs "I can submit when"
  score identically). Acceptable for a 1-month internship task.
* No synonym expansion ("download" vs "get" need to both appear in
  patterns). Mitigated by listing multiple paraphrases per intent.

Author: Bhoomika Gandhi (Arrowstack Batch 2026, AI domain)
Ref ID: AS-2026-26671
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Dict, List

import numpy as np

from .intents import Intent


class SemanticMatcher:
    """Build a TF-IDF matrix from intent patterns and score queries."""

    def __init__(self, intents: List[Intent]) -> None:
        self.intents = intents
        self._documents: List[str] = []  # one document per pattern
        self._doc_to_intent: List[str] = []  # parallel array
        self._vocab: Dict[str, int] = {}
        self._idf: np.ndarray = np.array([])
        self._tfidf_matrix: np.ndarray = np.array([])
        self._build()

    # ------------------------------------------------------------------
    # Building the matrix
    # ------------------------------------------------------------------
    def _build(self) -> None:
        for intent in self.intents:
            if not intent.patterns:
                continue
            for pattern in intent.patterns:
                tokens = pattern.lower().split()
                self._documents.append(" ".join(tokens))
                self._doc_to_intent.append(intent.name)

        if not self._documents:
            # Nothing to match against - everything will score 0.
            self._idf = np.zeros(0)
            self._tfidf_matrix = np.zeros((0, 0))
            return

        # Build vocabulary
        vocab: Dict[str, int] = {}
        for doc in self._documents:
            for token in doc.split():
                if token not in vocab:
                    vocab[token] = len(vocab)
        self._vocab = vocab

        n_docs = len(self._documents)
        n_vocab = len(vocab)

        # Term-frequency matrix (n_docs x n_vocab)
        tf = np.zeros((n_docs, n_vocab), dtype=np.float64)
        for i, doc in enumerate(self._documents):
            counts = Counter(doc.split())
            for token, count in counts.items():
                tf[i, vocab[token]] = count

        # Document frequency for each term
        df = (tf > 0).sum(axis=0)
        # Smoothed IDF: ln((1 + N) / (1 + df)) + 1
        self._idf = np.log((1.0 + n_docs) / (1.0 + df)) + 1.0

        # TF-IDF matrix
        self._tfidf_matrix = tf * self._idf

        # L2-normalise rows for cosine similarity
        norms = np.linalg.norm(self._tfidf_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._tfidf_matrix = self._tfidf_matrix / norms

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def score(self, query: str) -> Dict[str, float]:
        """Return intent_name -> best (max) cosine similarity score.

        The score is the maximum cosine similarity between the query
        vector and any document belonging to that intent. Using max
        instead of mean rewards intents that have at least one pattern
        very similar to the query.
        """
        if not self._documents or not query:
            return {}

        query_vec = self._vectorize(query)
        if query_vec is None:
            return {}

        # Cosine similarity = dot product (rows are already L2-normalised).
        sims = self._tfidf_matrix @ query_vec  # shape: (n_docs,)
        # For each intent, keep the max similarity.
        scores: Dict[str, float] = {}
        for doc_idx, intent_name in enumerate(self._doc_to_intent):
            sim = float(sims[doc_idx])
            if intent_name not in scores or sim > scores[intent_name]:
                scores[intent_name] = sim
        # Clip to [0, 1] (cosine of normalised non-negative vectors is in
        # [0, 1] but numerical noise can push it slightly above 1).
        return {k: max(0.0, min(v, 1.0)) for k, v in scores.items()}

    # ------------------------------------------------------------------
    def _vectorize(self, text: str) -> np.ndarray:
        """Return an L2-normalised TF-IDF row vector for ``text``."""
        tokens = text.lower().split()
        counts = Counter(tokens)
        vec = np.zeros(len(self._vocab), dtype=np.float64)
        for token, count in counts.items():
            idx = self._vocab.get(token)
            if idx is not None:
                vec[idx] = count
        if not vec.any():
            return None  # type: ignore[return-value]
        vec = vec * self._idf
        norm = np.linalg.norm(vec)
        if norm == 0:
            return None  # type: ignore[return-value]
        return vec / norm

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def explain(self, query: str, top_k: int = 3) -> List[Dict]:
        """Return the top-k matching documents for debugging."""
        if not self._documents or not query:
            return []
        qv = self._vectorize(query)
        if qv is None:
            return []
        sims = self._tfidf_matrix @ qv
        order = np.argsort(sims)[::-1][:top_k]
        result = []
        for idx in order:
            result.append({
                "intent": self._doc_to_intent[idx],
                "document": self._documents[idx],
                "score": float(sims[idx]),
            })
        return result

    def vocabulary_size(self) -> int:
        return len(self._vocab)
