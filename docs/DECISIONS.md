# Decision Log

Dated, append-only log of the most consequential design and scope
decisions made during Task 2. Newest entries at the bottom.

---

## 2026-11-08 - D1: Hybrid rule + semantic NLU

**Decision:** Combine rule-based keyword matching with TF-IDF + cosine
similarity, blended 0.4 / 0.6.

**Rationale:** Pure rule bots miss paraphrases; pure semantic bots miss
short commands. The blend consistently beats either alone on the eval
fixture (see EVIDENCE.md).

**Trade-off:** Two scorers to maintain. Mitigated by the fact that both
read from the same intent catalogue.

---

## 2026-11-09 - D2: TF-IDF instead of pre-trained embeddings

**Decision:** Implement TF-IDF + cosine similarity from scratch on
NumPy. No scikit-learn, no transformers.

**Rationale:**
* Dependency-light and reproducible - no model weights to download.
* Explainable - the top-k matching patterns are visible via `explain()`.
* Sufficient accuracy for a closed-domain FAQ bot (see EVIDENCE.md for
  measured numbers).

**Trade-off:** No synonym expansion and no word-order awareness.
Documented as a known limitation.

---

## 2026-11-10 - D3: Per-intent max instead of mean

**Decision:** Score each intent by the maximum cosine similarity
between the query and any of its patterns, not the mean.

**Rationale:** If one pattern matches strongly, the intent is relevant
even if other patterns don't match. Mean would dilute strong matches.

**Trade-off:** One outlier pattern can dominate an intent's score.
Mitigated by keeping pattern sets focused (3-6 patterns per intent).

---

## 2026-11-11 - D4: Confidence gate at 0.20

**Decision:** Fall back to the `fallback` intent when the blended
confidence is below 0.20.

**Rationale:** A confident-sounding wrong answer is worse than an
honest "I didn't understand". 0.20 was chosen empirically: below this
threshold, accuracy on out-of-domain queries dropped sharply.

**Trade-off:** Some legitimate queries might fall back if the user
phrases them very differently from any pattern. Mitigated by listing
multiple paraphrases per intent in the catalogue.

---

## 2026-11-12 - D5: Deterministic response selection

**Decision:** Pick the response template by `hash(last_user_text) %
len(responses)`, not randomly.

**Rationale:** Reproducible tests. The same query always gets the same
response, which is critical for the test suite and for mentor review.

**Trade-off:** Less variety in conversations. Acceptable for an FAQ
bot where consistency matters more than novelty.

---

## 2026-11-13 - D6: Followup state machine

**Decision:** Each intent may declare a `followup` (the next expected
intent). When the user says something unparseable, the agent nudges
them toward the expected intent instead of producing a generic fallback.

**Rationale:** Multi-turn dialogues should feel guided. After "hi",
expecting "help" and nudging toward it is friendlier than "I didn't
understand".

**Trade-off:** Two intents in the current catalogue declare followups
(greeting, help). Adding more followups requires intent edits, not
plumbing changes.

---

## 2026-11-14 - D7: Data-driven intent catalogue

**Decision:** All intents live in `chatbot/intents.py::CATALOGUE` as a
plain Python list of dataclasses. Adding a new intent is a single edit
in one file.

**Rationale:** Lower the barrier for non-engineers (mentors, support
staff) to extend the bot's knowledge. No YAML, no database, no
recompilation.

**Trade-off:** Intents are baked into the code rather than loaded at
runtime. For a real production system this would be JSON/YAML; for a
1-month internship task, code is fine and easier to test.

---

## 2026-11-15 - D8: No external API keys, no LLM dependency

**Decision:** Run fully offline. No OpenAI, no HuggingFace, no API keys.

**Rationale:**
* Reproducible by anyone - the mentor can clone and run without
  obtaining API access.
* No cost, no rate limits, no privacy concerns about sending user
  queries to a third party.
* The hybrid approach already achieves >90% accuracy on the eval
  fixture, so an LLM is not needed for this scope.

**Trade-off:** Cannot handle truly open-domain questions. The agent
gracefully falls back instead of guessing.

---

## 2026-11-16 - D9: Batch evaluation fixture

**Decision:** Ship a JSONL fixture of 50+ (query, expected_intent)
pairs at `tests/fixtures/eval.jsonl`. The CLI `--eval` flag runs all
of them and reports accuracy.

**Rationale:**
* Makes accuracy measurable, not anecdotal.
* Wired into CI via exit code (0 if accuracy >= 0.70, 1 otherwise).
* Future interns can extend the fixture to catch regressions.

**Trade-off:** Fixture maintenance - new intents need new test queries.
Worth the discipline.
