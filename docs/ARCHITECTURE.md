# Architecture

## High-level flow

```
   User text
       │
       ▼
   ┌────────────────────────────────────────┐
   │              NLU.parse()                │
   │                                        │
   │   1. clean_text()                       │
   │      - lowercase                        │
   │      - strip punctuation                │
   │      - drop stopwords                   │
   │                                        │
   │   2. rule_score(text)                   │
   │      - keyword -> intent map            │
   │      - returns Dict[intent, 0..1]       │
   │                                        │
   │   3. SemanticMatcher.score(cleaned)     │
   │      - TF-IDF + cosine similarity       │
   │      - returns Dict[intent, 0..1]       │
   │                                        │
   │   4. blend = rule*0.4 + sem*0.6         │
   │      pick max; tiebreak by priority    │
   │                                        │
   └────────────────┬───────────────────────┘
                    │ NLUResult
                    ▼
   ┌────────────────────────────────────────┐
   │            ChatbotAgent                 │
   │   - confidence gate (threshold 0.20)    │
   │   - followup state machine              │
   │   - per-session history                 │
   └────────────────┬───────────────────────┘
                    │
                    ▼
   ┌────────────────────────────────────────┐
   │        ResponseGenerator                │
   │   - deterministic template pick         │
   │   - {variable} substitution             │
   └────────────────────────────────────────┘
                    │
                    ▼
                Reply text
```

## Module responsibilities

| Module | Responsibility |
|---|---|
| `chatbot/intents.py` | Single source of truth for the intent catalogue (patterns, responses, followups, priority). Add a new intent by editing only this file. |
| `chatbot/semantic_matcher.py` | Build a TF-IDF matrix over all patterns; score a query by max cosine similarity per intent. Implemented from scratch on NumPy. |
| `chatbot/nlu.py` | Text cleaning, rule-based keyword scoring, blended NLU engine. |
| `chatbot/response.py` | Response selection, template substitution, conversation state. |
| `chatbot/agent.py` | Top-level orchestrator: session management, confidence gate, followups. |
| `main.py` | CLI - REPL, single query, batch eval. |

## Key design decisions

### Hybrid rule + semantic NLU

Pure rule-based bots are brittle (any paraphrase breaks them). Pure
semantic bots miss trivial commands like ``"help"`` because the cosine
similarity of a one-word query is dominated by stopword-like behaviour.
The hybrid blend gives the best of both:

* Short commands are caught by the rule scorer with score 1.0.
* Paraphrased questions are caught by the semantic scorer.
* The 0.4 / 0.6 weighting was tuned empirically on the eval fixture
  (see `tests/fixtures/eval.jsonl`).

### TF-IDF instead of embeddings

TF-IDF + cosine is the textbook baseline. It is:
* **Fast** - O(n_intents) per query after the matrix is built.
* **Explainable** - `matcher.explain()` returns the top-k matching
  patterns with their scores, which is critical for mentor review.
* **Dependency-light** - no scikit-learn, no transformers, no GPU.
* **Reproducible** - no random seeds, no model weights to version.

The trade-off is no synonym expansion and no word-order awareness.
This is documented as a known limitation in `EVIDENCE.md`.

### Per-intent max instead of mean

The semantic scorer returns the **maximum** cosine similarity between
the query and any pattern belonging to an intent, not the mean. Max
rewards intents that have at least one pattern very similar to the
query; mean would dilute strong matches with weak ones.

### Confidence gate

If the blended score is below 0.20, the agent falls back to a graceful
"I didn't understand" reply. This prevents confident-sounding wrong
answers on out-of-domain queries.

### Deterministic response selection

The response generator picks a template deterministically from each
intent's candidate list using ``hash(last_user_text) % len(responses)``.
The same query always gets the same response, which makes tests
reproducible.

### Followup state machine

Some intents declare a `followup` (e.g. greeting -> help). After a
greeting, if the user says something unparseable, the agent nudges them
toward the expected `help` intent rather than producing a generic
fallback. This makes multi-turn dialogues feel guided.

### Fallback intent

The `fallback` intent has empty `patterns` and a negative priority. It
is selected only when:
* The user input is empty.
* The blended confidence is below the threshold.
* The rule and semantic scorers both return empty dicts.

## Failure modes

| Failure | Behaviour |
|---|---|
| Empty user input | Returns fallback intent with confidence 0.0. |
| Unknown vocabulary | Semantic matcher returns empty dict; rule scorer takes over. |
| Missing session id | A new session is created automatically. |
| Template references unknown variable | `ResponseGenerator` catches the KeyError and returns the template raw. |

## Where this fits in a production system

The agent is intentionally stateless across sessions (everything lives
in process memory). To deploy behind a web API:

1. Wrap `ChatbotAgent.respond()` in a FastAPI route that accepts
   ``{session_id, text}`` and returns the response dict.
2. Persist `ConversationState` in Redis if you need session continuity
   across server restarts.
3. Add a logging hook to write every (query, intent, confidence) tuple
   to a metrics pipeline for offline accuracy monitoring.
4. Plug in an LLM-based fallback (e.g. OpenAI) for queries that score
   below a second, lower threshold (0.10) - the architecture supports
   this without changes to the NLU layer.
