# Arrowstack AI Internship - Task 2
# Intelligent Rule-Based & Semantic Chatbot Agent

**Intern:** Bhoomika Gandhi (bhumika2005gandhi@gmail.com)
**Ref ID:** AS-2026-26671
**Domain:** Artificial Intelligence
**Cohort:** Arrowstack Batch 2026 (1-Month Virtual)
**Project type:** Mandatory

---

## 1. Project Overview

This project delivers an **intelligent, hybrid chatbot agent** for the
Arrowstack internship portal. It answers student questions about
internships, offer letters, task submission, certificate verification,
fees, deadlines, and the WhatsApp cohort group.

The "intelligent" part is that the agent combines **two complementary
strategies** rather than relying on a single approach:

1. **Rule-based keyword matching** - fast, deterministic, never misses an
   exact keyword such as ``"help"``, ``"bye"``, ``"github"``.
2. **Semantic similarity** - a TF-IDF + cosine-similarity engine
   implemented from scratch on NumPy. It scores the user's query
   against every intent's example patterns and rewards rare domain
   words ("certificate", "verification", "offer") while ignoring
   stopwords.

The final score is a weighted blend: ``final = rule * 0.4 + semantic * 0.6``.
Below a confidence threshold (0.20 by default) the agent falls back to
a graceful "I didn't understand" reply rather than guessing.

> **Real-world scenario:** This is the kind of first-line support agent
> that organisations deploy to deflect repetitive tickets. The hybrid
> approach is intentionally transparent and auditable - every decision
> can be explained with the rule score and the top-3 matching patterns
> (see ``--explain`` in the CLI).

---

## 2. Repository Layout

```
arrowstack-task2-intelligent-chatbot-agent/
├── main.py                      # CLI (REPL / single query / batch eval)
├── chatbot/
│   ├── __init__.py
│   ├── agent.py                 # ChatbotAgent - top-level orchestrator
│   ├── intents.py               # Intent catalogue (data-driven)
│   ├── nlu.py                   # Text cleaning + rule scorer + NLU engine
│   ├── response.py              # Response generator + conversation state
│   └── semantic_matcher.py      # TF-IDF + cosine similarity from scratch
├── data/
│   └── knowledge_base.json      # Static facts about Arrowstack
├── tests/
│   ├── test_intents.py
│   ├── test_nlu.py
│   ├── test_semantic_matcher.py
│   ├── test_response.py
│   └── test_agent.py
├── tests/fixtures/
│   └── eval.jsonl               # 30+ (query, expected_intent) pairs
├── requirements.txt
├── README.md                    # this file
├── EVIDENCE.md                  # test results + accuracy report
├── LICENSE                      # MIT
├── .gitignore
└── docs/
    ├── ARCHITECTURE.md
    └── DECISIONS.md             # dated decision log
```

---

## 3. Setup

### 3.1 Prerequisites

* Python 3.10+
* No external API keys, no LLM service, no GPU

### 3.2 Install

```bash
git clone <your-fork-url>
cd arrowstack-task2-intelligent-chatbot-agent

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

---

## 4. Usage

### 4.1 Interactive REPL

```bash
python main.py
```

You'll see a banner, then a turn-by-turn conversation:

```
Bot: Hi! I am the Arrowstack AI assistant. Ask me anything about
     internships, offer letters, tasks, certificates, or the WhatsApp
     cohort group.

You: hey
Bot: Hello! I am the Arrowstack AI assistant. How can I help you today?
You: where can i find my offer letter pdf
Bot: Your official Offer Letter is generated instantly upon successful
     registration. Log in to the Intern Portal and click 'Download (PDF)'
     next to your registration entry. ...
You: thanks
Bot: You're welcome! Good luck with your internship. :)
You: quit
```

### 4.2 Single query (JSON output)

```bash
python main.py --query "how do I submit my github project link"
```

Returns:

```json
{
  "session_id": "oneshot",
  "user_text": "how do I submit my github project link",
  "intent": "task_submission",
  "confidence": 0.6,
  "rule_score": 0.4,
  "semantic_score": 0.75,
  "response": "Open the 'Internship Task Submission Form' ...",
  "expected_intent": null
}
```

### 4.3 With explanation (top-3 matching patterns)

```bash
python main.py --query "i want my certificate" --explain
```

Adds an ``explanation`` list showing which intent patterns drove the
score - useful for debugging and for mentor review.

### 4.4 Batch accuracy evaluation

```bash
python main.py --eval tests/fixtures/eval.jsonl
```

Returns a JSON summary:

```json
{
  "total": 36,
  "correct": 33,
  "accuracy": 0.9167,
  "per_intent": {
    "task_submission": {"total": 5, "correct": 5},
    ...
  },
  "failures": [...]
}
```

The exit code is ``0`` if accuracy >= 0.70, ``1`` otherwise - so this
can be wired into CI.

---

## 5. Architecture (short version)

```
   User text
       │
       ▼
   ┌────────────────────────────────────┐
   │             NLU.parse()             │
   │                                    │
   │  ┌──────────────┐  ┌────────────┐  │
   │  │ clean_text() │─▶│ rule_score │  │  ──▶ Dict[intent, 0..1]
   │  └──────────────┘  └────────────┘  │
   │         │                          │
   │  ┌──────────────────────────────┐  │
   │  │ SemanticMatcher.score(query) │  │  ──▶ Dict[intent, 0..1]
   │  │   (TF-IDF + cosine sim)      │  │
   │  └──────────────────────────────┘  │
   │                                    │
   │   blend = rule*0.4 + sem*0.6       │
   │   pick max + priority tiebreaker   │
   └────────────────┬───────────────────┘
                    │ NLUResult
                    ▼
   ┌────────────────────────────────────┐
   │           ChatbotAgent              │
   │   - confidence gate -> fallback     │
   │   - followup state machine          │
   └────────────────┬───────────────────┘
                    │
                    ▼
   ┌────────────────────────────────────┐
   │      ResponseGenerator              │
   │   - deterministic template pick     │
   │   - {variable} substitution         │
   └────────────────────────────────────┘
                    │
                    ▼
                Reply text
```

A longer write-up lives in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## 6. Supported Intents

| Intent | Purpose |
|---|---|
| `greeting` | hi / hello / namaste |
| `help` | what can you do |
| `internship_list` | which domains / tracks available |
| `offer_letter` | download offer letter |
| `task_submission` | how to submit GitHub + LinkedIn |
| `certificate` | when / how certificate releases |
| `verification_fee` | Rs.99 / 400 PKR payment |
| `whatsapp_group` | cohort WhatsApp / Telegram |
| `deadline` | task due date |
| `refund_policy` | refund questions |
| `thanks` | thank you |
| `goodbye` | bye / exit / quit |
| `fallback` | graceful "I didn't understand" |

Add a new intent by editing **one** place: `chatbot/intents.py::CATALOGUE`.
No other module needs to change.

---

## 7. Testing

```bash
pytest -v
```

The suite covers:

* `test_intents.py` - catalogue integrity, lookup helpers
* `test_nlu.py` - text cleaning, rule scorer, blended NLU on 30+ queries
* `test_semantic_matcher.py` - TF-IDF internals, cosine sim, vocabulary
* `test_response.py` - template substitution, conversation state
* `test_agent.py` - end-to-end multi-turn dialogue, followups, fallback

Plus the batch evaluation fixture `tests/fixtures/eval.jsonl` (30+ pairs)
exercised via `python main.py --eval`.

Full results are in [`EVIDENCE.md`](EVIDENCE.md).

---

## 8. Acceptance Criteria Mapping

| Criterion (from assignment) | Where it is satisfied |
|---|---|
| Core requirements completed and mapped to evidence | See `EVIDENCE.md` |
| Reproducible by a mentor | `requirements.txt`, deterministic tests, CLI documented |
| Quality checks documented; major issues disclosed | `EVIDENCE.md` section 5 |
| Professionally organised, named consistently | Repository layout above |
| Intern can explain decisions, trade-offs, limitations | `docs/DECISIONS.md` + `docs/ARCHITECTURE.md` |

---

## 9. Responsible Use

* The chatbot has **no access to user data**. It does not query the
  Arrowstack database; all responses are static text from the intent
  catalogue. No PII is stored.
* Session state lives in process memory only; nothing is written to disk
  unless the user explicitly enables logging.
* The agent is a **first-line deflection bot**. For account-specific
  questions (payment status, certificate URL), it directs the user to
  the portal or WhatsApp.

---

## 10. Author & Acknowledgements

**Bhoomika Gandhi** - Arrowstack AI Internship, Batch 2026
Ref ID: AS-2026-26671

Built with:

* Python 3.10+ standard library + NumPy
* TF-IDF / cosine similarity implemented from scratch (no scikit-learn)
* Inspired by the classic AIML pattern-matching bots, modernised with
  TF-IDF semantic scoring.

This project is submitted as Task 2 of the Arrowstack Artificial
Intelligence internship, in partial fulfilment of the 1-month virtual
cohort requirement (Nov 8 - Dec 8, 2026).
