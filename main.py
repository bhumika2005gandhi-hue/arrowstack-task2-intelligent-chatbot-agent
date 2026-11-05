#!/usr/bin/env python3
"""
main.py
=======
Arrowstack AI Internship - Task 2
Intelligent Rule-Based & Semantic Chatbot Agent

Entry point. Supports two modes:

1. Interactive REPL::

       python main.py

2. Single-shot query (for scripting / testing)::

       python main.py --query "how do I download my offer letter"

3. Batch evaluation against a JSONL file of {query, expected_intent}::

       python main.py --eval tests/fixtures/eval.jsonl

Author: Bhoomika Gandhi (Arrowstack Batch 2026, AI domain)
Ref ID: AS-2026-26671
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import List, Optional

from chatbot.agent import ChatbotAgent
from chatbot.nlu import NLU
from chatbot.semantic_matcher import SemanticMatcher


# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------
def setup_logging(verbose: bool = False) -> logging.Logger:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return logging.getLogger("arrowstack.task2")


# ----------------------------------------------------------------------
# REPL
# ----------------------------------------------------------------------
BANNER = """
+---------------------------------------------------------------+
|  Arrowstack AI Chatbot - Task 2                                |
|  Intelligent Rule-Based & Semantic Chatbot Agent              |
|  Ref ID: AS-2026-26671                                         |
|  Type 'quit' or 'exit' to leave. Type 'help' for options.     |
+---------------------------------------------------------------+
""".strip()


def run_repl(agent: ChatbotAgent, session_id: str = "cli") -> None:
    print(BANNER)
    sid = agent.new_session(session_id)
    # Kick off with a greeting from the bot.
    boot = agent.respond("__boot__", session_id=sid)
    # __boot__ won't match anything; just print a welcome.
    print("\nBot: Hi! I am the Arrowstack AI assistant. Ask me anything about "
          "internships, offer letters, tasks, certificates, or the WhatsApp "
          "cohort group.\n")
    while True:
        try:
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBot: Goodbye!")
            break
        if not user_text:
            continue
        if user_text.lower() in {"quit", "exit"}:
            print("Bot: Goodbye! Reach out anytime on the portal or WhatsApp.")
            break
        result = agent.respond(user_text, session_id=sid)
        print(f"Bot: {result['response']}")
        if os.environ.get("CHATBOT_DEBUG"):
            print(f"  [intent={result['intent']} conf={result['confidence']:.3f}]")


# ----------------------------------------------------------------------
# Single-shot
# ----------------------------------------------------------------------
def run_query(agent: ChatbotAgent, query: str, explain: bool = False) -> dict:
    sid = agent.new_session("oneshot")
    result = agent.respond(query, session_id=sid)
    if explain:
        result["explanation"] = agent.explain(query, top_k=3)
    return result


# ----------------------------------------------------------------------
# Batch evaluation
# ----------------------------------------------------------------------
def run_eval(agent: ChatbotAgent, eval_path: str) -> dict:
    """Run every query in ``eval_path`` (JSONL: {"query":..., "expected":...}).

    Returns a summary dict with accuracy and per-intent stats.
    """
    if not os.path.exists(eval_path):
        raise FileNotFoundError(eval_path)
    correct = 0
    total = 0
    per_intent: dict = {}
    failures: list = []
    with open(eval_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            query = item.get("query", "")
            expected = item.get("expected", "")
            result = agent.respond(query, session_id=f"eval-{total}")
            got = result["intent"]
            total += 1
            per_intent.setdefault(expected, {"total": 0, "correct": 0})
            per_intent[expected]["total"] += 1
            if got == expected:
                correct += 1
                per_intent[expected]["correct"] += 1
            else:
                failures.append({
                    "query": query,
                    "expected": expected,
                    "got": got,
                    "confidence": result["confidence"],
                })
    accuracy = (correct / total) if total else 0.0
    return {
        "total": total,
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "per_intent": per_intent,
        "failures": failures,
    }


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Arrowstack AI Task 2 - Rule-Based & Semantic Chatbot",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--query", help="Run a single query and print the response as JSON.")
    p.add_argument("--eval", help="Evaluate accuracy against a JSONL file.")
    p.add_argument("--explain", action="store_true", help="Add top-3 document matches to single-query output.")
    p.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    setup_logging(args.verbose)
    agent = ChatbotAgent()

    if args.eval:
        summary = run_eval(agent, args.eval)
        print(json.dumps(summary, indent=2))
        return 0 if summary["accuracy"] >= 0.7 else 1
    if args.query:
        result = run_query(agent, args.query, explain=args.explain)
        print(json.dumps(result, indent=2))
        return 0
    # Default: REPL
    run_repl(agent)
    return 0


if __name__ == "__main__":
    sys.exit(main())
