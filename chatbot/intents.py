"""
chatbot/intents.py
==================
Intent catalogue for the rule-based / semantic chatbot.

Each intent has:
* a unique ``name``
* a list of ``patterns`` - example user utterances used by the semantic
  matcher to score similarity against the user's query
* a list of ``responses`` - candidate reply templates (one is chosen
  deterministically per turn to keep tests reproducible)
* an optional ``followup`` - the next intent the bot should expect, used
  to drive multi-turn conversations (e.g. greeting -> help)

Adding a new intent is a single edit here - no other module needs to
change. The NLU layer picks intents up automatically.

Author: Bhoomika Gandhi (Arrowstack Batch 2026, AI domain)
Ref ID: AS-2026-26671
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Intent:
    name: str
    patterns: List[str]
    responses: List[str]
    followup: Optional[str] = None
    priority: int = 0  # higher = preferred when scores tie

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "patterns": list(self.patterns),
            "responses": list(self.responses),
            "followup": self.followup,
            "priority": self.priority,
        }


# ----------------------------------------------------------------------
# Catalogue
# ----------------------------------------------------------------------
CATALOGUE: List[Intent] = [
    Intent(
        name="greeting",
        patterns=[
            "hi",
            "hello",
            "hey there",
            "good morning",
            "good evening",
            "namaste",
        ],
        responses=[
            "Hello! I am the Arrowstack AI assistant. How can I help you today?",
            "Hi there! Ask me about internships, certificates, or task submissions.",
        ],
        followup="help",
        priority=1,
    ),
    Intent(
        name="help",
        patterns=[
            "what can you do",
            "help",
            "help me",
            "what do you support",
            "options",
            "menu",
        ],
        responses=[
            "I can answer questions about: internships, offer letters, task submission, "
            "certificate verification, and batch WhatsApp groups. Just ask!",
        ],
    ),
    Intent(
        name="internship_list",
        patterns=[
            "what internships do you offer",
            "list of internships",
            "which domains",
            "available tracks",
            "internship domains",
        ],
        responses=[
            "Arrowstack offers 40+ virtual internship tracks including Artificial "
            "Intelligence, Cloud Computing, Full Stack Development, Frontend "
            "Development, Data Science, DevOps, Cybersecurity, and many more. "
            "Visit the Internships page on the portal to apply.",
        ],
    ),
    Intent(
        name="offer_letter",
        patterns=[
            "how do i get my offer letter",
            "where is my offer letter",
            "download offer letter",
            "offer letter pdf",
            "when will i get offer letter",
        ],
        responses=[
            "Your official Offer Letter is generated instantly upon successful "
            "registration. Log in to the Intern Portal and click 'Download (PDF)' "
            "next to your registration entry. Each student sees only their own letter.",
        ],
    ),
    Intent(
        name="task_submission",
        patterns=[
            "how to submit tasks",
            "where do i submit project",
            "task submission link",
            "submit github link",
            "how to upload task",
        ],
        responses=[
            "Open the 'Internship Task Submission Form' on your portal, paste your "
            "two GitHub project URLs and the two LinkedIn post links, then click "
            "Submit. Tasks can only be submitted once per domain, so review before "
            "you submit.",
        ],
    ),
    Intent(
        name="certificate",
        patterns=[
            "when will i get certificate",
            "certificate release date",
            "verified certificate",
            "iso certificate",
            "how to get certificate",
        ],
        responses=[
            "Your verified Internship Completion Certificate (with ISO 9001:2015 "
            "seal and QR) is released on the final day of your 1-month cohort "
            "tenure, after the Rs.99 verification fee is paid. Check your portal "
            "for the exact release date.",
        ],
    ),
    Intent(
        name="verification_fee",
        patterns=[
            "verification fee",
            "99 rupees",
            "payment for certificate",
            "how much is verification",
            "pay verification",
            "easypaisa 400 pkr",
            "pakistan payment easypaisa",
            "upi payment verification",
        ],
        responses=[
            "The standard verification fee is Rs.99 (India / Cards / UPI). "
            "Students from Pakistan can pay 400 PKR via EasyPaisa. The fee "
            "reserves your certificate for release on cohort completion day.",
        ],
    ),
    Intent(
        name="whatsapp_group",
        patterns=[
            "whatsapp group link",
            "join cohort group",
            "batch whatsapp",
            "mentor support",
            "telegram channel",
        ],
        responses=[
            "Join your registered Batch 2026 cohort WhatsApp group from the "
            "portal's 'Official Batch 2026 Cohort WhatsApp Group & Channels' "
            "section. Live mentor doubts and announcements happen there.",
        ],
    ),
    Intent(
        name="deadline",
        patterns=[
            "task deadline",
            "submission deadline",
            "when is task due",
            "last date",
            "due date",
        ],
        responses=[
            "Task submission deadlines are listed on your portal under each "
            "domain. Submit before the deadline to remain eligible for the "
            "verified certificate.",
        ],
    ),
    Intent(
        name="refund_policy",
        patterns=[
            "refund policy",
            "money back",
            "cancel payment refund",
            "refund verification fee",
            "can i get my money back",
        ],
        responses=[
            "Please read the Payment & Refund Policy on the Arrowstack portal "
            "footer. Verification fees are non-refundable once the certificate "
            "is reserved for release.",
        ],
    ),
    Intent(
        name="thanks",
        patterns=[
            "thank you",
            "thanks",
            "thank you so much",
            "appreciate it",
        ],
        responses=[
            "You're welcome! Good luck with your internship. :)",
            "Happy to help. Don't forget to join the WhatsApp cohort group!",
        ],
    ),
    Intent(
        name="goodbye",
        patterns=[
            "bye",
            "goodbye",
            "see you",
            "later",
            "exit",
            "quit",
        ],
        responses=[
            "Goodbye! Reach out anytime on the portal or WhatsApp.",
            "See you later - and don't miss your task deadline!",
        ],
    ),
    Intent(
        name="fallback",
        patterns=[],  # fallback is selected when nothing else matches
        responses=[
            "I'm not sure I understood that. Could you rephrase? You can ask "
            "about internships, offer letters, task submission, certificate "
            "release, the verification fee, or the WhatsApp cohort group.",
            "Sorry, I didn't catch that. Type 'help' to see what I can do.",
        ],
        priority=-1,
    ),
]


def get_catalogue() -> List[Intent]:
    """Return a copy of the catalogue."""
    return list(CATALOGUE)


def get_intent(name: str) -> Optional[Intent]:
    """Look up an intent by name."""
    for intent in CATALOGUE:
        if intent.name == name:
            return intent
    return None


def get_intent_map() -> Dict[str, Intent]:
    """Return a name -> Intent mapping for quick lookup."""
    return {intent.name: intent for intent in CATALOGUE}
