"""Depot priority policy for new job cards.

The AI reads the defect; this policy decides how urgent it is. Keeping the decision in
code makes it consistent (Gemini, the on-device model and the rule engine all agree),
explainable (every override says why) and easy for the depot to change in one place.
The training data for the on-device model is labelled with this same function.

Order of precedence:
1. Words that demand urgency ("urgent", "withdraw immediately"...) -> Critical
2. A policy rule matches the defect -> that priority
   (a safety-critical rule can never be downgraded; others yield to "minor"/"routine")
3. Words that ask for low urgency ("minor", "cosmetic"...) -> Routine
4. No rule matches -> keep what the engine proposed
"""
from __future__ import annotations

import re

from .models import Priority

ESCALATE = re.compile(r"\b(urgent(ly)?|critical|safety|emergency|immediate(ly)?|withdraw|asap)\b", re.I)
DE_ESCALATE = re.compile(r"\b(minor|cosmetic|low[- ]priority|not urgent|routine)\b", re.I)

# (priority, label shown to the supervisor, pattern). First match wins, so specific rules come first.
RULES: list[tuple[Priority, str, re.Pattern]] = [
    # --- safety-critical: running gear, braking, train control, fire, doors that endanger passengers
    ("Critical", "brakes", re.compile(r"\bbrak(e|es|ing)\b", re.I)),
    ("Critical", "wheels and axles", re.compile(r"\b(wheel|flange|axle|bearing)s?\b", re.I)),
    ("Critical", "bogie and suspension", re.compile(r"\b(bogie|suspension|under-?frame)\b", re.I)),
    ("Critical", "train control (CBTC/ATP/ATO)", re.compile(r"\b(cbtc|atp|ato|balise|onboard unit|signall?ing (fault|failure))\b", re.I)),
    ("Critical", "smoke or fire", re.compile(r"\b(smoke|fire|burning|sparks?|short circuit)\b", re.I)),
    ("Critical", "door safety", re.compile(r"\bdoors?\b.*\b(obstruction|wrong side|while (moving|running)|not locking|open(ed|ing)? in motion)\b", re.I)),
    ("Critical", "derailment or collision", re.compile(r"\b(derail\w*|collision|hit an? |struck)\b", re.I)),
    # --- high: failures that affect passengers or traction
    ("High", "doors", re.compile(r"\bdoors?\b.*\b(not (closing|opening|working)|fail(ed|ure|s)?|stuck|won'?t|jammed|inoperative)\b", re.I)),
    ("High", "air conditioning", re.compile(r"\b(hvac|a/?c|air[- ]?con\w*|cooling)\b.*\b(not (working|cooling)|fail\w*|tripp?\w*|dead|no cooling|off)\b|\bno cooling\b", re.I)),
    ("High", "passenger information", re.compile(r"\b(pis|display|destination board|screen|announcements?|pa system|public address)\b.*\b(dead|blank|not (working|updating|audible)|fail\w*|off|inaudible)\b", re.I)),
    ("High", "CCTV", re.compile(r"\b(cctv|cameras?|camera feed)\b.*\b(offline|dead|not (working|recording)|fail\w*|lost|blank|off)\b", re.I)),
    ("High", "radio and communications", re.compile(r"\b(radio|intercom|talk-?back|emergency call)\b.*\b(not working|fail\w*|dead|no signal)\b", re.I)),
    ("High", "traction and power", re.compile(r"\b(pantograph|panto|traction|converter|inverter|motor)\b", re.I)),
    ("High", "headlights", re.compile(r"\b(head ?lights?|head ?lamps?)\b", re.I)),
    # --- routine: comfort, cosmetic and slow-but-working items
    ("Routine", "slow doors", re.compile(r"\bdoors?\b.*\b(slow(ly)?|delay\w*|sluggish|noisy|chime)\b", re.I)),
    ("Routine", "interior fittings", re.compile(r"\b(seat|grab ?pole|handrail|panel|floor|mat|window|sticker|light|lamp|led|wiper|rattl\w*)s?\b", re.I)),
    ("Routine", "minor noise", re.compile(r"\b(noise|squeak\w*|rattle)\b", re.I)),
]

RANK = {"Routine": 0, "High": 1, "Critical": 2}


def classify(text: str) -> tuple[Priority | None, str | None]:
    """Policy priority for a defect description, or (None, None) if no rule applies."""
    if ESCALATE.search(text):
        return "Critical", "marked urgent"
    for priority, label, pattern in RULES:
        if pattern.search(text):
            if priority != "Critical" and DE_ESCALATE.search(text):
                return "Routine", f"{label}, marked minor"
            return priority, label
    if DE_ESCALATE.search(text):
        return "Routine", "marked minor"
    return None, None


def apply_policy(description: str, source_text: str | None, proposed: Priority | None) -> tuple[Priority, str | None]:
    """Final priority and, if the policy changed the proposal, a note explaining why."""
    priority, label = classify(f"{description} {source_text or ''}")
    if priority is None:
        return proposed or "Routine", None
    if priority != proposed:
        return priority, f"priority set to {priority} by depot policy ({label})"
    return priority, None
