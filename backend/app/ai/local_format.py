"""Prompt and output format for the on-device model.

Single source of truth shared by the training pipeline (ml/) and the backend, so
the model is always queried exactly the way it was trained.
"""
from __future__ import annotations

import json
import re
from datetime import date

PARSE_SYSTEM = (
    "You are the KMRL depot copilot. Extract fleet updates from the logbook text. "
    "Reply with JSON only: {\"updates\": [...]}. Train ids are KMRL-TS-NNN. "
    "Actions: renew_certificate(department,date), close_job_card(job_card_id), "
    "open_job_card(description,priority), flag_cleaning, mark_cleaned, complete_maintenance, update_mileage(km). "
    "Resolve dates against today. Return an empty list when nothing changes."
)

ASK_SYSTEM = (
    "You are the KMRL depot copilot. Convert the supervisor's question into a JSON query. "
    "Intents: count, list, train_detail, explain_assignment, expiring_certificates, open_jobs, "
    "branding_status, plan_summary, highest_wear, help. Reply with JSON only."
)

# Field order the model is trained to emit (keeps outputs short and predictable).
UPDATE_FIELDS = ("train_name", "action", "department", "date", "job_card_id", "description", "priority", "km", "source_text")
INTENT_FIELDS = ("intent", "status", "train_name", "days", "priority", "limit")


def parse_messages(text: str, today: date) -> list[dict]:
    return [
        {"role": "system", "content": PARSE_SYSTEM},
        {"role": "user", "content": f"Today: {today.isoformat()}\nLogbook:\n{text.strip()}"},
    ]


def ask_messages(question: str) -> list[dict]:
    return [
        {"role": "system", "content": ASK_SYSTEM},
        {"role": "user", "content": question.strip()},
    ]


def dump_updates(updates: list[dict]) -> str:
    """Canonical target string for a list of update dicts."""
    clean = [{k: u[k] for k in UPDATE_FIELDS if u.get(k) not in (None, "")} for u in updates]
    return json.dumps({"updates": clean}, ensure_ascii=False, separators=(",", ":"))


def dump_intent(intent: dict) -> str:
    return json.dumps({k: intent[k] for k in INTENT_FIELDS if intent.get(k) is not None}, ensure_ascii=False, separators=(",", ":"))


_JSON_OBJECT = re.compile(r"\{.*\}", re.S)


def load_json(raw: str) -> dict:
    """Parse the model's reply, tolerating stray text around the JSON object."""
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = _JSON_OBJECT.search(raw)
        if not m:
            raise ValueError(f"no JSON object in model output: {raw[:120]!r}")
        return json.loads(m.group(0))
