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
# Certificates carry `date_text`, the expiry exactly as written ("for 6 months", "10/10/2027"):
# the model copies it, and code (rules.find_date) does the calendar arithmetic. `date` is
# accepted only for models trained before this change.
UPDATE_FIELDS = ("train_name", "action", "department", "date_text", "date", "job_card_id", "description", "priority", "km", "source_text")
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


_NUMBER = re.compile(r"\d[\d,]*")
_SPACES = re.compile(r"\s+")


def _norm(s: str) -> str:
    return _SPACES.sub(" ", s.lower()).strip()


def ground_and_resolve(updates: list[dict], text: str, today: date) -> tuple[list[dict], list[str]]:
    """Check each extracted update against the message and turn date phrases into dates.

    An update is kept only if every number it relies on (train, job card, km) and its expiry
    phrase actually occur in the text: a small model must not invent facts. Returns
    (kept updates, reasons for anything dropped).
    """
    from .rules import find_date   # local import: rules imports models, keep this module light

    from typing import get_args

    from ..models import UpdateAction

    actions = set(get_args(UpdateAction))
    numbers = {int(n.replace(",", "")) for n in _NUMBER.findall(text)}
    norm_text = _norm(text)
    kept, dropped = [], []
    for u in updates:
        if not isinstance(u, dict):
            continue
        u = dict(u)
        if u.get("action") not in actions:   # e.g. an invented "reprofiling_done"
            dropped.append(f"Ignored a suggestion with an unknown action '{u.get('action')}'")
            continue
        name = u.get("train_name") or ""
        problems = []
        if name[-3:].isdigit() and int(name[-3:]) not in numbers:
            problems.append(f"train {name} is not mentioned")
        if u.get("job_card_id") is not None and int(u["job_card_id"]) not in numbers:
            problems.append(f"job card #{u['job_card_id']} is not mentioned")
        if u.get("km") is not None and int(u["km"]) not in numbers:
            problems.append(f"{u['km']} km is not mentioned")
        if u.get("action") == "renew_certificate" and u.get("date_text"):
            found = find_date(u["date_text"], today)
            if _norm(u["date_text"]) not in norm_text:
                problems.append(f"expiry '{u['date_text']}' is not in the text")
            elif found is None:
                problems.append(f"could not read the expiry '{u['date_text']}'")
            else:
                u["date"] = found[1].isoformat()
        u.pop("date_text", None)
        if problems:
            dropped.append(f"Ignored a suggested {u.get('action', 'update').replace('_', ' ')}: " + "; ".join(problems))
        else:
            kept.append(u)
    return kept, dropped


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
