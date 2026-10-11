"""Applying FleetUpdates. The single write path for manual edits, AI suggestions and simulations."""
from __future__ import annotations

import re
from datetime import date

from .models import DEPARTMENTS, Certificate, Fleet, FleetUpdate, JobCard
from .priority import apply_policy

_TRAIN_NUM = re.compile(r"(\d{1,3})\s*$")


def normalize_train_name(raw: str | None, fleet: Fleet) -> str | None:
    """'TS-4', 'train 04', '4', 'kmrl-ts-004' -> 'KMRL-TS-004' (if that train exists)."""
    if not raw:
        return None
    names = {t.name for t in fleet.trains}
    candidate = raw.strip().upper()
    if candidate in names:
        return candidate
    m = _TRAIN_NUM.search(candidate)
    if m:
        name = f"KMRL-TS-{int(m.group(1)):03d}"
        if name in names:
            return name
    return None


def validate_updates(updates: list[FleetUpdate], fleet: Fleet, today: date) -> tuple[list[FleetUpdate], list[str]]:
    """Normalise and sanity-check updates. Returns (valid updates, human-readable warnings)."""
    valid: list[FleetUpdate] = []
    warnings: list[str] = []
    jobs = {j.id: j for j in fleet.job_cards}
    train_by_id = {t.id: t for t in fleet.trains}

    for u in updates:
        u = u.model_copy()
        if u.action == "close_job_card" and u.job_card_id in jobs:
            # The job card number is authoritative for which train it belongs to.
            owner = train_by_id[jobs[u.job_card_id].train_id].name
            given = normalize_train_name(u.train_name, fleet)
            if given and given != owner:
                warnings.append(f"Job card #{u.job_card_id} belongs to {owner}, not {given} — corrected")
            u.train_name = owner
        name = normalize_train_name(u.train_name, fleet)
        if name is None:
            warnings.append(f"Skipped: unknown train '{u.train_name}'")
            continue
        u.train_name = name

        problem = _check(u, jobs, today)
        if problem:
            warnings.append(f"Skipped {name} {u.action.replace('_', ' ')}: {problem}")
            continue
        if u.action == "open_job_card":
            # The engine reads the defect; the depot policy decides how urgent it is (app/priority.py).
            u.priority, note = apply_policy(u.description or "", u.source_text, u.priority)
            if note:
                warnings.append(f"{name}: {note}")
        valid.append(u)
    return valid, warnings


def _check(u: FleetUpdate, jobs: dict[int, JobCard], today: date) -> str | None:
    if u.action == "renew_certificate":
        if u.department not in DEPARTMENTS:
            return "department must be Rolling-Stock, Signalling or Telecom"
        try:
            expires = date.fromisoformat(u.date or "")
        except ValueError:
            return f"invalid expiry date '{u.date}'"
        if expires <= today:
            return f"new expiry {expires.isoformat()} is not in the future"
    elif u.action == "close_job_card":
        if u.job_card_id not in jobs:
            return f"job card #{u.job_card_id} not found"
        if jobs[u.job_card_id].status == "Closed":
            return f"job card #{u.job_card_id} is already closed"
    elif u.action == "open_job_card":
        if not (u.description or "").strip():
            return "a description is required"
    elif u.action == "update_mileage":
        if u.km is None or not 0 <= u.km <= 50_000:
            return f"km since inspection must be between 0 and 50,000 (got {u.km})"
    return None


def apply_updates(fleet: Fleet, updates: list[FleetUpdate], today: date) -> tuple[Fleet, list[str]]:
    """Return a new Fleet with `updates` applied (input is not mutated) and one summary line per update.

    Updates must already have passed `validate_updates`.
    """
    f = fleet.model_copy(deep=True)
    trains = {t.name: t for t in f.trains}
    summaries: list[str] = []

    for u in updates:
        t = trains[u.train_name]
        if u.action == "renew_certificate":
            cert = next((c for c in f.certificates if c.train_id == t.id and c.department == u.department), None)
            expires = date.fromisoformat(u.date)
            if cert is None:
                f.certificates.append(Certificate(
                    id=max((c.id for c in f.certificates), default=0) + 1, train_id=t.id,
                    department=u.department, issued=today, expires=expires,
                ))
            else:
                cert.issued, cert.expires = today, expires
            summaries.append(f"{t.name}: {u.department} certificate valid until {expires.isoformat()}")
        elif u.action == "close_job_card":
            job = next(j for j in f.job_cards if j.id == u.job_card_id)
            job.status, job.closed = "Closed", today
            summaries.append(f"{t.name}: job card #{job.id} closed ({job.description})")
        elif u.action == "open_job_card":
            job = JobCard(
                id=max((j.id for j in f.job_cards), default=1000) + 1, train_id=t.id,
                system=_guess_system(u.description or ""), description=u.description.strip(),
                priority=u.priority or "Routine", status="Open", opened=today,
            )
            f.job_cards.append(job)
            summaries.append(f"{t.name}: job card #{job.id} opened ({job.priority}) — {job.description}")
        elif u.action == "flag_cleaning":
            t.cleaning_flagged = True
            summaries.append(f"{t.name}: flagged for deep cleaning")
        elif u.action == "mark_cleaned":
            t.cleaning_flagged, t.last_deep_clean = False, today
            summaries.append(f"{t.name}: deep clean completed")
        elif u.action == "complete_maintenance":
            t.km_since_maintenance = 0
            summaries.append(f"{t.name}: inspection completed, wear counter reset")
        elif u.action == "update_mileage":
            t.km_since_maintenance = u.km
            summaries.append(f"{t.name}: {u.km:,} km since last inspection")
    return f, summaries


_SYSTEM_WORDS = {
    "Brakes": ("brake",), "HVAC": ("hvac", "ac ", "air con", "cooling", "temperature"),
    "Doors": ("door",), "Bogie": ("bogie", "wheel", "axle", "suspension"),
    "Pantograph": ("pantograph", "panto", "carbon strip"),
    "Telecom": ("pis", "cctv", "radio", "display", "announcement", "camera"),
    "Signalling": ("cbtc", "atp", "ato", "balise", "signal"),
    "Interior": ("seat", "light", "grab", "floor", "window", "interior"),
}


def _guess_system(description: str) -> str:
    d = description.lower() + " "
    return next((s for s, words in _SYSTEM_WORDS.items() if any(w in d for w in words)), "General")
