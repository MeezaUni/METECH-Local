"""Deterministic, dependency-free parser for logbook text and questions.

Used as the fallback whenever the AI model is unavailable, and as a cross-check.
It handles the common phrasings depot staff use; the AI handles everything else.
"""
from __future__ import annotations

import calendar
import re
from datetime import date, timedelta

from ..models import FleetUpdate, QueryIntent

TRAIN_RE = re.compile(r"\b(?:KMRL[-\s]?)?(?:TS|trainset|train|rake|set|unit)[-\s#.]*0*(\d{1,3})\b", re.I)
JOB_RE = re.compile(r"\b(?:job[-\s]?card|jc|work[-\s]?order|wo|job)\s*(?:no\.?|number)?\s*#?\s*(\d{2,6})\b", re.I)
KM_RE = re.compile(r"\b(\d{1,3}(?:,\d{3})+|\d{2,6})\s*(?:km|kms|kilomet(?:er|re)s?)\b", re.I)

DEPT_PATTERNS = [
    ("Rolling-Stock", re.compile(r"\brolling[-\s]?stock\b|\bRS\b|\brolling\b", re.I)),
    ("Signalling", re.compile(r"\bsignal+ing\b|\bsignal\b|\bsig\b", re.I)),
    ("Telecom", re.compile(r"\btele[-\s]?com(?:munication)?s?\b|\bcomms\b", re.I)),
]
MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
MONTHS.update({m.lower(): i for i, m in enumerate(calendar.month_abbr) if m})
MONTHS["sept"] = 9
MONTH_RE = "|".join(sorted(MONTHS, key=len, reverse=True))

CRITICAL_WORDS = ("critical", "urgent", "safety", "wheel flat", "crack", "brake failure", "smoke", "fire", "derail", "emergency")
HIGH_WORDS = ("high priority", "major", "serious", "not working", "failed", "failure")


# ------------------------------------------------------------------ dates
_ABSOLUTE = [
    ("iso", re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")),
    ("dmy_num", re.compile(r"\b(\d{1,2})[/.-](\d{1,2})[/.-](\d{2,4})\b")),          # Indian day-first
    ("dmy", re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({MONTH_RE})\b\.?(?:,?\s*(\d{{4}})\b)?")),
    ("mdy", re.compile(rf"\b({MONTH_RE})\b\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?\b(?:,?\s*(\d{{4}})\b)?")),
]
_RELATIVE = re.compile(
    r"\b(?P<lead>(?:for|valid for)\s+)?(?P<mod>(?:another|next|in)\s+)?"
    r"(?P<n>\d{1,3}|an?|one|two|three|four|six|nine|twelve)?\s*(?P<unit>days?|weeks?|fortnights?|months?|years?)\b",
    re.I,
)
_WORDS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "six": 6, "nine": 9, "twelve": 12}


def find_date(text: str, today: date) -> tuple[str, date] | None:
    """Find a date expression; return (the exact span as written, resolved date).

    Absolute forms win over relative ones. Relative forms ("for 6 months", "for a fortnight",
    "next week") count from `today`. A day and month without a year means the next such date.
    """
    t = text.lower()
    for kind, rx in _ABSOLUTE:
        m = next((m for m in rx.finditer(t) if not (m.start() and t[m.start() - 1] in "-#/")), None)
        if not m:   # skip numbers glued to an id: "TS-05 may need cleaning" is not 5 May
            continue
        if kind == "iso":
            d = _safe_date(int(m[1]), int(m[2]), int(m[3]))
        elif kind == "dmy_num":
            d = _safe_date(int(m[3]) + (2000 if len(m[3]) == 2 else 0), int(m[2]), int(m[1]))
        elif kind == "dmy":
            d = _with_year(int(m[1]), MONTHS[m[2]], m[3], today)
        else:
            d = _with_year(int(m[2]), MONTHS[m[1]], m[3], today)
        if d:
            return text[m.start():m.end()], d
    if m := re.search(r"\bday after tomorrow\b", t):
        return text[m.start():m.end()], today + timedelta(days=2)
    if m := re.search(r"\btomorrow\b", t):
        return text[m.start():m.end()], today + timedelta(days=1)
    for m in _RELATIVE.finditer(t):
        if not (m["lead"] or m["mod"] or m["n"]):
            continue   # a bare "year" or "months" is not a date
        n = _WORDS.get(m["n"] or "a") or int(m["n"])
        unit = m["unit"].rstrip("s")
        if unit == "day":
            d = today + timedelta(days=n)
        elif unit == "week":
            d = today + timedelta(weeks=n)
        elif unit == "fortnight":
            d = today + timedelta(days=14 * n)
        elif unit == "month":
            d = _add_months(today, n)
        else:
            d = _add_months(today, 12 * n)
        return text[m.start():m.end()].strip(), d
    return None


def parse_date(text: str, today: date) -> date | None:
    found = find_date(text, today)
    return found[1] if found else None


def _safe_date(y: int, m: int, d: int) -> date | None:
    try:
        return date(y, m, d)
    except ValueError:
        return None


def _with_year(d: int, m: int, year: str | None, today: date) -> date | None:
    if year:
        return _safe_date(int(year), m, d)
    candidate = _safe_date(today.year, m, d)
    if candidate and candidate <= today:   # an expiry "on 11 Oct" written on 11 Oct means next year
        candidate = _safe_date(today.year + 1, m, d)
    return candidate


def _add_months(d: date, n: int) -> date:
    y, m = divmod(d.month - 1 + n, 12)
    y, m = d.year + y, m + 1
    return date(y, m, min(d.day, calendar.monthrange(y, m)[1]))


# ---------------------------------------------------------------- logbook
def _split_clauses(text: str) -> list[str]:
    parts = re.split(r"[!?;\n]+|\.+(?=\s|$)|\s+(?:also|then)\s+", text)
    clauses: list[str] = []
    for part in parts:
        part = part.strip(" ,")
        if not part:
            continue
        # Split further where a new train is mentioned after the first one.
        starts = [m.start() for m in TRAIN_RE.finditer(part)]
        if len(starts) <= 1:
            clauses.append(part)
            continue
        # Cut only where a later train mention follows a separator (",", "and", "while").
        chunk_start = 0
        for s in starts[1:]:
            if re.search(r"(,|\band|\bwhile|\bwhereas)\s*$", part[chunk_start:s], re.I):
                clauses.append(part[chunk_start:s].strip(" ,"))
                chunk_start = s
        clauses.append(part[chunk_start:].strip(" ,"))
    return [c for c in clauses if c]


def parse_logbook(text: str, today: date, job_owner: dict[int, str] | None = None) -> list[FleetUpdate]:
    """Extract FleetUpdates from free text. `job_owner` maps job card id -> train name."""
    job_owner = job_owner or {}
    updates: list[FleetUpdate] = []
    last_train: str | None = None

    for clause in _split_clauses(text):
        low = clause.lower()
        trains = [f"KMRL-TS-{int(n):03d}" for n in TRAIN_RE.findall(clause)]
        train = trains[0] if trains else last_train
        if trains:
            last_train = trains[0]

        jobs = [int(j) for j in JOB_RE.findall(clause)]
        closing = re.search(r"\b(clos|complet|done|resolv|fix|attend|finish|rectif|clear)", low)
        if jobs and closing:
            for jid in jobs:
                owner = job_owner.get(jid, train)
                if owner:
                    updates.append(FleetUpdate(train_name=owner, action="close_job_card", job_card_id=jid, source_text=clause))
            continue

        if train is None:
            continue

        if re.search(r"\b(certs?|certificates?|fitness|fc)\b", low) and re.search(r"\b(renew|valid|extend|issued|until|till|upto|up to|good|reissued|updated)", low):
            when = parse_date(clause, today)
            depts = [d for d, rx in DEPT_PATTERNS if rx.search(clause)]
            for dept in depts or []:
                updates.append(FleetUpdate(train_name=train, action="renew_certificate", department=dept,
                                           date=when.isoformat() if when else None, source_text=clause))
            if depts:
                continue

        if re.search(r"\b(inspection|a-check|b-check|ibl|maintenance|servicing|overhaul)\b", low) and re.search(r"\b(complet|done|finish|clear|over|carried out)", low):
            updates.append(FleetUpdate(train_name=train, action="complete_maintenance", source_text=clause))
            continue

        if re.search(r"\b(deep[-\s]?clean(?:ed|ing)? (?:done|complete|completed|finished)|cleaned|washed|detailing (?:done|complete|completed))\b", low) and not re.search(r"\b(needs?|requires?|due|pending|not)\b", low):
            updates.append(FleetUpdate(train_name=train, action="mark_cleaned", source_text=clause))
            continue

        if re.search(r"\b(dirty|graffiti|vandal|vomit|spill|stain|litter|needs? (?:a )?(?:deep )?clean|requires? (?:a )?(?:deep )?clean|clean(?:ing)? (?:due|required|needed))", low):
            updates.append(FleetUpdate(train_name=train, action="flag_cleaning", source_text=clause))
            continue

        if (km := KM_RE.search(clause)) and re.search(r"\b(since|odometer|reading|run|covered|clocked|mileage)\b", low):
            updates.append(FleetUpdate(train_name=train, action="update_mileage", km=int(km[1].replace(",", "")), source_text=clause))
            continue

        if re.search(r"\b(fault|defect|issue|problem|fail|failure|not working|broken|leak|noise|damaged|reported|raise|faulty|stuck|malfunction|worn|crack|slow|intermittent|error|alarm|tripping|vibration|loose|dead|blank|flicker)", low):
            priority = "Critical" if any(w in low for w in CRITICAL_WORDS) else ("High" if any(w in low for w in HIGH_WORDS) else "Routine")
            desc = TRAIN_RE.sub("", clause)
            desc = re.sub(r"^\W+|\b(reported|raise[d]?|on|for|in)\s*$", "", desc.strip(), flags=re.I).strip(" ,:-")
            updates.append(FleetUpdate(train_name=train, action="open_job_card", description=desc[:1].upper() + desc[1:] if desc else clause,
                                       priority=priority, source_text=clause))
    return updates


# ---------------------------------------------------------------- questions
STATUS_WORDS = [
    ("Service", r"\b(in service|inducted|running|revenue|service)\b"),
    ("Standby", r"\b(standby|stand-by|spare|reserve)\b"),
    ("Inspection Bay", r"\b(inspection bay|ibl|inspection|maintenance bay)\b"),
    ("Cleaning", r"\b(cleaning|being cleaned|wash)\b"),
    ("Held", r"\b(held|hold|stuck)\b"),
    ("Unfit", r"\b(unfit|grounded|red|not fit|out of service|blocked)\b"),
    ("Warning", r"\b(warning|yellow|amber|attention|flagged)\b"),
    ("Fit", r"\b(fit|green|healthy|ok|operational|available)\b"),
]


def interpret_question(question: str) -> QueryIntent:
    q = question.lower()
    m = TRAIN_RE.search(question)
    train = f"KMRL-TS-{int(m[1]):03d}" if m else None
    status = next((s for s, rx in STATUS_WORDS if re.search(rx, q)), None)
    days = int(d[1]) if (d := re.search(r"\b(\d{1,3})\s*days?\b", q)) else None
    limit = int(n[1]) if (n := re.search(r"\btop\s+(\d{1,2})\b", q)) else None

    if train and re.search(r"\b(why|reason|explain|how come)\b", q):
        return QueryIntent(intent="explain_assignment", train_name=train)
    if train:
        return QueryIntent(intent="train_detail", train_name=train)
    if re.search(r"\b(certs?|certificates?|fitness)\b", q) and re.search(r"\b(expir|due|lapse|renew)", q):
        return QueryIntent(intent="expiring_certificates", days=days or 7)
    if re.search(r"\b(jobs?|job cards?|work orders?|defects?|maintenance tasks?|pending work)\b", q):
        pr = next((p for p in ("Critical", "High", "Routine") if p.lower() in q), None)
        return QueryIntent(intent="open_jobs", priority=pr)
    if re.search(r"\b(brand|advert|sla|exposure|wrap)", q):
        return QueryIntent(intent="branding_status")
    if re.search(r"\b(mileage|wear|km|kilomet|most used|highest)", q):
        return QueryIntent(intent="highest_wear", limit=limit or 5)
    if re.search(r"\b(plan|summary|tonight|tomorrow|overview|schedule)\b", q) and not status:
        return QueryIntent(intent="plan_summary")
    if status and re.search(r"\b(how many|count|number of)\b", q):
        return QueryIntent(intent="count", status=status)
    if status:
        return QueryIntent(intent="list", status=status)
    if re.search(r"\b(plan|summary|overview)\b", q):
        return QueryIntent(intent="plan_summary")
    return QueryIntent(intent="help")
