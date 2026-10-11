"""Answers supervisor questions from live data.

The AI only turns a question into a QueryIntent; every number in the answer is
computed here from the fleet and the current plan, so the assistant cannot
hallucinate train states.
"""
from __future__ import annotations

from datetime import date

from .fleet import branding_status, open_jobs
from .models import Fleet, Plan, QueryIntent, TrainHealth

ASSIGNMENTS = {"Service", "Standby", "Inspection Bay", "Cleaning", "Held"}
READINESS = {"Fit", "Warning", "Unfit"}
HELP = (
    "I can answer questions about the fleet and tonight's plan, for example:\n"
    "• How many trains are in service tomorrow?\n"
    "• Which trains are unfit?\n"
    "• Why is TS-009 on standby?\n"
    "• Tell me about TS-014\n"
    "• Which certificates expire in the next 7 days?\n"
    "• Show open critical job cards\n"
    "• How are the branding contracts doing?\n"
    "• Which trains have the highest wear?"
)


def _status_of(name: str, plan: Plan, health: dict[int, TrainHealth], status: str) -> bool:
    if status in ASSIGNMENTS:
        return any(d.train_name == name and d.assignment.value == status for d in plan.decisions)
    return any(h.train_name == name and h.readiness.value == status for h in health.values())


def answer(intent: QueryIntent, fleet: Fleet, plan: Plan, health: dict[int, TrainHealth], today: date) -> str:
    names = [t.name for t in fleet.trains]
    decision = {d.train_name: d for d in plan.decisions}
    train = next((t for t in fleet.trains if t.name == intent.train_name), None)

    if intent.intent in ("count", "list"):
        status = _normalize_status(intent.status)
        if status is None:
            return f"I didn't recognise the status '{intent.status}'. Try Service, Standby, Inspection Bay, Cleaning, Held, Fit, Warning or Unfit."
        hits = [n for n in names if _status_of(n, plan, health, status)]
        label = f"assigned to {status}" if status in ASSIGNMENTS else f"rated {status}"
        if intent.intent == "count":
            return f"{len(hits)} of {len(names)} trainsets are {label}."
        return f"{len(hits)} trainsets {label}:\n" + (", ".join(hits) if hits else "none")

    if intent.intent in ("train_detail", "explain_assignment"):
        if train is None:
            return f"I couldn't find train '{intent.train_name}'. Trains run from KMRL-TS-001 to KMRL-TS-{len(names):03d}."
        d = decision[train.name]
        h = health[train.id]
        if intent.intent == "explain_assignment":
            return f"{train.name} → {d.assignment.value}\n" + "\n".join(f"• {r}" for r in d.reasons)
        certs = sorted((c for c in fleet.certificates if c.train_id == train.id), key=lambda c: c.department)
        jobs = open_jobs(fleet, train.id)
        lines = [
            f"{train.name} — {h.readiness.value}, planned for {d.assignment.value}",
            f"Stabled at {train.stabling_track} position {train.stabling_position}",
            f"Wear: {train.km_since_maintenance:,} km since inspection · {train.total_km:,} km lifetime",
            "Certificates: " + ", ".join(f"{c.department} until {c.expires.isoformat()}" for c in certs),
            f"Last deep clean: {train.last_deep_clean.isoformat()}" + (" (flagged)" if train.cleaning_flagged else ""),
            "Open job cards: " + ("; ".join(f"#{j.id} {j.priority} — {j.description}" for j in jobs) if jobs else "none"),
        ]
        if h.blockers:
            lines.append("Blockers: " + "; ".join(h.blockers))
        return "\n".join(lines)

    if intent.intent == "expiring_certificates":
        days = intent.days or 7
        by_id = {t.id: t.name for t in fleet.trains}
        rows = sorted(
            (c for c in fleet.certificates if (c.expires - today).days <= days),
            key=lambda c: c.expires,
        )
        if not rows:
            return f"No fitness certificates expire in the next {days} days."
        return f"Certificates expired or expiring within {days} days:\n" + "\n".join(
            f"• {by_id[c.train_id]} {c.department}: {c.expires.isoformat()}"
            + (" (EXPIRED)" if c.expires < today else "") for c in rows
        )

    if intent.intent == "open_jobs":
        by_id = {t.id: t.name for t in fleet.trains}
        rank = {"Critical": 0, "High": 1, "Routine": 2}
        jobs = sorted(
            (j for j in fleet.job_cards if j.status == "Open" and (intent.priority is None or j.priority == intent.priority)),
            key=lambda j: (rank[j.priority], j.id),
        )
        label = f"open {intent.priority.lower()} job cards" if intent.priority else "open job cards"
        if not jobs:
            return f"There are no {label}."
        return f"{len(jobs)} {label}:\n" + "\n".join(f"• #{j.id} {by_id[j.train_id]} [{j.priority}] {j.description}" for j in jobs)

    if intent.intent == "branding_status":
        by_id = {t.id: t.name for t in fleet.trains}
        lines = []
        for b in fleet.branding:
            s = branding_status(b, today)
            planned = decision[by_id[b.train_id]].assignment.value
            state = "AT RISK" if s.at_risk else (f"{s.deficit_hours:.0f} h behind" if s.deficit_hours > 0 else "on track")
            lines.append(f"• {b.advertiser} on {by_id[b.train_id]}: {b.delivered_hours:.0f}/{b.target_hours} h, {state}, {s.days_left} days left — planned {planned}")
        return "Branding contracts:\n" + "\n".join(lines)

    if intent.intent == "highest_wear":
        n = max(1, min(intent.limit or 5, len(names)))
        top = sorted(fleet.trains, key=lambda t: t.km_since_maintenance, reverse=True)[:n]
        return f"Highest wear since last inspection:\n" + "\n".join(
            f"• {t.name}: {t.km_since_maintenance:,} km ({decision[t.name].assignment.value})" for t in top
        )

    if intent.intent == "plan_summary":
        k = plan.kpis
        lines = [
            f"Plan for {plan.plan_date.isoformat()}: {k.service}/{k.service_target} in service, {k.standby} standby, "
            f"{k.inspection} in inspection bay, {k.cleaning} cleaning, {k.held} held.",
            f"Shunting moves: {k.shunting_moves}. Branded trains in service: {k.branding_trains_in_service}/{k.branding_trains_total}.",
        ]
        if plan.alerts:
            lines.append("Alerts:\n" + "\n".join(f"• {a}" for a in plan.alerts))
        return "\n".join(lines)

    return HELP


def _normalize_status(raw: str | None) -> str | None:
    if not raw:
        return None
    r = raw.strip().lower()
    aliases = {
        "service": "Service", "in service": "Service", "inducted": "Service", "operational": "Fit",
        "standby": "Standby", "inspection bay": "Inspection Bay", "ibl": "Inspection Bay", "inspection": "Inspection Bay",
        "maintenance": "Inspection Bay", "cleaning": "Cleaning", "held": "Held", "hold": "Held",
        "fit": "Fit", "green": "Fit", "warning": "Warning", "yellow": "Warning", "unfit": "Unfit",
        "red": "Unfit", "grounded": "Unfit",
    }
    return aliases.get(r) or next((s for s in ASSIGNMENTS | READINESS if s.lower() == r), None)
