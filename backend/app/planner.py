"""Nightly induction planner.

Assigns every trainset to exactly one of Service / Standby / Inspection Bay /
Cleaning / Held with Google OR-Tools CP-SAT, balancing the six factors in the
KMRL problem statement:

1. Fitness certificates   - hard: any expired certificate keeps a train out of service
2. Job cards              - hard for Critical; open High/Routine work prefers the IBL
3. Branding commitments   - trains behind on contracted exposure are preferred for service
4. Mileage balancing      - low km-since-inspection trains are preferred for service
5. Cleaning slots         - overdue trains compete for limited deep-cleaning bays
6. Stabling geometry      - inducting a train parked behind a non-departing one costs a shunt

Every decision is returned with human-readable reasons and a score breakdown so a
supervisor can see *why*, not just *what*.
"""
from __future__ import annotations

import time
from datetime import date, datetime, timedelta

from ortools.sat.python import cp_model

from .config import PlannerConfig
from .fleet import assess_fleet, branding_status, cleaning_overdue_days, needs_cleaning, open_jobs
from .models import (
    Assignment, Fleet, Plan, PlanChange, PlanKPIs, Readiness, ScoreBreakdown, Train, TrainDecision,
)

S = Assignment
STATES = (S.SERVICE, S.STANDBY, S.IBL, S.CLEANING, S.HOLD)

# Objective weights (costs, minimised). Shortfalls dominate everything else.
W_SERVICE_SHORTFALL = 100_000
W_STANDBY_SHORTFALL = 20_000
W_HOLD = {"critical": 9_000, "overdue": 7_000, "cert": 5_000}
W_OPEN_JOB = {"High": 1_500, "Routine": 600}
W_CLEAN_DUE = 1_200
W_CLEAN_PER_DAY = 150
W_CLEAN_FLAGGED = 2_500
W_CLEAN_WASTED = 300
W_BRANDING_MAX = 3_000
W_BRANDING_KEEP = 300
W_MILEAGE = 900
W_CERT_RISK = 700
W_SHUNT = 250
W_INSPECTION_DUE = 800


def _fmt_km(km: int) -> str:
    return f"{km:,} km"


def generate_plan(fleet: Fleet, cfg: PlannerConfig, today: date | None = None) -> Plan:
    today = today or date.today()
    started = time.perf_counter()
    health = assess_fleet(fleet, today, cfg)
    trains = fleet.trains
    by_id = {t.id: t for t in trains}
    brand = {b.train_id: branding_status(b, today) for b in fleet.branding}

    fit_ids = [t.id for t in trains if health[t.id].readiness != Readiness.UNFIT]
    wear = [by_id[i].km_since_maintenance for i in fit_ids] or [0]
    wear_lo, wear_hi = min(wear), max(wear)

    m = cp_model.CpModel()
    x = {(t.id, s): m.NewBoolVar(f"x_{t.id}_{s.name}") for t in trains for s in STATES}
    costs: list = []
    # Per-train cost expressions, kept so the explanation can show the same numbers the solver used.
    service_value: dict[int, dict[str, int]] = {}

    for t in trains:
        m.AddExactlyOne(x[t.id, s] for s in STATES)
        h = health[t.id]
        jobs = open_jobs(fleet, t.id)

        if h.readiness == Readiness.UNFIT:
            for s in (S.SERVICE, S.STANDBY, S.CLEANING):
                m.Add(x[t.id, s] == 0)
            if any(j.priority == "Critical" for j in jobs):
                hold_cost = W_HOLD["critical"]
            elif t.km_since_maintenance >= cfg.maint_limit_km:
                hold_cost = W_HOLD["overdue"]
            else:
                hold_cost = W_HOLD["cert"]
            costs.append(hold_cost * x[t.id, S.HOLD])
            continue

        m.Add(x[t.id, S.HOLD] == 0)

        # Open non-critical work prefers the Inspection Bay when there is room.
        for j in jobs:
            costs.append(W_OPEN_JOB[j.priority] * (1 - x[t.id, S.IBL]))
        if t.km_since_maintenance >= cfg.maint_warning_km:
            ratio = (t.km_since_maintenance - cfg.maint_warning_km) / max(1, cfg.maint_limit_km - cfg.maint_warning_km)
            costs.append(int(200 + W_INSPECTION_DUE * ratio) * (1 - x[t.id, S.IBL]))
        if not jobs and t.km_since_maintenance < cfg.maint_warning_km:
            costs.append(W_CLEAN_WASTED * x[t.id, S.IBL])   # don't waste IBL slots on healthy trains

        # Cleaning
        if t.cleaning_flagged:
            costs.append(W_CLEAN_FLAGGED * (1 - x[t.id, S.CLEANING]))
        elif needs_cleaning(t, today, cfg):
            overdue = cleaning_overdue_days(t, today, cfg)
            costs.append((W_CLEAN_DUE + W_CLEAN_PER_DAY * overdue) * (1 - x[t.id, S.CLEANING]))
        else:
            costs.append(W_CLEAN_WASTED * x[t.id, S.CLEANING])

        # Service desirability (negative cost = bonus)
        value: dict[str, int] = {}
        b = brand.get(t.id)
        if b is not None:
            value["branding"] = int(min(W_BRANDING_MAX, b.deficit_ratio * 4 * W_BRANDING_MAX)) + W_BRANDING_KEEP
        norm = (t.km_since_maintenance - wear_lo) / max(1, wear_hi - wear_lo)
        value["mileage"] = int(W_MILEAGE * (0.5 - norm))
        if any("expires" in w for w in h.warnings):
            value["risk"] = -W_CERT_RISK
        service_value[t.id] = value
        costs.append(-sum(value.values()) * x[t.id, S.SERVICE])

    # Stabling geometry: a train leaving for service from behind a train that stays costs a shunt.
    shunts: dict[tuple[int, int], cp_model.IntVar] = {}
    for t in trains:
        for a in trains:
            if a.stabling_track == t.stabling_track and a.stabling_position < t.stabling_position:
                v = m.NewBoolVar(f"shunt_{a.id}_{t.id}")
                m.Add(v >= x[t.id, S.SERVICE] - x[a.id, S.SERVICE])
                shunts[a.id, t.id] = v
                costs.append(W_SHUNT * v)

    # Depot capacities and timetable requirements
    m.Add(sum(x[t.id, S.IBL] for t in trains) <= cfg.ibl_slots)
    m.Add(sum(x[t.id, S.CLEANING] for t in trains) <= cfg.cleaning_slots)
    n_service = sum(x[t.id, S.SERVICE] for t in trains)
    m.Add(n_service <= cfg.service_target)
    service_short = m.NewIntVar(0, cfg.service_target, "service_short")
    m.Add(service_short == cfg.service_target - n_service)
    standby_short = m.NewIntVar(0, cfg.min_standby, "standby_short")
    m.Add(standby_short >= cfg.min_standby - sum(x[t.id, S.STANDBY] for t in trains))
    costs += [W_SERVICE_SHORTFALL * service_short, W_STANDBY_SHORTFALL * standby_short]

    m.Minimize(sum(costs))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = cfg.solver_time_limit_s
    solver.parameters.num_workers = 1   # deterministic: the same fleet always gives the same plan
    solver.parameters.random_seed = 7
    status = solver.Solve(m)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"Planner could not find a feasible plan ({solver.StatusName(status)})")

    assign = {t.id: next(s for s in STATES if solver.Value(x[t.id, s])) for t in trains}
    moves = {t.id: sum(solver.Value(v) for (a, tt), v in shunts.items() if tt == t.id) for t in trains}
    solve_ms = int((time.perf_counter() - started) * 1000)

    decisions = [
        _explain(t, assign[t.id], health[t.id], fleet, today, cfg, service_value.get(t.id, {}), brand.get(t.id), moves[t.id], assign)
        for t in trains
    ]
    order = {s: i for i, s in enumerate(STATES)}
    decisions.sort(key=lambda d: (order[d.assignment], d.stabling_track, d.stabling_position))

    in_service = [d for d in decisions if d.assignment == S.SERVICE]
    fit = [by_id[i] for i in fit_ids]
    branded_in_service = [b for tid, b in brand.items() if assign[tid] == S.SERVICE]
    at_risk = sorted(
        b.contract.advertiser for tid, b in brand.items()
        if b.at_risk or (b.deficit_hours > 0 and assign[tid] != S.SERVICE)
    )
    kpis = PlanKPIs(
        service=len(in_service), service_target=cfg.service_target,
        standby=sum(d.assignment == S.STANDBY for d in decisions),
        inspection=sum(d.assignment == S.IBL for d in decisions),
        cleaning=sum(d.assignment == S.CLEANING for d in decisions),
        held=sum(d.assignment == S.HOLD for d in decisions),
        shunting_moves=sum(moves.values()),
        branding_trains_in_service=len(branded_in_service), branding_trains_total=len(brand),
        branding_at_risk=at_risk,
        avg_service_wear_km=int(sum(d.km_since_maintenance for d in in_service) / max(1, len(in_service))),
        avg_fleet_wear_km=int(sum(t.km_since_maintenance for t in fit) / max(1, len(fit))),
        solve_ms=solve_ms, solver_status=solver.StatusName(status),
    )
    return Plan(
        generated_at=datetime.now(), plan_date=today + timedelta(days=1),
        decisions=decisions, kpis=kpis, alerts=_alerts(kpis, decisions, health, cfg),
    )


def _explain(t: Train, a: Assignment, h, fleet: Fleet, today: date, cfg: PlannerConfig,
             value: dict[str, int], b, moves: int, assign: dict[int, Assignment]) -> TrainDecision:
    reasons: list[str] = []
    score = ScoreBreakdown()
    jobs = open_jobs(fleet, t.id)

    if a == S.SERVICE:
        reasons.append("All fitness certificates valid and no critical job cards")
        if b is not None:
            if b.deficit_hours > 0:
                reasons.append(f"Branding: {b.contract.advertiser} is {b.deficit_hours:.0f} h behind contract — prioritised for exposure")
            else:
                reasons.append(f"Branding: keeps {b.contract.advertiser} exposure on schedule")
        if value.get("mileage", 0) >= 0:
            reasons.append(f"Low wear: {_fmt_km(t.km_since_maintenance)} since last inspection")
        else:
            reasons.append(f"Needed to meet the timetable despite higher wear ({_fmt_km(t.km_since_maintenance)} since inspection)")
        reasons.append("Exit-side stabling, no shunting needed" if moves == 0 else f"Needs {moves} shunting move{'s' if moves > 1 else ''} to reach the exit")
        reasons += [f"Watch: {w}" for w in h.warnings]
        score = ScoreBreakdown(
            branding=value.get("branding", 0), mileage=value.get("mileage", 0),
            shunting=-W_SHUNT * moves, risk=value.get("risk", 0),
        )
    elif a == S.STANDBY:
        reasons.append("Fit for service, held as hot standby")
        if value.get("risk"):
            reasons.append("Certificate expiring soon — kept off the timetable to avoid a mid-day withdrawal")
        elif value.get("mileage", 0) < 0:
            reasons.append(f"Higher wear than inducted trains ({_fmt_km(t.km_since_maintenance)} since inspection)")
        if t.stabling_position > 1 and any(
            o.stabling_track == t.stabling_track and o.stabling_position < t.stabling_position and assign[o.id] != S.SERVICE
            for o in fleet.trains
        ):
            reasons.append("Parked behind a non-departing train — inducting it would cost extra shunting")
        reasons += [f"Watch: {w}" for w in h.warnings]
        score = ScoreBreakdown(mileage=value.get("mileage", 0), risk=value.get("risk", 0), branding=value.get("branding", 0))
    elif a == S.IBL:
        if h.readiness == Readiness.UNFIT:
            reasons.append("Sent to Inspection Bay Line to clear blockers")
            reasons += h.blockers
        else:
            reasons.append("Spare Inspection Bay capacity used for pending work")
            reasons += [f"Job card #{j.id} ({j.priority}): {j.description}" for j in jobs]
            if t.km_since_maintenance >= cfg.maint_warning_km:
                reasons.append(f"Inspection due soon ({_fmt_km(t.km_since_maintenance)} since last)")
        score = ScoreBreakdown(maintenance=sum(W_OPEN_JOB.get(j.priority, 0) for j in jobs))
    elif a == S.CLEANING:
        if t.cleaning_flagged:
            reasons.append("Flagged for deep cleaning by depot staff")
        else:
            days = (today - t.last_deep_clean).days
            reasons.append(f"Deep clean due — last cleaned {days} days ago (interval {cfg.cleaning_interval_days} days)")
        reasons += [f"Watch: {w}" for w in h.warnings if "clean" not in w.lower()]
        score = ScoreBreakdown(cleaning=W_CLEAN_DUE + W_CLEAN_PER_DAY * cleaning_overdue_days(t, today, cfg))
    else:  # HOLD
        reasons.append("Unfit for service and the Inspection Bay is full tonight")
        reasons += h.blockers

    return TrainDecision(
        train_id=t.id, train_name=t.name, assignment=a, readiness=h.readiness, reasons=reasons, score=score,
        stabling_track=t.stabling_track, stabling_position=t.stabling_position, shunting_moves=moves,
        km_since_maintenance=t.km_since_maintenance, total_km=t.total_km,
    )


def _alerts(k: PlanKPIs, decisions: list[TrainDecision], health, cfg: PlannerConfig) -> list[str]:
    alerts: list[str] = []
    if k.service < k.service_target:
        alerts.append(f"Service shortfall: only {k.service} of {k.service_target} trainsets available for the timetable")
    if k.standby < cfg.min_standby:
        alerts.append(f"Standby below minimum ({k.standby} of {cfg.min_standby})")
    for d in decisions:
        if d.assignment == S.HOLD:
            alerts.append(f"{d.train_name} held out of service — Inspection Bay full")
    for h in health.values():
        for w in h.warnings:
            if "expires today" in w or "expires tomorrow" in w:
                alerts.append(f"{h.train_name}: {w}")
    if k.branding_at_risk:
        alerts.append("Branding exposure at risk: " + ", ".join(k.branding_at_risk))
    return alerts


def diff_plans(before: Plan, after: Plan) -> list[PlanChange]:
    prev = {d.train_name: d.assignment for d in before.decisions}
    return [
        PlanChange(train_name=d.train_name, before=prev[d.train_name], after=d.assignment)
        for d in sorted(after.decisions, key=lambda d: d.train_name)
        if prev.get(d.train_name) not in (None, d.assignment)
    ]
