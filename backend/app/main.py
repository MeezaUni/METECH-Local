"""KMRL Sentient Operations Hub — HTTP API."""
from __future__ import annotations

import logging
from dataclasses import asdict
from datetime import date
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import assistant
from .ai import AIContext, AIUnavailable, build_provider
from .ai import rules
from .config import settings
from .db import Database
from .fleet import assess_fleet, branding_status
from .models import (
    AIInfo, ApplyRequest, ApplyResult, AskRequest, AskResponse, AuditEntry, BrandingView, Fleet, FleetView,
    ParseRequest, ParseResponse, Plan, SimulateRequest, SimulationResult, TrainView,
)
from .planner import diff_plans, generate_plan
from .updates import apply_updates, validate_updates

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

app = FastAPI(title="KMRL Sentient Operations Hub", version="2.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=list(settings.cors_origins), allow_methods=["*"], allow_headers=["*"],
)

db = Database(settings.db_path)
ai = build_provider(settings)
cfg = settings.planner


def today() -> date:
    return date.today()


def _context(fleet: Fleet) -> AIContext:
    names = {t.id: t.name for t in fleet.trains}
    open_jobs = [j for j in fleet.job_cards if j.status == "Open"]
    return AIContext(
        today=today(),
        train_names=[t.name for t in fleet.trains],
        open_jobs=[f"#{j.id} {names[j.train_id]} {j.priority}: {j.description}" for j in open_jobs],
        job_owner={j.id: names[j.train_id] for j in fleet.job_cards},
    )


def _engine_label() -> str:
    info = ai.info()
    return f"{info.engine} ({info.model})"


# ---------------------------------------------------------------- fleet
@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "edition": ai.info().edition}


@app.get("/api/fleet", response_model=FleetView)
def get_fleet() -> FleetView:
    fleet = db.load_fleet()
    t0 = today()
    health = assess_fleet(fleet, t0, cfg)
    views = []
    for t in fleet.trains:
        h = health[t.id]
        contract = next((b for b in fleet.branding if b.train_id == t.id), None)
        bview = None
        if contract:
            s = branding_status(contract, t0)
            bview = BrandingView(**contract.model_dump(), expected_hours=round(s.expected_hours, 1),
                                 deficit_hours=round(s.deficit_hours, 1), days_left=s.days_left, at_risk=s.at_risk)
        views.append(TrainView(
            **t.model_dump(), readiness=h.readiness, blockers=h.blockers, warnings=h.warnings,
            certificates=[c for c in fleet.certificates if c.train_id == t.id],
            job_cards=sorted((j for j in fleet.job_cards if j.train_id == t.id), key=lambda j: (j.status != "Open", -j.id)),
            branding=bview,
        ))
    return FleetView(today=t0, trains=views, planner=asdict(cfg))


@app.post("/api/fleet/updates", response_model=ApplyResult)
def post_updates(req: ApplyRequest) -> ApplyResult:
    fleet = db.load_fleet()
    valid, warnings = validate_updates(req.updates, fleet, today())
    if not valid:
        return ApplyResult(applied=[], rejected=warnings or ["No updates to apply"])
    new_fleet, summaries = apply_updates(fleet, valid, today())
    db.save_fleet(new_fleet)
    for u, line in zip(valid, summaries):
        detail = line + (f' — source: "{u.source_text}"' if u.source_text else "")
        db.audit(req.actor, req.via, u.action, detail)
    return ApplyResult(applied=summaries, rejected=warnings)


# ----------------------------------------------------------------- plan
@app.get("/api/plan", response_model=Plan)
def get_plan() -> Plan:
    return generate_plan(db.load_fleet(), cfg, today())


@app.post("/api/plan/simulate", response_model=SimulationResult)
def simulate(req: SimulateRequest) -> SimulationResult:
    fleet = db.load_fleet()
    baseline = generate_plan(fleet, cfg, today())
    valid, warnings = validate_updates(req.updates, fleet, today())
    what_if, applied = apply_updates(fleet, valid, today())
    simulated = generate_plan(what_if, cfg, today())
    return SimulationResult(baseline=baseline, simulated=simulated, changes=diff_plans(baseline, simulated),
                            applied=applied, warnings=warnings)


class ApproveRequest(BaseModel):
    approved_by: str = "Depot supervisor"


@app.post("/api/plan/approve")
def approve(req: ApproveRequest) -> dict:
    plan = generate_plan(db.load_fleet(), cfg, today())
    plan_id = db.approve_plan(plan, req.approved_by)
    k = plan.kpis
    db.audit(req.approved_by, "manual", "approve_plan",
             f"Approved induction plan #{plan_id} for {plan.plan_date}: {k.service} service, {k.standby} standby, "
             f"{k.inspection} IBL, {k.cleaning} cleaning, {k.held} held")
    return {"id": plan_id, "plan_date": plan.plan_date}


@app.get("/api/plan/history")
def plan_history() -> list[dict]:
    return db.approved_plans()


# ------------------------------------------------------------------- AI
@app.get("/api/ai/info", response_model=AIInfo)
def ai_info() -> AIInfo:
    return ai.info()


@app.post("/api/ai/parse", response_model=ParseResponse)
def ai_parse(req: ParseRequest) -> ParseResponse:
    fleet = db.load_fleet()
    ctx = _context(fleet)
    notes: list[str] = []
    try:
        raw = ai.parse_logbook(req.text, ctx)
        engine = _engine_label()
        notes += getattr(ai, "take_warnings", lambda: [])()   # e.g. suggestions dropped by a grounding check
    except AIUnavailable as exc:
        raw = rules.parse_logbook(req.text, ctx.today, ctx.job_owner)
        engine = "Rule engine"
        notes.append(f"AI unavailable, used rule engine: {exc}")
    valid, warnings = validate_updates(raw, fleet, ctx.today)
    return ParseResponse(updates=valid, warnings=notes + warnings, engine=engine)


@app.post("/api/ai/ask", response_model=AskResponse)
def ai_ask(req: AskRequest) -> AskResponse:
    fleet = db.load_fleet()
    ctx = _context(fleet)
    try:
        intent = ai.interpret_question(req.question, ctx)
        engine = _engine_label()
    except AIUnavailable:
        intent = rules.interpret_question(req.question)
        engine = "Rule engine"
    plan = generate_plan(fleet, cfg, ctx.today)
    health = assess_fleet(fleet, ctx.today, cfg)
    text = assistant.answer(intent, fleet, plan, health, ctx.today)
    if intent.intent == "help":
        facts = assistant.answer(intent.model_copy(update={"intent": "plan_summary"}), fleet, plan, health, ctx.today)
        free = ai.freeform_answer(req.question, facts)
        if free:
            text = free
    return AskResponse(answer=text, intent=intent.model_dump(exclude_none=True), engine=engine)


# ---------------------------------------------------------------- admin
@app.get("/api/audit", response_model=list[AuditEntry])
def audit(limit: int = 100) -> list[AuditEntry]:
    return db.audit_entries(min(max(limit, 1), 500))


@app.post("/api/admin/reset")
def reset() -> dict:
    db.reset(today())
    return {"status": "reset"}


@app.exception_handler(RuntimeError)
def runtime_error(_, exc: RuntimeError) -> JSONResponse:
    return JSONResponse(status_code=500, content={"detail": str(exc)})


# Serve the built frontend (npm run build) from the same origin when present.
_dist = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if _dist.is_dir():
    app.mount("/", StaticFiles(directory=_dist, html=True), name="frontend")
