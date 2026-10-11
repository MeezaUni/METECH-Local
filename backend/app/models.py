"""Pydantic schemas shared by the API, the planner and the AI providers."""
from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

Department = Literal["Rolling-Stock", "Signalling", "Telecom"]
Priority = Literal["Critical", "High", "Routine"]
DEPARTMENTS: tuple[Department, ...] = ("Rolling-Stock", "Signalling", "Telecom")


# ---------------------------------------------------------------- fleet data
class Train(BaseModel):
    id: int
    name: str
    commissioned: date
    total_km: int
    km_since_maintenance: int
    last_deep_clean: date
    cleaning_flagged: bool = False
    stabling_track: str
    stabling_position: int


class Certificate(BaseModel):
    id: int
    train_id: int
    department: Department
    issued: date
    expires: date


class JobCard(BaseModel):
    id: int
    train_id: int
    system: str
    description: str
    priority: Priority
    status: Literal["Open", "Closed"]
    opened: date
    closed: date | None = None


class BrandingContract(BaseModel):
    id: int
    train_id: int
    advertiser: str
    target_hours: int      # contracted exposure hours for the period
    delivered_hours: float
    period_start: date
    period_end: date


class BrandingView(BrandingContract):
    expected_hours: float
    deficit_hours: float
    days_left: int
    at_risk: bool


class Fleet(BaseModel):
    trains: list[Train]
    certificates: list[Certificate]
    job_cards: list[JobCard]
    branding: list[BrandingContract]


# ----------------------------------------------------------------- readiness
class Readiness(str, Enum):
    FIT = "Fit"
    WARNING = "Warning"
    UNFIT = "Unfit"


class TrainHealth(BaseModel):
    train_id: int
    train_name: str
    readiness: Readiness
    blockers: list[str] = []   # anything here keeps the train out of service
    warnings: list[str] = []


# -------------------------------------------------------------------- planning
class Assignment(str, Enum):
    SERVICE = "Service"
    STANDBY = "Standby"
    IBL = "Inspection Bay"
    CLEANING = "Cleaning"
    HOLD = "Held"


class ScoreBreakdown(BaseModel):
    branding: float = 0
    mileage: float = 0
    shunting: float = 0
    maintenance: float = 0
    cleaning: float = 0
    risk: float = 0


class TrainDecision(BaseModel):
    train_id: int
    train_name: str
    assignment: Assignment
    readiness: Readiness
    reasons: list[str]
    score: ScoreBreakdown
    stabling_track: str
    stabling_position: int
    shunting_moves: int = 0
    km_since_maintenance: int
    total_km: int


class PlanKPIs(BaseModel):
    service: int
    service_target: int
    standby: int
    inspection: int
    cleaning: int
    held: int
    shunting_moves: int
    branding_trains_in_service: int
    branding_trains_total: int
    branding_at_risk: list[str]
    avg_service_wear_km: int   # mean km-since-inspection of inducted trains
    avg_fleet_wear_km: int     # same, across all fit trains
    solve_ms: int
    solver_status: str


class Plan(BaseModel):
    generated_at: datetime
    plan_date: date
    decisions: list[TrainDecision]
    kpis: PlanKPIs
    alerts: list[str]


class PlanChange(BaseModel):
    train_name: str
    before: Assignment
    after: Assignment


class SimulationResult(BaseModel):
    baseline: Plan
    simulated: Plan
    changes: list[PlanChange]
    applied: list[str] = []
    warnings: list[str] = []


# ------------------------------------------------------- updates (manual + AI)
UpdateAction = Literal[
    "renew_certificate",
    "close_job_card",
    "open_job_card",
    "flag_cleaning",
    "mark_cleaned",
    "complete_maintenance",
    "update_mileage",
]


class FleetUpdate(BaseModel):
    """One atomic change to the fleet. Produced by the UI, the AI parser or a simulation."""

    train_name: str = Field(description="Official train id, e.g. KMRL-TS-004")
    action: UpdateAction
    department: Department | None = Field(default=None, description="For renew_certificate")
    date: str | None = Field(default=None, description="ISO date YYYY-MM-DD, for renew_certificate")
    job_card_id: int | None = Field(default=None, description="For close_job_card")
    description: str | None = Field(default=None, description="For open_job_card")
    priority: Priority | None = Field(default=None, description="For open_job_card")
    km: int | None = Field(default=None, description="For update_mileage: km run since last inspection")
    source_text: str | None = Field(default=None, description="Exact snippet of the input that justifies this update")


class ParsedUpdates(BaseModel):
    updates: list[FleetUpdate]


class ApplyRequest(BaseModel):
    updates: list[FleetUpdate]
    actor: str = "supervisor"
    via: Literal["manual", "ai"] = "manual"


class ApplyResult(BaseModel):
    applied: list[str]
    rejected: list[str]


class SimulateRequest(BaseModel):
    updates: list[FleetUpdate] = []


class ParseRequest(BaseModel):
    text: str = Field(min_length=1, max_length=8000)


class ParseResponse(BaseModel):
    updates: list[FleetUpdate]
    warnings: list[str] = []
    engine: str


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


QueryKind = Literal[
    "count", "list", "train_detail", "explain_assignment", "expiring_certificates",
    "open_jobs", "branding_status", "plan_summary", "highest_wear", "help",
]


class QueryIntent(BaseModel):
    """Structured form of a supervisor's question. The answer is computed by code, never invented."""

    intent: QueryKind
    status: str | None = Field(default=None, description="Fit/Warning/Unfit or Service/Standby/Inspection Bay/Cleaning/Held")
    train_name: str | None = Field(default=None, description="Official train id, e.g. KMRL-TS-018")
    days: int | None = Field(default=None, description="Look-ahead window in days")
    priority: Priority | None = None
    limit: int | None = None


class AskResponse(BaseModel):
    answer: str
    intent: dict
    engine: str


class AuditEntry(BaseModel):
    id: int
    ts: datetime
    actor: str
    via: str
    action: str
    detail: str


class AIInfo(BaseModel):
    edition: Literal["cloud", "local"]
    engine: str
    model: str
    ready: bool
    note: str = ""


class TrainView(Train):
    readiness: Readiness
    blockers: list[str]
    warnings: list[str]
    certificates: list[Certificate]
    job_cards: list[JobCard]
    branding: BrandingView | None = None


class FleetView(BaseModel):
    today: date
    trains: list[TrainView]
    planner: dict
