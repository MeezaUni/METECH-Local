"""Per-train readiness: which trains may legally enter service, and what to watch."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from .config import PlannerConfig
from .models import DEPARTMENTS, BrandingContract, Fleet, JobCard, Readiness, Train, TrainHealth

HOURS_IN_SERVICE_PER_DAY = 16


@dataclass(frozen=True)
class BrandingStatus:
    contract: BrandingContract
    expected_hours: float     # where delivery should be today, pro rata
    deficit_hours: float      # expected - delivered (positive = behind)
    days_left: int
    at_risk: bool             # cannot reach target even running every remaining day

    @property
    def deficit_ratio(self) -> float:
        return max(0.0, self.deficit_hours) / self.contract.target_hours


def branding_status(contract: BrandingContract, today: date) -> BrandingStatus:
    period = max(1, (contract.period_end - contract.period_start).days)
    elapsed = min(period, max(0, (today - contract.period_start).days))
    expected = contract.target_hours * elapsed / period
    days_left = max(0, (contract.period_end - today).days)
    remaining = contract.target_hours - contract.delivered_hours
    at_risk = remaining > days_left * HOURS_IN_SERVICE_PER_DAY * 0.9
    return BrandingStatus(contract, expected, expected - contract.delivered_hours, days_left, at_risk)


def cleaning_overdue_days(train: Train, today: date, cfg: PlannerConfig) -> int:
    """Days past the deep-cleaning interval (0 if not due)."""
    return max(0, (today - train.last_deep_clean).days - cfg.cleaning_interval_days)


def needs_cleaning(train: Train, today: date, cfg: PlannerConfig) -> bool:
    return train.cleaning_flagged or (today - train.last_deep_clean).days >= cfg.cleaning_interval_days


def open_jobs(fleet: Fleet, train_id: int) -> list[JobCard]:
    return [j for j in fleet.job_cards if j.train_id == train_id and j.status == "Open"]


def assess_train(fleet: Fleet, train: Train, today: date, cfg: PlannerConfig) -> TrainHealth:
    blockers: list[str] = []
    warnings: list[str] = []

    certs = {c.department: c for c in fleet.certificates if c.train_id == train.id}
    for dept in DEPARTMENTS:
        cert = certs.get(dept)
        if cert is None:
            blockers.append(f"No {dept} fitness certificate on record")
            continue
        days_left = (cert.expires - today).days
        if days_left < 0:
            blockers.append(f"{dept} fitness certificate expired {cert.expires.isoformat()}")
        elif days_left <= cfg.cert_warning_days:
            when = "today" if days_left == 0 else ("tomorrow" if days_left == 1 else f"in {days_left} days")
            warnings.append(f"{dept} fitness certificate expires {when}")

    for job in open_jobs(fleet, train.id):
        label = f"Job card #{job.id} ({job.priority}): {job.description}"
        (blockers if job.priority == "Critical" else warnings).append(label)

    if train.km_since_maintenance >= cfg.maint_limit_km:
        blockers.append(f"Inspection overdue: {train.km_since_maintenance:,} km since last inspection (limit {cfg.maint_limit_km:,})")
    elif train.km_since_maintenance >= cfg.maint_warning_km:
        warnings.append(f"Inspection due soon: {train.km_since_maintenance:,} km since last inspection")

    if train.cleaning_flagged:
        warnings.append("Flagged for deep cleaning")
    elif needs_cleaning(train, today, cfg):
        warnings.append(f"Deep clean due (last cleaned {(today - train.last_deep_clean).days} days ago)")

    readiness = Readiness.UNFIT if blockers else (Readiness.WARNING if warnings else Readiness.FIT)
    return TrainHealth(train_id=train.id, train_name=train.name, readiness=readiness, blockers=blockers, warnings=warnings)


def assess_fleet(fleet: Fleet, today: date, cfg: PlannerConfig) -> dict[int, TrainHealth]:
    return {t.id: assess_train(fleet, t, today, cfg) for t in fleet.trains}
