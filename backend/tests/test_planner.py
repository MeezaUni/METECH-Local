from datetime import timedelta

from app.config import PlannerConfig
from app.fleet import assess_fleet
from app.models import Assignment, FleetUpdate, Readiness
from app.planner import diff_plans, generate_plan
from app.updates import apply_updates


def by_name(plan):
    return {d.train_name: d for d in plan.decisions}


def test_every_train_gets_exactly_one_assignment(fleet, cfg, today):
    plan = generate_plan(fleet, cfg, today)
    assert sorted(d.train_name for d in plan.decisions) == sorted(t.name for t in fleet.trains)
    assert plan.kpis.solver_status == "OPTIMAL"


def test_unfit_trains_never_enter_service_or_standby(fleet, cfg, today):
    plan = generate_plan(fleet, cfg, today)
    health = assess_fleet(fleet, today, cfg)
    for d in plan.decisions:
        if health[d.train_id].readiness == Readiness.UNFIT:
            assert d.assignment in (Assignment.IBL, Assignment.HOLD), d


def test_single_expired_certificate_grounds_train(fleet, cfg, today):
    """Regression for v1.x: the old backend only grounded a train when *all* certificates had expired."""
    t = fleet.trains[0]
    certs = [c for c in fleet.certificates if c.train_id == t.id]
    certs[0].expires = today - timedelta(days=1)
    assert all(c.expires > today for c in certs[1:])
    plan = generate_plan(fleet, cfg, today)
    assert by_name(plan)[t.name].assignment not in (Assignment.SERVICE, Assignment.STANDBY)


def test_capacities_and_targets_respected(fleet, cfg, today):
    plan = generate_plan(fleet, cfg, today)
    k = plan.kpis
    assert k.service == cfg.service_target
    assert k.standby >= cfg.min_standby
    assert k.inspection <= cfg.ibl_slots
    assert k.cleaning <= cfg.cleaning_slots


def test_shortfall_is_reported_not_crashed(fleet, today):
    cfg = PlannerConfig(service_target=24)
    plan = generate_plan(fleet, cfg, today)
    assert plan.kpis.service < 24
    assert any("shortfall" in a.lower() for a in plan.alerts)


def test_branded_train_behind_on_hours_is_inducted(fleet, cfg, today):
    plan = generate_plan(fleet, cfg, today)
    decisions = by_name(plan)
    names = {t.id: t.name for t in fleet.trains}
    for b in fleet.branding:
        assert decisions[names[b.train_id]].assignment == Assignment.SERVICE


def test_plan_is_deterministic(fleet, cfg, today):
    a = generate_plan(fleet, cfg, today)
    b = generate_plan(fleet, cfg, today)
    assert [(d.train_name, d.assignment) for d in a.decisions] == [(d.train_name, d.assignment) for d in b.decisions]


def test_what_if_closing_critical_job_frees_train(fleet, cfg, today):
    baseline = generate_plan(fleet, cfg, today)
    critical = next(j for j in fleet.job_cards if j.status == "Open" and j.priority == "Critical")
    train = next(t.name for t in fleet.trains if t.id == critical.train_id)
    what_if, _ = apply_updates(fleet, [FleetUpdate(train_name=train, action="close_job_card", job_card_id=critical.id)], today)
    simulated = generate_plan(what_if, cfg, today)
    assert by_name(simulated)[train].readiness != Readiness.UNFIT or by_name(baseline)[train].readiness == Readiness.UNFIT
    # Freeing an IBL slot lets the held train in, so the plan must change.
    assert diff_plans(baseline, simulated)


def test_every_decision_is_explained(fleet, cfg, today):
    plan = generate_plan(fleet, cfg, today)
    assert all(d.reasons for d in plan.decisions)
