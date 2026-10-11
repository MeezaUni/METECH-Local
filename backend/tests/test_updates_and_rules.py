from datetime import timedelta

from app.ai import rules
from app.models import FleetUpdate
from app.updates import apply_updates, normalize_train_name, validate_updates


def test_normalize_train_name(fleet):
    assert normalize_train_name("TS-4", fleet) == "KMRL-TS-004"
    assert normalize_train_name("kmrl-ts-017", fleet) == "KMRL-TS-017"
    assert normalize_train_name("train 25", fleet) == "KMRL-TS-025"
    assert normalize_train_name("TS-99", fleet) is None


def test_validate_rejects_bad_updates(fleet, today):
    updates = [
        FleetUpdate(train_name="TS-99", action="mark_cleaned"),
        FleetUpdate(train_name="TS-1", action="renew_certificate", department="Telecom", date=(today - timedelta(days=1)).isoformat()),
        FleetUpdate(train_name="TS-1", action="close_job_card", job_card_id=999999),
        FleetUpdate(train_name="TS-1", action="open_job_card", description="  "),
        FleetUpdate(train_name="TS-2", action="mark_cleaned"),
    ]
    valid, warnings = validate_updates(updates, fleet, today)
    assert [u.train_name for u in valid] == ["KMRL-TS-002"]
    assert len(warnings) == 4


def test_job_card_number_decides_the_train(fleet, today):
    job = next(j for j in fleet.job_cards if j.status == "Open")
    valid, warnings = validate_updates([FleetUpdate(train_name="TS-1", action="close_job_card", job_card_id=job.id)], fleet, today)
    assert valid[0].train_name == f"KMRL-TS-{job.train_id:03d}"
    assert warnings and "corrected" in warnings[0]


def test_apply_does_not_mutate_input(fleet, today):
    before = fleet.model_dump()
    new, summaries = apply_updates(fleet, [FleetUpdate(train_name="KMRL-TS-003", action="mark_cleaned")], today)
    assert fleet.model_dump() == before
    assert next(t for t in new.trains if t.name == "KMRL-TS-003").last_deep_clean == today
    assert summaries


def test_rule_parser_handles_mixed_entry(fleet, today):
    owner = {j.id: f"KMRL-TS-{j.train_id:03d}" for j in fleet.job_cards}
    text = ("Telecom cert for TS-15 renewed till 5 March 2027. Closed job card #1070. "
            "Train 7 inspection completed, train 21 deep cleaned. TS 9 brake pad wear reported, urgent.")
    ups = rules.parse_logbook(text, today, owner)
    got = {(u.train_name, u.action) for u in ups}
    assert ("KMRL-TS-015", "renew_certificate") in got
    assert ("KMRL-TS-011", "close_job_card") in got
    assert ("KMRL-TS-007", "complete_maintenance") in got
    assert ("KMRL-TS-021", "mark_cleaned") in got
    assert ("KMRL-TS-009", "open_job_card") in got
    cert = next(u for u in ups if u.action == "renew_certificate")
    assert cert.department == "Telecom" and cert.date == "2027-03-05"
    job = next(u for u in ups if u.action == "open_job_card")
    assert job.priority == "Critical"


def test_rule_parser_dates(today):
    assert rules.parse_date("valid till 2027-01-31", today).isoformat() == "2027-01-31"
    assert rules.parse_date("upto 15/02/2027", today).isoformat() == "2027-02-15"
    assert rules.parse_date("good until Jan 5", today).isoformat() == "2027-01-05"
    assert rules.parse_date("extended for 6 months", today).isoformat() == "2027-04-11"


def test_rule_intents():
    assert rules.interpret_question("how many trains are red?").model_dump(exclude_none=True) == {"intent": "count", "status": "Unfit"}
    assert rules.interpret_question("why is TS-9 on standby").intent == "explain_assignment"
    assert rules.interpret_question("tell me about train 14").train_name == "KMRL-TS-014"
    assert rules.interpret_question("which certificates expire in 10 days").days == 10
    assert rules.interpret_question("show open critical jobs").priority == "Critical"
