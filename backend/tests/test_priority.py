import pytest

from app.models import FleetUpdate
from app.priority import apply_policy, classify
from app.updates import validate_updates


@pytest.mark.parametrize("text, expected", [
    # The three cases Gemini and our own labels disagreed on
    ("TS-20 CCTV not recording in car 2", "High"),
    ("Door 4 on TS-09 opening slowly", "Routine"),
    ("TS-16 PIS screen blank in car 1", "High"),
    # Safety-critical
    ("wheel flat detected", "Critical"),
    ("brake pad on car 2 worn out", "Critical"),
    ("vibration noise from bogie", "Critical"),
    ("CBTC onboard unit fault", "Critical"),
    ("door obstruction detection not working", "Critical"),
    ("burning smell from under-frame", "Critical"),
    # High
    ("Coach 2 AC not working, passengers complaining", "High"),
    ("door 3 not closing", "High"),
    ("traction motor overheating", "High"),
    # Routine
    ("saloon light flickering", "Routine"),
    ("seat torn in car 4", "Routine"),
    ("window glass cracked", "Routine"),
    # Explicit overrides
    ("seat torn, urgent", "Critical"),
    ("HVAC tripping repeatedly, minor issue", "Routine"),
    ("raise routine JC for the PIS display blank", "Routine"),
])
def test_policy(text, expected):
    assert classify(text)[0] == expected


def test_safety_rules_cannot_be_downgraded():
    assert classify("minor brake noise")[0] == "Critical"


def test_unknown_defect_keeps_engine_proposal():
    assert apply_policy("strange smell of coffee", None, "High") == ("High", None)
    assert apply_policy("strange smell of coffee", None, None) == ("Routine", None)


def test_validate_applies_policy_and_explains(fleet, today):
    ups = [FleetUpdate(train_name="TS-20", action="open_job_card", description="CCTV not recording in car 2", priority="Routine")]
    valid, warnings = validate_updates(ups, fleet, today)
    assert valid[0].priority == "High"
    assert any("depot policy (CCTV)" in w for w in warnings)
