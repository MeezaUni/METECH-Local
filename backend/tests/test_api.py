from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def setup_function():
    client.post("/api/admin/reset")


def test_fleet_and_plan():
    fleet = client.get("/api/fleet").json()
    assert len(fleet["trains"]) == 25
    plan = client.get("/api/plan").json()
    assert plan["kpis"]["service"] == fleet["planner"]["service_target"]


def test_ai_falls_back_to_rules_without_key():
    info = client.get("/api/ai/info").json()
    assert info["ready"] is False
    r = client.post("/api/ai/parse", json={"text": "Train 3 deep cleaned. TS-12 needs a deep clean, graffiti on car 2."}).json()
    assert r["engine"] == "Rule engine"
    assert {(u["train_name"], u["action"]) for u in r["updates"]} == {("KMRL-TS-003", "mark_cleaned"), ("KMRL-TS-012", "flag_cleaning")}


def test_apply_updates_persists_and_audits():
    r = client.post("/api/fleet/updates", json={"updates": [{"train_name": "TS-3", "action": "mark_cleaned"}], "via": "ai"}).json()
    assert r["applied"] and not r["rejected"]
    audit = client.get("/api/audit").json()
    assert audit[0]["action"] == "mark_cleaned" and audit[0]["via"] == "ai"


def test_simulate_does_not_persist():
    before = client.get("/api/plan").json()
    sim = client.post("/api/plan/simulate", json={"updates": [{"train_name": "TS-11", "action": "close_job_card", "job_card_id": 1070}]}).json()
    assert sim["changes"]
    after = client.get("/api/plan").json()
    assert [d["assignment"] for d in before["decisions"]] == [d["assignment"] for d in after["decisions"]]


def test_ask_answers_from_data():
    r = client.post("/api/ai/ask", json={"question": "How many trains are unfit?"}).json()
    assert r["intent"]["intent"] == "count"
    assert "of 25 trainsets" in r["answer"]


def test_approve_plan_recorded():
    r = client.post("/api/plan/approve", json={"approved_by": "Test"}).json()
    assert r["id"] >= 1
    assert client.get("/api/plan/history").json()[0]["approved_by"] == "Test"
