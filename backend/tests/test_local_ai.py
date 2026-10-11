"""The training data and the on-device prompt format must agree with the API schemas."""
import json
import random
import sys
from datetime import date
from pathlib import Path

from app.ai.local_format import dump_updates, load_json
from app.models import FleetUpdate, QueryIntent

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ml"))
import generate_dataset as gen  # noqa: E402


def test_generated_labels_are_valid_and_grounded():
    rng = random.Random(0)
    for _ in range(400):
        row = gen.parse_example(date(2026, 10, 11), rng)
        text = row["messages"][1]["content"].split("\nLogbook:\n", 1)[1]
        for u in json.loads(row["target"])["updates"]:
            FleetUpdate.model_validate({**u, "train_name": u.get("train_name", "")})
            assert u["source_text"] in text, (u["source_text"], text)
            if u["action"] == "renew_certificate":
                assert date.fromisoformat(u["date"]) > date(2026, 10, 11)


def test_generated_intents_are_valid():
    rng = random.Random(0)
    for _ in range(200):
        QueryIntent.model_validate_json(gen.ask_example(rng)["target"])


def test_load_json_tolerates_stray_text():
    assert load_json('sure! {"updates": []} ') == {"updates": []}
    assert json.loads(dump_updates([{"train_name": "KMRL-TS-001", "action": "mark_cleaned", "km": None}])) == {
        "updates": [{"train_name": "KMRL-TS-001", "action": "mark_cleaned"}]
    }


def test_missing_model_reports_not_ready(tmp_path):
    from app.ai.local_model import LocalModelProvider

    info = LocalModelProvider(tmp_path / "nope").info()
    assert info.edition == "local" and not info.ready and "train.py" in info.note


def test_generated_priorities_follow_depot_policy():
    from app.priority import apply_policy

    rng = random.Random(1)
    checked = 0
    for _ in range(1500):
        row = gen.parse_example(date(2026, 10, 11), rng)
        for u in json.loads(row["target"])["updates"]:
            if u["action"] == "open_job_card":
                assert apply_policy(u["description"], u["source_text"], u["priority"])[0] == u["priority"], u
                checked += 1
    assert checked > 100


def test_real_world_test_set_never_leaks_into_training_data():
    """The hand-written test set must stay unseen, or its scores mean nothing."""
    test_set = Path(__file__).resolve().parents[2] / "ml" / "data" / "realworld_test.jsonl"
    held_out = {(json.loads(l).get("text") or json.loads(l).get("question")).lower() for l in open(test_set, encoding="utf-8")}
    rng = random.Random(2)
    for _ in range(3000):
        row = gen.parse_example(date(2026, 10, 11), rng) if rng.random() < 0.75 else gen.ask_example(rng)
        content = row["messages"][1]["content"].split("\nLogbook:\n")[-1].lower()
        assert not any(h in content for h in held_out if len(h) > 12), content


def test_teacher_rewrites_are_rejected_when_facts_change():
    import paraphrase as para

    original = "Telecom cert for TS-05 renewed till 10/10/2027"
    protect = ["TS-05", "Telecom", "10/10/2027"]
    assert para.acceptable("TS-05 telecom FC came thru, valid till 10/10/2027", original, protect)
    assert not para.acceptable("TS-06 telecom FC valid till 10/10/2027", original, protect)        # train changed
    assert not para.acceptable("TS-05 telecom FC valid till Oct 2027", original, protect)          # date dropped
    assert not para.acceptable("TS-05 telecom FC valid till 10/10/2027, JC 1099 closed", original, protect)  # invented fact
    assert not para.acceptable(original, original, protect)                                         # not a rewrite


def test_teacher_question_protects_status_and_train():
    import paraphrase as para

    got = para.question_protect("Which rakes are on standby?", {"intent": "list", "status": "Standby"})
    assert "on standby" in got
    assert "TS-09" in para.question_protect("why is TS-09 on standby", {"intent": "explain_assignment", "train_name": "KMRL-TS-009"})
