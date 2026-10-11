"""The training data and the on-device prompt format must agree with the API schemas."""
import json
import random
import sys
from datetime import date
from pathlib import Path

from app.ai.local_format import dump_updates, ground_and_resolve, load_json
from app.models import FleetUpdate, QueryIntent

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ml"))
import generate_dataset as gen  # noqa: E402

TODAY = date(2026, 10, 11)


def test_generated_labels_are_valid_and_grounded():
    rng = random.Random(0)
    for _ in range(600):
        row = gen.parse_example(TODAY, rng)
        text = row["messages"][1]["content"].split("\nLogbook:\n", 1)[1]
        updates = json.loads(row["target"])["updates"]
        for u in updates:
            assert u["source_text"] in text, (u["source_text"], text)
            if u["action"] == "renew_certificate":
                assert "date" not in u and u["date_text"] in u["source_text"], u   # an exact, copyable span
        kept, dropped = ground_and_resolve(updates, text, TODAY)
        assert not dropped, (dropped, text)                                       # every label passes grounding
        for u in kept:
            FleetUpdate.model_validate({**u, "train_name": u.get("train_name", "")})
            if u["action"] == "renew_certificate":
                assert date.fromisoformat(u["date"]) > TODAY


def test_grounding_blocks_invented_updates():
    text = "B check of TS-24 over. TS-24 good to go"
    proposed = [
        {"train_name": "KMRL-TS-024", "action": "complete_maintenance"},
        {"train_name": "KMRL-TS-024", "action": "renew_certificate", "department": "Telecom", "date_text": "for 6 months"},  # round-1 invention
        {"train_name": "KMRL-TS-031", "action": "mark_cleaned"},                     # train not in the text
        {"train_name": "", "action": "close_job_card", "job_card_id": 1099},          # job card not in the text
    ]
    kept, dropped = ground_and_resolve(proposed, text, TODAY)
    assert [u["action"] for u in kept] == ["complete_maintenance"]
    assert len(dropped) == 3 and any("for 6 months" in d for d in dropped)


def test_expiry_phrase_is_resolved_by_code():
    text = "TS-10 Telecom and Signalling certificates renewed for 6 months"
    kept, _ = ground_and_resolve(
        [{"train_name": "KMRL-TS-010", "action": "renew_certificate", "department": d, "date_text": "for 6 months"} for d in ("Telecom", "Signalling")],
        text, TODAY,
    )
    assert [u["date"] for u in kept] == ["2027-04-11", "2027-04-11"]   # round 1 got 2027-03-11


def test_round_one_models_still_work():
    """Models trained before date_text emit a computed date; keep accepting it."""
    kept, dropped = ground_and_resolve([{"train_name": "KMRL-TS-005", "action": "renew_certificate", "department": "Telecom", "date": "2027-10-10"}],
                                       "TS 05 telecom FC valid till 10/10/2027", TODAY)
    assert kept[0]["date"] == "2027-10-10" and not dropped


def test_old_teacher_rows_are_converted_and_verified():
    msg = [{"role": "system", "content": "x"}, {"role": "user", "content": "Today: 2026-10-11\nLogbook:\nTS-22 signalling FC extended for a fortnight"}]
    def row(d):
        return {"base_id": "m1", "task": "parse", "messages": msg, "target": json.dumps({"updates": [
            {"train_name": "KMRL-TS-022", "action": "renew_certificate", "department": "Signalling", "date": d,
             "source_text": "TS-22 signalling FC extended for a fortnight"}]})}
    good = gen.normalise_teacher_row(row("2026-10-25"))
    assert json.loads(good["target"])["updates"][0]["date_text"] == "for a fortnight"
    assert gen.normalise_teacher_row(row("2026-11-07")) is None   # label and text disagree: dropped


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
