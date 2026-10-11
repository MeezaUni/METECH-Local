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
