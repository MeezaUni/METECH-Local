import os
import sys
import tempfile
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# Isolated database and no real API calls during tests.
os.environ["KMRL_DB_PATH"] = str(Path(tempfile.mkdtemp()) / "test.db")
os.environ["KMRL_MODEL_DIR"] = str(Path(tempfile.mkdtemp()) / "no-model")  # tests use the rule fallback

from app.config import PlannerConfig  # noqa: E402
from app.seed import generate_fleet  # noqa: E402

TODAY = date(2026, 10, 11)


@pytest.fixture
def today() -> date:
    return TODAY


@pytest.fixture
def fleet():
    return generate_fleet(TODAY)


@pytest.fixture
def cfg() -> PlannerConfig:
    return PlannerConfig()
