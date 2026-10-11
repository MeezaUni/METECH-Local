"""Runtime configuration, read once from the environment (and backend/.env)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BACKEND_DIR / ".env")


@dataclass(frozen=True)
class PlannerConfig:
    """Depot-level constraints for one night's induction plan."""

    service_target: int = 17          # trainsets the timetable needs at 05:30
    min_standby: int = 2              # hot spares kept ready at the depot
    ibl_slots: int = 3                # Inspection Bay Line capacity tonight
    cleaning_slots: int = 3           # deep-cleaning bay capacity tonight
    cert_warning_days: int = 2        # certificates expiring this soon are flagged
    cleaning_interval_days: int = 10  # deep clean due after this many days
    maint_warning_km: int = 4500      # km since last inspection -> flagged
    maint_limit_km: int = 6000        # km since last inspection -> must be inspected
    solver_time_limit_s: float = 5.0


@dataclass(frozen=True)
class Settings:
    db_path: Path = field(default_factory=lambda: Path(os.getenv("KMRL_DB_PATH", BACKEND_DIR / "data" / "kmrl.db")))
    cors_origins: tuple[str, ...] = tuple(
        o.strip() for o in os.getenv("KMRL_CORS_ORIGINS", "http://localhost:5174,http://127.0.0.1:5174").split(",") if o.strip()
    )
    local_engine: str = os.getenv("KMRL_LOCAL_ENGINE", "trained").lower()
    model_dir: Path = field(default_factory=lambda: Path(os.getenv("KMRL_MODEL_DIR", BACKEND_DIR / "models" / "kmrl-copilot")))
    model_device: str = os.getenv("KMRL_MODEL_DEVICE", "auto")   # auto | cuda | cpu
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "qwen3:14b")
    planner: PlannerConfig = field(default_factory=PlannerConfig)


settings = Settings()
