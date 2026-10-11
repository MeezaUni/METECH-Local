"""The contract every AI engine implements, plus the context it is given."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Protocol

from ..models import AIInfo, FleetUpdate, QueryIntent


@dataclass(frozen=True)
class AIContext:
    today: date
    train_names: list[str]
    open_jobs: list[str]          # "#1070 KMRL-TS-011 Critical: Wheel flat detected"
    job_owner: dict[int, str]     # job card id -> train name


class AIProvider(Protocol):
    def info(self) -> AIInfo: ...

    def parse_logbook(self, text: str, ctx: AIContext) -> list[FleetUpdate]:
        """Turn free-text logbook entries / messages into structured updates."""

    def interpret_question(self, question: str, ctx: AIContext) -> QueryIntent:
        """Turn a supervisor's question into a structured query."""

    def freeform_answer(self, question: str, facts: str) -> str | None:
        """Optional: answer a question the structured intents don't cover, grounded in `facts`."""


class AIUnavailable(RuntimeError):
    """Raised when the engine can't serve a request; callers fall back to the rule engine."""
