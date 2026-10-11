"""Optional engine: a general-purpose LLM served by a local Ollama install.

No training needed, but needs a much bigger model (several GB) than the
fine-tuned copilot. Select with KMRL_LOCAL_ENGINE=ollama and OLLAMA_MODEL.
"""
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request

from ..models import AIInfo, FleetUpdate, ParsedUpdates, QueryIntent
from .base import AIContext, AIUnavailable
from .prompts import PARSER_INSTRUCTIONS, QUESTION_INSTRUCTIONS

log = logging.getLogger(__name__)


class OllamaProvider:
    def __init__(self, host: str, model: str):
        self.host = host.rstrip("/")
        self.model = model

    def _post(self, path: str, body: dict, timeout: float = 120) -> dict:
        req = urllib.request.Request(
            f"{self.host}{path}", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise AIUnavailable(f"Ollama at {self.host} unavailable: {exc}") from exc

    def info(self) -> AIInfo:
        try:
            tags = self._post("/api/show", {"model": self.model}, timeout=3)
            ready = "modelfile" in tags or "details" in tags
            return AIInfo(edition="local", engine="Ollama", model=self.model, ready=ready)
        except AIUnavailable as exc:
            return AIInfo(edition="local", engine="Ollama", model=self.model, ready=False, note=str(exc))

    def _chat(self, system: str, user: str, schema: type) -> str:
        r = self._post("/api/chat", {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "format": schema.model_json_schema(),
            "stream": False,
            "think": False,
            "options": {"temperature": 0},
        })
        return r.get("message", {}).get("content", "")

    def parse_logbook(self, text: str, ctx: AIContext) -> list[FleetUpdate]:
        user = (f"Today's date: {ctx.today.isoformat()}\nOpen job cards:\n" + ("\n".join(ctx.open_jobs) or "(none)")
                + f"\n\nLogbook text:\n\"\"\"\n{text}\n\"\"\"")
        try:
            return ParsedUpdates.model_validate_json(self._chat(PARSER_INSTRUCTIONS, user, ParsedUpdates)).updates
        except ValueError as exc:
            raise AIUnavailable(f"Ollama returned malformed JSON: {exc}") from exc

    def interpret_question(self, question: str, ctx: AIContext) -> QueryIntent:
        try:
            return QueryIntent.model_validate_json(self._chat(QUESTION_INSTRUCTIONS, f"Question: {question}", QueryIntent))
        except ValueError as exc:
            raise AIUnavailable(f"Ollama returned malformed JSON: {exc}") from exc

    def freeform_answer(self, question: str, facts: str) -> str | None:
        try:
            r = self._post("/api/chat", {
                "model": self.model, "stream": False, "think": False, "options": {"temperature": 0.2},
                "messages": [
                    {"role": "system", "content": "You are the KMRL depot operations assistant. Answer briefly using ONLY the facts given."},
                    {"role": "user", "content": f"Depot facts:\n{facts}\n\nQuestion: {question}"},
                ],
            })
            return r.get("message", {}).get("content") or None
        except AIUnavailable:
            return None
