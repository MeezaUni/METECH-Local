"""On-device engine: the KMRL copilot model fine-tuned by ml/train.py.

Runs fully offline with Hugging Face transformers on the GPU when one is
available (CPU otherwise). The model loads in a background thread at startup so
the API is usable immediately; requests made before it is ready fall back to the
rule engine.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path

from ..models import AIInfo, FleetUpdate, QueryIntent
from .base import AIContext, AIUnavailable
from .local_format import ask_messages, ground_and_resolve, load_json, parse_messages

log = logging.getLogger(__name__)
_warnings = threading.local()   # per-request notes about suggestions the grounding check dropped


class LocalModelProvider:
    def __init__(self, model_dir: Path, device: str = "auto"):
        self.model_dir = model_dir
        self.requested_device = device
        self.device = "cpu"
        self._model = None
        self._tok = None
        self._error: str | None = None
        self._lock = threading.Lock()   # one generation at a time on a single GPU
        self._ready = threading.Event()
        self.base_model = "?"
        if (model_dir / "config.json").exists():
            threading.Thread(target=self._load, name="kmrl-model-loader", daemon=True).start()
        else:
            self._error = f"No trained model at {model_dir}. Run ml/train.py (see ml/README.md)."

    def _load(self) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            started = time.perf_counter()
            if self.requested_device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = self.requested_device
            dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
            self._tok = AutoTokenizer.from_pretrained(self.model_dir)
            self._model = AutoModelForCausalLM.from_pretrained(self.model_dir, dtype=dtype).to(self.device).eval()
            meta = self.model_dir / "kmrl_model.json"
            if meta.exists():
                self.base_model = json.loads(meta.read_text(encoding="utf-8")).get("base_model", "?")
            log.info("Local model loaded on %s in %.1fs", self.device, time.perf_counter() - started)
            self._ready.set()
        except Exception as exc:  # missing torch, corrupt weights, CUDA error...
            self._error = f"Could not load the local model: {exc}"
            log.exception("Local model failed to load")

    # ------------------------------------------------------------------ API
    def info(self) -> AIInfo:
        name = f"kmrl-copilot ({self.base_model.split('/')[-1]} fine-tune, {self.device.upper()})"
        if self._ready.is_set():
            return AIInfo(edition="local", engine="On-device model", model=name, ready=True)
        note = self._error or "Loading the on-device model…"
        return AIInfo(edition="local", engine="On-device model", model=name, ready=False, note=note)

    def _generate(self, messages: list[dict], max_new_tokens: int) -> str:
        if not self._ready.wait(timeout=0 if self._error else 30):
            raise AIUnavailable(self._error or "model still loading")
        import torch

        prompt = self._tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._tok(prompt, return_tensors="pt").to(self.device)
        with self._lock, torch.inference_mode():
            out = self._model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=self._tok.pad_token_id or self._tok.eos_token_id,
            )
        return self._tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    def parse_logbook(self, text: str, ctx: AIContext) -> list[FleetUpdate]:
        raw = self._generate(parse_messages(text, ctx.today), max_new_tokens=min(1024, 160 + 4 * len(text)))
        try:
            proposed = [{"train_name": "", **u} for u in load_json(raw).get("updates", [])]   # closures may not name a train
            kept, dropped = ground_and_resolve(proposed, text, ctx.today)
            _warnings.items = dropped
            return [FleetUpdate.model_validate(u) for u in kept]
        except (ValueError, TypeError, AttributeError) as exc:
            raise AIUnavailable(f"model output was not valid: {exc}") from exc

    def take_warnings(self) -> list[str]:
        """Notes from the last parse on this thread (suggestions dropped by the grounding check)."""
        items, _warnings.items = getattr(_warnings, "items", []), []
        return items

    def interpret_question(self, question: str, ctx: AIContext) -> QueryIntent:
        raw = self._generate(ask_messages(question), max_new_tokens=64)
        try:
            return QueryIntent.model_validate(load_json(raw))
        except ValueError as exc:
            raise AIUnavailable(f"model output was not valid: {exc}") from exc

    def freeform_answer(self, question: str, facts: str) -> str | None:
        return None  # the small model is trained for structured extraction only
