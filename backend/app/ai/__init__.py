"""AI engine selection. The local edition never calls a cloud API.

KMRL_LOCAL_ENGINE=trained (default)  fine-tuned on-device kmrl-logreader model (ml/train.py)
KMRL_LOCAL_ENGINE=ollama             a general LLM from a local Ollama install
"""
from .base import AIContext, AIProvider, AIUnavailable


def build_provider(settings) -> AIProvider:
    if settings.local_engine == "ollama":
        from .ollama import OllamaProvider
        return OllamaProvider(settings.ollama_host, settings.ollama_model)
    from .local_model import LocalModelProvider
    return LocalModelProvider(settings.model_dir, settings.model_device)


__all__ = ["AIContext", "AIProvider", "AIUnavailable", "build_provider"]
