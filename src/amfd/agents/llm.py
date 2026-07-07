from __future__ import annotations

import os
from importlib import import_module
from typing import Any

from amfd.agents.tools import build_tool_registry
from amfd.core.config import DiagnosisConfig


def resolve_chat_model(config: DiagnosisConfig) -> Any | None:
    provider = config.llm_provider.strip().lower()
    model_name = config.llm_model.strip()
    if provider == "auto":
        for candidate in ("openai", "anthropic", "gemini"):
            llm = _build_chat_model(candidate, model_name or _model_from_env(candidate))
            if llm is not None:
                return llm
        return None
    return _build_chat_model(provider, model_name)


def bind_manufacturing_tools(llm: Any) -> Any:
    """Attach the manufacturing tool registry to a LangChain chat model."""
    return llm.bind_tools(build_tool_registry())


def _build_chat_model(provider: str, model_name: str) -> Any | None:
    if not model_name:
        return None

    module_name: str
    class_name: str
    if provider == "openai":
        module_name = "langchain_openai"
        class_name = "ChatOpenAI"
    elif provider == "anthropic":
        module_name = "langchain_anthropic"
        class_name = "ChatAnthropic"
    elif provider == "gemini":
        module_name = "langchain_google_genai"
        class_name = "ChatGoogleGenerativeAI"
    else:
        return None

    try:
        module = import_module(module_name)
    except ImportError:
        return None

    chat_model = getattr(module, class_name, None)
    if chat_model is None:
        return None

    return chat_model(model=model_name, temperature=0)


def _model_from_env(provider: str) -> str:
    if provider == "openai":
        return os.getenv("OPENAI_MODEL", "")
    if provider == "anthropic":
        return os.getenv("ANTHROPIC_MODEL", "")
    if provider == "gemini":
        return os.getenv("GEMINI_MODEL", "")
    return ""
