from __future__ import annotations

from typing import Any

from amfd.agents.tools import build_tool_registry


def bind_manufacturing_tools(llm: Any) -> Any:
    """Attach the manufacturing tool registry to a LangChain chat model."""
    return llm.bind_tools(build_tool_registry())

