"""A tiny, dependency-free agent harness you can read in one sitting."""

from .config import (
    DEFAULT_PROMPT_PATH,
    DEFAULT_TOOLS_PATH,
    ConfigError,
    load_text,
    load_tool_schemas,
)
from .harness import Agent, AgentResult
from .providers import (
    AssistantMessage,
    MockProvider,
    OpenAICompatibleProvider,
    Provider,
    ProviderError,
    ToolCall,
)
from .tools import Tool, Toolbox, ToolError, build_toolbox

__all__ = [
    "Agent",
    "AgentResult",
    "ConfigError",
    "DEFAULT_PROMPT_PATH",
    "DEFAULT_TOOLS_PATH",
    "AssistantMessage",
    "MockProvider",
    "OpenAICompatibleProvider",
    "Provider",
    "ProviderError",
    "Tool",
    "ToolCall",
    "Toolbox",
    "ToolError",
    "build_toolbox",
    "load_text",
    "load_tool_schemas",
]
