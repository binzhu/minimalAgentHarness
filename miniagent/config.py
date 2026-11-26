"""Configuration: prompts and tool schemas live in files, not in code.

Two things are configuration rather than logic, because they change for reasons
that have nothing to do with the loop:

* the **system prompt** — policy text you tune constantly, and want to diff,
* the **tool schemas** — the JSON the model reads to decide what it can call.

Neither is allowed to fail silently: a missing or malformed file raises
`ConfigError` so a run never quietly uses stale built-in defaults.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Everything is resolved from the project root, so the CLI works from any cwd.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROMPT_PATH = PROJECT_ROOT / "config" / "system_prompt.md"
DEFAULT_TOOLS_PATH = PROJECT_ROOT / "config" / "tools.json"


class ConfigError(RuntimeError):
    """A configuration file is missing, unreadable, or malformed."""


def load_text(path: str | Path) -> str:
    """Read a plain-text file (the system prompt) and refuse an empty one."""
    target = Path(path)
    try:
        text = target.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise ConfigError(f"prompt file not found: {target}") from None
    except OSError as exc:
        raise ConfigError(f"could not read {target}: {exc}") from exc
    if not text.strip():
        raise ConfigError(f"prompt file is empty: {target}")
    return text


def load_tool_schemas(path: str | Path) -> list[dict[str, Any]]:
    """Read tool schemas from `{"tools": [...]}` (or a bare list) and validate shape."""
    target = Path(path)
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ConfigError(f"tool schema file not found: {target}") from None
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{target} is not valid JSON: {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"could not read {target}: {exc}") from exc

    specs = data.get("tools") if isinstance(data, dict) else data
    if not isinstance(specs, list) or not specs:
        raise ConfigError(f'{target} must contain a non-empty "tools" list')

    seen: set[str] = set()
    for index, spec in enumerate(specs):
        where = f"{target}: tool #{index}"
        if not isinstance(spec, dict):
            raise ConfigError(f"{where} must be an object")
        for key in ("name", "description", "parameters"):
            if key not in spec:
                raise ConfigError(f"{where} is missing {key!r}")
        if not isinstance(spec["name"], str) or not spec["name"].strip():
            raise ConfigError(f"{where} has an invalid 'name'")
        if not isinstance(spec["description"], str) or not spec["description"].strip():
            # The description *is* the prompt; an empty one is a bug, not a default.
            raise ConfigError(f"{where} ({spec['name']}) needs a description")
        if not isinstance(spec["parameters"], dict):
            raise ConfigError(f"{where} ({spec['name']}) 'parameters' must be an object")
        if spec["name"] in seen:
            raise ConfigError(f"{where}: duplicate tool name {spec['name']!r}")
        seen.add(spec["name"])
    return specs
