"""Tools: the agent's hands.

The model can only *ask* for something to happen. A tool is the harness-side
function that actually does it. Two halves travel separately:

1. **the schema** — name, description, parameters — this is prompt material and
   lives in `config/tools.json`, where you can edit it without touching code.
2. **the implementation** — a Python function — this lives here, because it is
   behavior and needs code review.

`build_toolbox()` joins them by name. A schema with no implementation is a
startup error; an implementation with no schema is simply not exposed to the
model (it is unreachable, so keeping one around is pointless but harmless).

Descriptions matter: they are read at the moment the model chooses. A vague
description is a bug in how the model behaves, not in the model.

Safety note: every tool below is scoped to the workspace directory it is given,
and every failure is returned to the model as an "ERROR: ..." string instead of
raising. Errors are data in an agent loop — the model can often recover if it
can see what went wrong.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .config import DEFAULT_TOOLS_PATH, ConfigError, load_tool_schemas


class ToolError(RuntimeError):
    """A tool failed in a way the model should be told about."""


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[..., str]

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class Toolbox:
    """A registry of tools, plus the dispatch layer that runs them."""

    def __init__(self, workspace: Path, max_output: int = 4000):
        self.workspace = Path(workspace).resolve()
        self.max_output = max_output
        self._tools: dict[str, Tool] = {}

    # -- registration ------------------------------------------------------- #

    def register(self, tool: Tool) -> Tool:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool name: {tool.name}")
        self._tools[tool.name] = tool
        return tool

    def schemas(self) -> list[dict]:
        return [t.schema() for t in self._tools.values()]

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    # -- dispatch ----------------------------------------------------------- #

    def call(self, name: str, arguments: str) -> str:
        """Run a tool by name. Never raises: returns a string for the model."""
        tool = self._tools.get(name)
        if tool is None:
            return f"ERROR: unknown tool '{name}'. Available: {', '.join(self.names)}"

        try:
            args = json.loads(arguments or "{}")
        except json.JSONDecodeError as exc:
            return f"ERROR: arguments were not valid JSON ({exc}). You sent: {arguments!r}"
        if not isinstance(args, dict):
            return "ERROR: arguments must be a JSON object"

        try:
            result = tool.fn(**args)
        except ToolError as exc:
            return f"ERROR: {exc}"
        except TypeError as exc:  # wrong/missing parameters
            return f"ERROR: bad arguments for '{name}': {exc}"
        except Exception as exc:  # noqa: BLE001 - a tool must never crash the loop
            return f"ERROR: '{name}' failed: {type(exc).__name__}: {exc}"

        result = str(result)
        if len(result) > self.max_output:
            result = result[: self.max_output] + f"\n... [truncated, {len(result)} chars total]"
        return result

    # -- shared helpers ----------------------------------------------------- #

    def _resolve(self, raw: str) -> Path:
        """Resolve a user-supplied path, refusing anything outside the workspace."""
        path = (self.workspace / raw).resolve()
        if path != self.workspace and self.workspace not in path.parents:
            raise ToolError(f"path escapes the workspace: {raw!r}")
        return path


# --------------------------------------------------------------------------- #
# Implementations, keyed by the name used in config/tools.json
# --------------------------------------------------------------------------- #


def _implementations(box: Toolbox) -> dict[str, Callable[..., str]]:
    def list_dir(path: str = ".") -> str:
        target = box._resolve(path)
        if not target.is_dir():
            raise ToolError(f"not a directory: {path!r}")
        entries = sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name))
        if not entries:
            return f"{path} is empty"
        lines = [f"{'d' if e.is_dir() else 'f'} {e.name}" for e in entries]
        return "\n".join(lines)

    def read_file(path: str, max_bytes: int = 20000) -> str:
        target = box._resolve(path)
        if not target.is_file():
            raise ToolError(f"no such file: {path!r}")
        data = target.read_bytes()[: max(1, int(max_bytes))]
        return data.decode("utf-8", "replace") or "(empty file)"

    def write_file(path: str, content: str) -> str:
        target = box._resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"wrote {len(content)} chars to {target.relative_to(box.workspace)}"

    def run_python(code: str, timeout_seconds: int = 10) -> str:
        try:
            proc = subprocess.run(
                [sys.executable, "-c", code],
                cwd=box.workspace,
                capture_output=True,
                text=True,
                timeout=min(max(1, int(timeout_seconds)), 30),
            )
        except subprocess.TimeoutExpired:
            raise ToolError(f"python timed out after {timeout_seconds}s") from None
        output = (proc.stdout or "") + (("\n[stderr]\n" + proc.stderr) if proc.stderr else "")
        return f"exit code {proc.returncode}\n{output.strip() or '(no output)'}"

    return {
        "list_dir": list_dir,
        "read_file": read_file,
        "write_file": write_file,
        "run_python": run_python,
    }


def build_toolbox(
    workspace: Path,
    schemas_path: str | Path | None = None,
    max_output: int = 4000,
) -> Toolbox:
    """Build a toolbox by binding `config/tools.json` to the implementations above."""
    box = Toolbox(workspace, max_output)
    implementations = _implementations(box)
    path = Path(schemas_path) if schemas_path is not None else DEFAULT_TOOLS_PATH

    for spec in load_tool_schemas(path):
        name = spec["name"]
        fn = implementations.get(name)
        if fn is None:
            raise ConfigError(
                f"tool {name!r} is declared in {path} but has no implementation. "
                f"Implemented: {', '.join(sorted(implementations))}"
            )
        box.register(
            Tool(
                name=name,
                description=spec["description"],
                parameters=spec["parameters"],
                fn=fn,
            )
        )
    return box
