#!/usr/bin/env python3
"""Run the mini agent.

Offline demo (no API key, scripted fake model):

    python3 main.py --demo

Local Ollama on the LAN (the default provider):

    python3 main.py "list this folder and summarise what it does"
    python3 main.py --model qwen2.5-coder:14b "write fizzbuzz.py and test it"
    python3 main.py --no-stream "same task, but wait for the whole turn"

Any OpenAI-compatible endpoint:

    export OPENAI_API_KEY=sk-...
    python3 main.py --provider openai --model gpt-4o-mini "summarise this repo"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from miniagent import (
    DEFAULT_PROMPT_PATH,
    DEFAULT_TOOLS_PATH,
    Agent,
    AssistantMessage,
    ConfigError,
    MockProvider,
    OpenAICompatibleProvider,
    ToolCall,
    build_toolbox,
    load_text,
)

BOLD, DIM, RESET = "\033[1m", "\033[2m", "\033[0m"

# Ollama serves an OpenAI-compatible API under /v1. Model capabilities are
# reported by `GET /api/tags`; gemma4:12b lists "tools", which this harness needs.
OLLAMA_BASE_URL = os.environ.get("MINIAGENT_BASE_URL", "http://10.20.30.99:11434/v1")
OLLAMA_MODEL = os.environ.get("MINIAGENT_MODEL", "gemma4:12b")

# Rendering state: which channel is currently being written inline, and whether
# the turn that just ended was already streamed to the screen.
_ui = {"streaming": None, "streamed_turn": False}


def _close_stream() -> None:
    if _ui["streaming"] is not None:
        sys.stdout.write(f"{RESET}\n\n")
        sys.stdout.flush()
        _ui["streaming"] = None


def render(kind: str, event: dict) -> None:
    """Print the loop as it happens — this is your window into the agent."""
    if kind == "delta":
        # Fragments arrive one at a time; switch channel prefixes as needed.
        if _ui["streaming"] != event["kind"]:
            _close_stream()
            if event["kind"] == "reasoning":
                sys.stdout.write(f"{DIM}thinking  {RESET}{DIM}")
            else:
                sys.stdout.write(f"{BOLD}model{RESET}    ")
            _ui["streaming"] = event["kind"]
        sys.stdout.write(event["text"])
        sys.stdout.flush()
        _ui["streamed_turn"] = True
        return

    if kind == "start":
        print(f"{BOLD}task{RESET}     {event['task']}")
        print(f"{DIM}tools    {', '.join(event['tools'])}{RESET}\n")

    elif kind == "assistant":
        _close_stream()
        # Buffered providers hand over the whole turn here; streamed ones have
        # already shown it, so don't print it twice.
        if not _ui["streamed_turn"]:
            if event.get("reasoning"):
                thought = " ".join(event["reasoning"].split())[:300]
                print(f"{DIM}thinking  {thought}{RESET}")
            if event["content"]:
                print(f"{BOLD}model{RESET}    {event['content']}")
        for name in event["tool_calls"]:
            print(f"{BOLD}model{RESET}    {DIM}-> calls {name}{RESET}")

    elif kind == "tool_result":
        _close_stream()
        first, *rest = event["result"].splitlines() or [""]
        print(f"{BOLD}tool{RESET}     {event['name']}: {first}")
        for line in rest[:6]:
            print(f"         {DIM}{line}{RESET}")
        print()

    elif kind == "tool_skipped":
        _close_stream()
        print(f"{BOLD}guard{RESET}    {event['name']}: duplicate call, result already in context\n")

    elif kind == "final":
        _close_stream()
        if not _ui["streamed_turn"]:
            print(f"\n{BOLD}answer{RESET}   {event['content']}")
        print()
        _ui["streamed_turn"] = False

    elif kind == "stop":
        _close_stream()
        print(f"{BOLD}stopped{RESET}  {event['reason']} after {event['steps']} steps")
        _ui["streamed_turn"] = False

    elif kind == "error":
        _close_stream()
        print(f"{BOLD}error{RESET}    {event['message']}", file=sys.stderr)


def demo_script() -> list:
    """What a well-behaved model *would* emit for the demo task.

    The last step is a callable so it can read the real tool output — the same
    way a real model reacts to what it just saw.
    """
    box = {"path": "notes/hello.txt", "content": "Hello from your first agent loop.\n"}

    def announce(messages: list[dict]) -> AssistantMessage:
        last = messages[-1]["content"] if messages else ""
        return AssistantMessage(
            content=(
                "Done. I listed the workspace, wrote notes/hello.txt, and read it "
                f"back to confirm.\nThe file now contains: {json.dumps(last)}"
            )
        )

    return [
        AssistantMessage(
            content="I'll see what is here first.",
            tool_calls=[ToolCall("call_1", "list_dir", json.dumps({"path": "."}))],
        ),
        AssistantMessage(
            tool_calls=[
                ToolCall("call_2", "write_file", json.dumps(box)),
            ]
        ),
        AssistantMessage(
            tool_calls=[ToolCall("call_3", "read_file", json.dumps({"path": box["path"]}))],
        ),
        announce,
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="A minimal agent harness.")
    parser.add_argument("task", nargs="?", default="Inspect this folder and create notes/hello.txt",
                        help="What you want the agent to do")
    parser.add_argument("--demo", action="store_true", help="Run the offline scripted demo")
    parser.add_argument("--provider", choices=["mock", "ollama", "openai"], default="ollama",
                        help="'ollama' = LAN Ollama with local defaults; 'mock' = offline demo")
    parser.add_argument("--model", default=os.environ.get("MINIAGENT_MODEL"))
    parser.add_argument("--base-url", default=os.environ.get("MINIAGENT_BASE_URL"))
    parser.add_argument("--api-key", default=os.environ.get("MINIAGENT_API_KEY") or os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--workspace", default=".", help="Directory the tools are confined to")
    parser.add_argument("--no-stream", action="store_true",
                        help="Buffer each turn and print it at once instead of streaming")
    parser.add_argument("--system-prompt", default=None,
                        help=f"Path to the prompt file (default: {DEFAULT_PROMPT_PATH})")
    parser.add_argument("--tools", default=None,
                        help=f"Path to the tool schema file (default: {DEFAULT_TOOLS_PATH})")
    args = parser.parse_args()

    # Configuration is read from files, and failing to find it is an error —
    # never a silent fall back to a built-in default.
    prompt_path = Path(args.system_prompt) if args.system_prompt else DEFAULT_PROMPT_PATH
    tools_path = Path(args.tools) if args.tools else DEFAULT_TOOLS_PATH
    try:
        system_prompt = load_text(prompt_path)
        toolbox = build_toolbox(Path(args.workspace), tools_path)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.demo or args.provider == "mock":
        provider = MockProvider(demo_script())
    else:
        if args.provider == "ollama":
            base_url = args.base_url or OLLAMA_BASE_URL
            model = args.model or OLLAMA_MODEL
            # Ollama ignores the key, but the Authorization header must be present.
            api_key = args.api_key or "ollama"
        else:
            base_url = args.base_url or "https://api.openai.com/v1"
            model = args.model or "gpt-4o-mini"
            api_key = args.api_key
            if not api_key:
                print("error: no API key. Set OPENAI_API_KEY or pass --api-key.", file=sys.stderr)
                return 2
        mode = "buffered" if args.no_stream else "streaming"
        print(f"{DIM}{model} @ {base_url} ({mode}){RESET}")
        print(f"{DIM}prompt {prompt_path}{RESET}")
        print(f"{DIM}tools  {tools_path}{RESET}\n")
        provider = OpenAICompatibleProvider(
            model=model, api_key=api_key, base_url=base_url, stream=not args.no_stream
        )

    agent = Agent(
        provider=provider,
        toolbox=toolbox,
        system_prompt=system_prompt,
        max_steps=args.max_steps,
        on_event=render,
    )
    result = agent.run(args.task)

    if result.stop_reason == "provider_error":
        print(f"\nprovider error: {result.error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
