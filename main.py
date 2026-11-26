#!/usr/bin/env python3
"""Run the mini agent.

Local Ollama on the LAN (the default provider), as a back-and-forth conversation:

    python3 main.py
    python3 main.py "list this folder and summarise what it does"
    python3 main.py --model qwen2.5-coder:14b "write fizzbuzz.py and test it"
    python3 main.py --no-stream "same task, but wait for the whole turn"

Any OpenAI-compatible endpoint:

    export OPENAI_API_KEY=sk-...
    python3 main.py --provider openai --model gpt-4o-mini "summarise this repo"

Type `/context` at the prompt to see what the next request will send, split by
system prompt, tool schemas, and each kind of message. It is answered locally.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from miniagent import (
    DEFAULT_PROMPT_PATH,
    DEFAULT_TOOLS_PATH,
    Agent,
    ConfigError,
    OpenAICompatibleProvider,
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
        # A turn that never produced a reply must not leak into the next one.
        _ui["streamed_turn"] = False


def show_context(agent: Agent) -> None:
    """`/context` — a local answer, so it costs nothing and works offline."""
    report = agent.context_report()
    label_width = max(len(section.label) for section in report.sections)

    print(f"\n{BOLD}context{RESET}   {len(agent.messages)} messages, "
          f"{report.chars:,} chars ~ {report.tokens:,} tokens (estimate)\n")
    for section in report.sections:
        print(f"  {section.label:<{label_width}}  {section.count:>4}  "
              f"{section.chars:>8,} chars  ~{section.tokens:>6,} tok")
    print(f"  {'total':<{label_width}}  {'':>4}  "
          f"{report.chars:>8,} chars  ~{report.tokens:>6,} tok\n")
    print(f"{DIM}estimate is chars / 4; JSON framing and chat-template tokens "
          f"are not counted{RESET}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="A minimal agent harness.")
    parser.add_argument("task", nargs="?",
                        help="First message. Omit it to start with an empty conversation")
    parser.add_argument("--provider", choices=["ollama", "openai"], default="ollama",
                        help="'ollama' = LAN Ollama with local defaults")
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
    exit_code = 0

    def turn(text: str) -> None:
        """One exchange. A provider error ends this turn, not the session."""
        nonlocal exit_code
        try:
            result = agent.run(text)
        except KeyboardInterrupt:
            # Ctrl-C during a turn leaves a half-finished transcript behind, so
            # quit rather than continue from it: an assistant message with tool
            # calls but no matching tool results is rejected by any server.
            print()
            raise SystemExit(130)
        if result.stop_reason == "provider_error":
            print(f"\nprovider error: {result.error}", file=sys.stderr)
            exit_code = 1

    if args.task:
        turn(args.task)

    print(f"{DIM}type 'quit' or press Ctrl-D to end the conversation, "
          f"/context for the context budget{RESET}")
    while True:
        try:
            line = input(f"{BOLD}you{RESET}      ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line in {"quit", "exit"}:
            break
        if line == "/context":
            # Handled here, not by the agent: this is a question about the
            # transcript, and answering it through the model would be both
            # slower and a way to grow the very context it is reporting on.
            show_context(agent)
            continue
        turn(line)

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
