#!/usr/bin/env python3
"""Drive a running miniagent server: attach a terminal, or manage its lifetime.

The server owns the conversation; this program owns the terminal. That split is
the whole point — closing your terminal must not close the session.

    python3 agentctl.py start                 # start the server, detached
    python3 agentctl.py attach                # connect this terminal
    python3 agentctl.py serve                 # start if needed, then attach
    python3 agentctl.py serve "summarise this repo"
    python3 agentctl.py status                # state, owner, context size
    python3 agentctl.py stop                  # graceful: waits for the turn
    python3 agentctl.py stop --now            # don't wait

Attaching while another client is idle offers a takeover; attaching while a
turn is running is refused ("please wait"), never granted. There is no
automatic reconnect: if the server or the stream goes away, rerun the command.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from main import BOLD, DIM, RESET, render
from miniagent.config import PROJECT_ROOT
from miniagent.server import RUNTIME_DIR

PING_EVERY = 15.0  # must be well under Session.lease_ttl (60s)
START_TIMEOUT = 20.0

# Set by the event-stream and ping threads to stop the prompt loop. The reason
# is printed by the thread that noticed it; this is only for the exit path.
_STATE: dict = {"leaving": threading.Event(), "bail": None, "expect_reset": 0}


# --------------------------------------------------------------------------- #
# Talking to the server
# --------------------------------------------------------------------------- #


class Unreachable(RuntimeError):
    pass


def _request(
    base: str,
    token: str,
    method: str,
    path: str,
    payload: dict | None = None,
    timeout: float = 10.0,
) -> tuple[int, dict]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        base + path,
        data=data,
        method=method,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
            return response.status, (json.loads(body) if body else {})
    except urllib.error.HTTPError as exc:
        body = exc.read()
        try:
            return exc.code, (json.loads(body) if body else {})
        except json.JSONDecodeError:
            return exc.code, {"error": body.decode("utf-8", "replace")[:300]}
    except (urllib.error.URLError, OSError) as exc:
        raise Unreachable(str(exc)) from exc


def _runtime() -> dict | None:
    try:
        return json.loads((RUNTIME_DIR / "server.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _base(info: dict) -> str:
    return f"http://{info['host']}:{info['port']}"


def _alive(info: dict, timeout: float = 1.0) -> bool:
    try:
        code, _ = _request(_base(info), info.get("token", ""), "GET", "/status",
                           timeout=timeout)
        return code == 200
    except Unreachable:
        return False


def _client_id() -> str:
    return f"{socket.gethostname()}:{os.getpid()}"


def _die(message: str, code: int = 1) -> int:
    print(f"error: {message}", file=sys.stderr)
    return code


# --------------------------------------------------------------------------- #
# Background threads
# --------------------------------------------------------------------------- #


def _bail(reason: str) -> None:
    """Stop the prompt loop from a background thread.

    The main thread may be blocked in `input()` or waiting for a turn, so a
    flag alone would not wake it. SIGINT does, and lands as KeyboardInterrupt
    in the main thread only.
    """
    if _STATE["leaving"].is_set() or _STATE["bail"]:
        return
    _STATE["bail"] = reason
    sys.stderr.write(f"\n{reason}\n")
    sys.stderr.flush()
    try:
        os.kill(os.getpid(), signal.SIGINT)
    except OSError:  # pragma: no cover - only if signals are unavailable
        os._exit(1)


def _stream(base: str, token: str, since: int, events: queue.Queue) -> None:
    request = urllib.request.Request(
        f"{base}/events?since={since}",
        headers={"Authorization": f"Bearer {token}"},
    )
    try:
        with urllib.request.urlopen(request) as response:
            for raw in response:
                line = raw.decode("utf-8", "replace").rstrip("\r\n")
                if not line.startswith("data:"):
                    continue  # comment/heartbeat or blank separator
                try:
                    record = json.loads(line[len("data:"):].strip())
                except json.JSONDecodeError:
                    continue
                kind = record.get("event")
                if kind == "detached":
                    _bail("another client took over this session")
                    return
                if kind == "shutdown":
                    _bail("the server is shutting down")
                    return
                events.put(record)
    except (urllib.error.URLError, OSError) as exc:
        _bail(f"lost the connection to the server ({exc}); not reconnecting")
        return
    _bail("the server closed the event stream; not reconnecting")


def _ping(base: str, token: str, lease: str) -> None:
    """Tell the server we are still here, and learn if we were replaced."""
    while not _STATE["leaving"].wait(PING_EVERY):
        try:
            code, _ = _request(base, token, "POST", "/ping", {"lease": lease}, timeout=5)
        except Unreachable:
            continue  # a single missed ping is not evidence of anything
        if code == 401:
            _bail("another client took over this session")
            return


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def _duration(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"


def _show_status(status: dict) -> None:
    holder = status.get("holder")
    owner = (
        f"{holder['client']} ({holder['state']}, seen {holder['seen_ago']}s ago)"
        if holder else "nobody"
    )
    context = status.get("context") or {}
    print(f"{BOLD}status{RESET}    {status.get('state')}, up "
          f"{_duration(status.get('uptime') or 0)}")
    print(f"  model     {status.get('model') or '?'}")
    print(f"  owner     {owner}")
    print(f"  context   {status.get('messages', 0)} messages, "
          f"{context.get('chars', 0):,} chars "
          f"~{context.get('tokens', 0):,} tokens (estimate)")


def _show_context(status: dict) -> None:
    context = status.get("context") or {}
    sections = context.get("sections") or []
    if not sections:
        print("context is empty")
        return
    width = max(len(section["label"]) for section in sections)
    print(f"\n{BOLD}context{RESET}   {status.get('messages', 0)} messages, "
          f"{context.get('chars', 0):,} chars ~ {context.get('tokens', 0):,} tokens "
          f"(estimate)\n")
    for section in sections:
        print(f"  {section['label']:<{width}}  {section['count']:>4}  "
              f"{section['chars']:>8,} chars  ~{section['tokens']:>6,} tok")
    print(f"  {'total':<{width}}  {'':>4}  {context.get('chars', 0):>8,} chars  "
          f"~{context.get('tokens', 0):>6,} tok\n")


HELP = f"""{BOLD}commands{RESET}
  /status    who owns the session, and how big the context is
  /context   context broken down by section
  /reset     discard the conversation (refused while a turn runs)
  /detach    leave the session running and exit
  /shutdown  stop the server; the conversation is lost
  /help      this
Anything else is sent to the model."""


# --------------------------------------------------------------------------- #
# One turn
# --------------------------------------------------------------------------- #


def _report(code: int, payload: dict) -> None:
    message = (payload.get("message") or payload.get("error") or f"HTTP {code}")
    print(f"{BOLD}error{RESET}    {message}", file=sys.stderr)


def _consume(events: queue.Queue) -> bool:
    """Render events until the turn ends. Nothing reads stdin meanwhile."""
    while True:
        record = events.get()
        kind = record.get("event")
        if kind == "turn_end":
            if not record.get("ok"):
                print(f"{DIM}turn rolled back ({record.get('stop_reason')}); "
                      f"the conversation is unchanged{RESET}\n")
            return True
        if kind == "gap":
            print(f"{DIM}... dropped a burst of events; resuming at seq "
                  f"{record.get('seq')}{RESET}", file=sys.stderr)
            continue
        if kind == "reset":
            if _STATE["expect_reset"]:
                _STATE["expect_reset"] -= 1
                continue
            print(f"{DIM}context cleared{RESET}")
            continue
        render(kind, record)


# --------------------------------------------------------------------------- #
# The prompt loop
# --------------------------------------------------------------------------- #


def _leave(base: str, token: str, lease: str) -> None:
    if _STATE["leaving"].is_set():
        return
    _STATE["leaving"].set()
    try:
        _request(base, token, "POST", "/detach", {"lease": lease}, timeout=3)
    except Unreachable:
        pass  # the lease expires on its own; nothing to clean up here


def _command(line: str, base: str, token: str, lease: str) -> str | None:
    """Handle a slash command. Returns "leave" when the client should exit."""
    words = line.split()
    name = words[0].lower()
    if name in {"/help", "/?"}:
        print(HELP)
        return None
    if name in {"/status", "/context"}:
        try:
            code, status = _request(base, token, "GET", "/status")
        except Unreachable as exc:
            print(f"error: {exc}", file=sys.stderr)
            return None
        if code != 200:
            _report(code, status)
        elif name == "/context":
            _show_context(status)
        else:
            _show_status(status)
        return None
    if name == "/reset":
        try:
            code, payload = _request(base, token, "POST", "/reset", {"lease": lease})
        except Unreachable as exc:
            print(f"error: {exc}", file=sys.stderr)
            return None
        if code == 202:
            _STATE["expect_reset"] += 1
            print(f"{DIM}context cleared{RESET}")
        elif code == 401:
            _bail("another client took over this session")
        else:
            _report(code, payload)
        return None
    if name in {"/detach", "/exit", "/quit"}:
        print(f"{DIM}detached; the session is still running "
              f"(agentctl attach to return){RESET}")
        return "leave"
    if name == "/shutdown":
        _STATE["leaving"].set()
        try:
            code, payload = _request(base, token, "POST", "/shutdown", {})
        except Unreachable as exc:
            print(f"error: {exc}", file=sys.stderr)
            return "leave"
        if code != 202:
            _report(code, payload)
            return "leave"
        if payload.get("mode") == "graceful" and payload.get("turn_active"):
            print("turn in progress; the server stops when it finishes")
        print(f"{DIM}server stopped; session context discarded{RESET}")
        return "leave"
    print(f"unknown command {name}; try /help")
    return None


def _session_loop(base: str, token: str, lease: str, since: int,
                  task: str | None, info: dict) -> int:
    events: queue.Queue = queue.Queue()
    threading.Thread(target=_stream, args=(base, token, since, events),
                     name="events", daemon=True).start()
    threading.Thread(target=_ping, args=(base, token, lease),
                     name="ping", daemon=True).start()

    print(f"{DIM}{info.get('model') or '?'} @ {base} (pid {info.get('pid')}){RESET}")
    print(f"{DIM}type /help; Ctrl-D or /detach leaves the session running{RESET}\n")

    pending = task
    code = 0
    try:
        while True:
            if pending:
                line, pending = pending.strip(), None
                print(f"{BOLD}you{RESET}      {line}")
            else:
                try:
                    line = input(f"{BOLD}you{RESET}      ").strip()
                except EOFError:
                    print()
                    break
                except KeyboardInterrupt:
                    print()
                    break

            if not line:
                continue
            if line in {"quit", "exit"} or line.startswith("/"):
                if _command(line, base, token, lease) == "leave":
                    break
                continue

            try:
                status, payload = _request(base, token, "POST", "/turn",
                                           {"lease": lease, "text": line}, timeout=30)
            except Unreachable as exc:
                print(f"error: {exc}", file=sys.stderr)
                code = 1
                break
            if status == 401:
                _bail("another client took over this session")
                code = 1
                break
            if status != 202:
                _report(status, payload)
                continue
            _consume(events)
    except KeyboardInterrupt:
        if _STATE["bail"]:
            code = 1
        else:
            print(f"{DIM}detached; a turn in flight finishes on the server{RESET}")
    finally:
        _leave(base, token, lease)
    return code


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #


def _server_argv(args) -> list[str]:
    argv = [sys.executable, "-m", "miniagent.server",
            "--host", args.host, "--port", str(args.port),
            "--provider", args.provider, "--max-steps", str(args.max_steps),
            "--workspace", args.workspace]
    for flag, value in (("--model", args.model), ("--base-url", args.base_url),
                        ("--api-key", args.api_key),
                        ("--system-prompt", args.system_prompt),
                        ("--tools", args.tools)):
        if value:
            argv += [flag, value]
    if args.no_stream:
        argv.append("--no-stream")
    if args.verbose:
        argv.append("--verbose")
    return argv


def _start(args) -> dict:
    info = _runtime()
    if info and _alive(info):
        print(f"{DIM}already running: {_base(info)} (pid {info.get('pid')}){RESET}")
        return info

    # A stale file from a SIGKILLed server would otherwise be mistaken for ours.
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    (RUNTIME_DIR / "server.json").unlink(missing_ok=True)

    log_path = RUNTIME_DIR / "server.log"
    with open(log_path, "ab") as log:
        proc = subprocess.Popen(
            _server_argv(args),
            cwd=str(PROJECT_ROOT),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # survives this terminal, and Ctrl-C
        )

    deadline = time.monotonic() + START_TIMEOUT
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return _start_failed(proc.returncode, log_path)
        info = _runtime()
        if info and _alive(info, timeout=0.5):
            print(f"{DIM}started: {_base(info)} (pid {info.get('pid')}){RESET}")
            return info
        time.sleep(0.1)
    _die(f"server did not come up within {START_TIMEOUT:.0f}s; see {log_path}")
    raise SystemExit(1)


def _start_failed(returncode: int, log_path: Path) -> dict:
    tail = ""
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        tail = "\n".join(lines[-5:])
    except OSError:
        pass
    print(f"error: server exited with code {returncode}", file=sys.stderr)
    if tail:
        print(tail, file=sys.stderr)
    raise SystemExit(1)


def _attach(args, info: dict | None = None) -> int:
    info = info or _runtime()
    if not info:
        return _die("no server is running; start one with: agentctl start")
    base, token = _base(info), info["token"]
    identity = {"client": _client_id(), "host": socket.gethostname()}

    try:
        code, payload = _request(base, token, "POST", "/attach", identity)
    except Unreachable as exc:
        return _die(f"cannot reach {base}: {exc}")

    if code == 409 and payload.get("reason") == "held":
        holder = payload.get("holder") or {}
        description = (f"{holder.get('client', 'another client')} "
                       f"({holder.get('state', '?')}, seen "
                       f"{holder.get('seen_ago', 0)}s ago)")
        if not args.takeover:
            if not sys.stdin.isatty():
                return _die(f"session is held by {description}; "
                            f"rerun with --takeover to take it")
            print(f"{DIM}session is held by {description}{RESET}")
            try:
                answer = input("take over? [y/N] ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print()
                return 1
            if answer not in {"y", "yes"}:
                print("left the session alone")
                return 0
        try:
            code, payload = _request(base, token, "POST", "/takeover", identity)
        except Unreachable as exc:
            return _die(f"cannot reach {base}: {exc}")

    if code != 200:
        if payload.get("reason") == "busy":
            return _die(payload.get("message") or "turn in progress, please wait")
        if code == 503:
            return _die("server is shutting down")
        return _die(payload.get("message") or payload.get("error") or f"HTTP {code}")

    return _session_loop(base, token, payload["lease"], payload.get("since", 0),
                         getattr(args, "task", None), info)


def _stop(args) -> int:
    info = _runtime()
    if not info:
        print("not running")
        return 1
    base, token = _base(info), info["token"]
    path = "/shutdown?now=1" if args.now else "/shutdown"
    try:
        code, payload = _request(base, token, "POST", path, {}, timeout=10)
    except Unreachable:
        (RUNTIME_DIR / "server.json").unlink(missing_ok=True)
        return _die(f"server at {base} was not reachable; removed its runtime file")
    if code != 202:
        return _die(payload.get("message") or payload.get("error") or f"HTTP {code}")

    if payload.get("turn_active") and not args.now:
        print(f"{DIM}turn in progress; waiting for it to finish "
              f"(Ctrl-C to give up){RESET}")
    deadline = time.monotonic() + (10 if args.now else 180)
    while time.monotonic() < deadline:
        if not _alive(info, timeout=0.5):
            print("stopped")
            return 0
        time.sleep(0.2)
    return _die("still running; try: agentctl stop --now")


def _status(args) -> int:  # noqa: ARG001
    info = _runtime()
    if not info:
        print("not running")
        return 1
    try:
        code, status = _request(_base(info), info["token"], "GET", "/status")
    except Unreachable as exc:
        return _die(f"server file exists but nothing answers at {_base(info)}: {exc}")
    if code != 200:
        return _die(status.get("error") or f"HTTP {code}")
    _show_status(status)
    return 0


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def _add_server_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--host", default="127.0.0.1",
                        help="Bind address for a new server (no TLS: keep it local)")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--provider", choices=["ollama", "openai"], default="ollama")
    parser.add_argument("--model", default=os.environ.get("MINIAGENT_MODEL"))
    parser.add_argument("--base-url", default=os.environ.get("MINIAGENT_BASE_URL"))
    parser.add_argument("--api-key",
                        default=os.environ.get("MINIAGENT_API_KEY")
                        or os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--no-stream", action="store_true")
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument("--tools", default=None)
    parser.add_argument("--verbose", action="store_true")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Attach to, or manage, a miniagent server.")
    sub = parser.add_subparsers(dest="command", required=True)

    start = sub.add_parser("start", help="Start the server in the background")
    _add_server_args(start)

    serve = sub.add_parser("serve", help="Start the server if needed, then attach")
    _add_server_args(serve)
    serve.add_argument("--takeover", action="store_true")
    serve.add_argument("task", nargs="?", help="First message")

    attach = sub.add_parser("attach", help="Attach this terminal to a running server")
    attach.add_argument("--takeover", action="store_true",
                        help="Take the session without asking (fails if a turn is running)")
    attach.add_argument("task", nargs="?", help="First message")

    stop = sub.add_parser("stop", help="Stop the server")
    stop.add_argument("--now", action="store_true", help="Do not wait for a running turn")

    sub.add_parser("status", help="Show state, owner, and context size")

    args = parser.parse_args()
    try:
        if args.command == "start":
            _start(args)
            return 0
        if args.command == "serve":
            return _attach(args, _start(args))
        if args.command == "attach":
            return _attach(args)
        if args.command == "stop":
            return _stop(args)
        return _status(args)
    except KeyboardInterrupt:
        print()
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
