"""HTTP transport for a Session.

The server exists for one reason: the conversation should survive the terminal.
Everything else follows from that split.

Two shapes of endpoint:

* **request/response** — `POST /turn`, `/attach`, … one short connection each.
  Commands are operations on the session, never a place where a reply is
  streamed back, so a dropped command connection cannot corrupt anything.
* **event stream** — `GET /events?since=N`, one long-lived SSE connection.
  Server-sent events are one-way, which is exactly the shape of the data: the
  agent talks, the client listens and sends commands on a separate connection.

Auth is a bearer token minted at startup and written to `.miniagent/server.json`.
It covers the whole surface, `run_python` included: the tools execute arbitrary
code in this workspace, so the token is not decoration. There is no TLS; do not
bind this to a public interface. Use a tunnel or WireGuard for anything that
leaves the machine.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import secrets
import signal
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from .config import (
    DEFAULT_PROMPT_PATH,
    DEFAULT_TOOLS_PATH,
    PROJECT_ROOT,
    ConfigError,
    load_text,
)
from .harness import Agent
from .providers import OpenAICompatibleProvider
from .session import Session
from .tools import build_toolbox

RUNTIME_DIR = PROJECT_ROOT / ".miniagent"
HEARTBEAT = 15.0  # SSE comment interval; also how fast a dead peer is noticed
OLLAMA_BASE_URL = os.environ.get("MINIAGENT_BASE_URL", "http://10.20.30.99:11434/v1")
OLLAMA_MODEL = os.environ.get("MINIAGENT_MODEL", "gemma4:12b")


class Runtime:
    """The small files a second process needs to find and trust the server."""

    def __init__(self, directory: Path | str = RUNTIME_DIR) -> None:
        self.dir = Path(directory)

    @property
    def info_path(self) -> Path:
        return self.dir / "server.json"

    @property
    def pid_path(self) -> Path:
        return self.dir / "server.pid"

    def claim(self):
        """Take the single-instance lock, or fail because another server has it.

        `flock` is released by the kernel when the process dies, including on
        SIGKILL, so a crashed server never blocks the next start.
        """
        self.dir.mkdir(parents=True, exist_ok=True)
        handle = open(self.pid_path, "a+", encoding="utf-8")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            handle.close()
            raise RuntimeError(f"another server holds {self.pid_path}") from None
        handle.seek(0)
        handle.truncate()
        handle.write(f"{os.getpid()}\n")
        handle.flush()
        return handle

    def write_info(self, **fields) -> None:
        path = self.info_path
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.write(fd, json.dumps(fields, indent=2).encode("utf-8"))
        finally:
            os.close(fd)
        os.chmod(path, 0o600)  # the mode above only applies to a fresh file

    def clear(self) -> None:
        self.info_path.unlink(missing_ok=True)


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    session: Session
    token: str
    verbose: bool

    def stop(self, now: bool = False) -> None:
        # Graceful by default: a turn in flight is allowed to finish, because
        # there is nothing to abort it with and a half turn is not resumable.
        if not now:
            self.session.wait_idle(120)
        self.shutdown()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"  # no chunking; SSE ends when the socket does
    server_version = "miniagent/0.1"

    # -- plumbing ----------------------------------------------------------- #

    def log_message(self, fmt: str, *args) -> None:
        if getattr(self.server, "verbose", False):
            super().log_message(fmt, *args)

    @property
    def session(self) -> Session:
        return self.server.session  # type: ignore[attr-defined]

    def _json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _no_content(self) -> None:
        self.send_response(204)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _authorized(self) -> bool:
        header = self.headers.get("Authorization") or ""
        return secrets.compare_digest(header, f"Bearer {self.server.token}")  # type: ignore[attr-defined]

    def _body(self) -> dict | None:
        """Read a JSON object body; `None` means the client sent junk."""
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return None
        if length <= 0:
            return {}
        if length > 1_000_000:
            return None
        raw = self.rfile.read(length)
        try:
            body = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            return None
        return body if isinstance(body, dict) else None

    def _result(self, result: dict, ok_code: int = 202) -> None:
        if result.get("ok"):
            self._json(ok_code, result)
        elif result.get("reason") in {"busy", "held"}:
            self._json(409, result)
        elif result.get("reason") == "shutting_down":
            self._json(503, result)
        else:
            self._json(400, result)

    def _lease_lost(self) -> None:
        self._json(401, {
            "error": "lease_lost",
            "message": "another client controls this session",
        })

    # -- GET ---------------------------------------------------------------- #

    def do_GET(self) -> None:  # noqa: N802 - http.server naming
        if not self._authorized():
            return self._json(401, {"error": "unauthorized"})
        url = urlsplit(self.path)
        if url.path == "/status":
            return self._json(200, self.session.status())
        if url.path == "/events":
            return self._events(url.query)
        self._json(404, {"error": "not_found", "path": url.path})

    def _events(self, query: str) -> None:
        try:
            since = int(parse_qs(query).get("since", ["0"])[0])
        except ValueError:
            since = 0
        sub = self.session.subscribe(since)

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        try:
            self.wfile.write(b": connected\n\n")
            self.wfile.flush()
            while not sub.closed:
                records, gap = sub.next_batch(HEARTBEAT)
                if gap is not None:
                    record = {"seq": gap, "event": "gap"}
                    self.wfile.write(
                        f"event: gap\ndata: {json.dumps(record)}\n\n".encode("utf-8")
                    )
                for record in records:
                    self.wfile.write(
                        f"event: {record['event']}\ndata: {json.dumps(record)}\n\n"
                        .encode("utf-8")
                    )
                if not records and gap is None:
                    self.wfile.write(b": hb\n\n")  # also detects a dead peer
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            self.session.unsubscribe(sub)
            self.close_connection = True

    # -- POST --------------------------------------------------------------- #

    def do_POST(self) -> None:  # noqa: N802
        if not self._authorized():
            return self._json(401, {"error": "unauthorized"})

        url = urlsplit(self.path)
        body = self._body()
        if body is None:
            return self._json(400, {"error": "bad_json"})

        route = url.path
        if route in {"/attach", "/takeover"}:
            client = str(body.get("client") or "unknown")[:120]
            host = str(body.get("host") or "unknown")[:120]
            method = self.session.attach if route == "/attach" else self.session.takeover
            return self._result(method(client, host), ok_code=200)

        lease = body.get("lease")

        if route == "/detach":
            if not self.session.verify(lease):
                return self._lease_lost()
            self.session.release(lease)
            return self._no_content()

        if route == "/ping":
            if not self.session.verify(lease):
                return self._lease_lost()
            return self._json(200, {"ok": True, "busy": self.session.busy})

        if route == "/turn":
            if not self.session.verify(lease):
                return self._lease_lost()
            return self._result(self.session.start_turn(str(body.get("text") or "")))

        if route == "/reset":
            if not self.session.verify(lease):
                return self._lease_lost()
            return self._result(self.session.reset())

        if route == "/shutdown":
            # Token-only on purpose: operator commands must not depend on a
            # lease that may be orphaned by a client that went to sleep.
            now = parse_qs(url.query).get("now", ["0"])[0] not in {"", "0", "false"}
            self.session.begin_shutdown()
            self._json(202, {
                "mode": "now" if now else "graceful",
                "turn_active": self.session.busy,
            })
            self.wfile.flush()
            threading.Thread(
                target=self.server.stop, args=(now,), name="stop", daemon=True
            ).start()
            return

        self._json(404, {"error": "not_found", "path": route})


def make_server(
    session: Session,
    host: str = "127.0.0.1",
    port: int = 8765,
    token: str | None = None,
    verbose: bool = False,
) -> _Server:
    httpd = _Server((host, port), Handler)
    httpd.session = session
    httpd.token = token or secrets.token_urlsafe(32)
    httpd.verbose = verbose
    return httpd


# --------------------------------------------------------------------------- #
# Process lifecycle
# --------------------------------------------------------------------------- #


def build_agent(args) -> object:
    """Assemble the agent from files, the same way main.py does."""
    prompt_path = Path(args.system_prompt) if args.system_prompt else DEFAULT_PROMPT_PATH
    tools_path = Path(args.tools) if args.tools else DEFAULT_TOOLS_PATH
    system_prompt = load_text(prompt_path)
    toolbox = build_toolbox(Path(args.workspace), tools_path)

    if args.provider == "ollama":
        base_url = args.base_url or OLLAMA_BASE_URL
        model = args.model or OLLAMA_MODEL
        api_key = args.api_key or "ollama"
    else:
        base_url = args.base_url or "https://api.openai.com/v1"
        model = args.model or "gpt-4o-mini"
        api_key = args.api_key
        if not api_key:
            raise ConfigError("no API key: set OPENAI_API_KEY or pass --api-key")

    return Agent(
        provider=OpenAICompatibleProvider(
            model=model, api_key=api_key, base_url=base_url, stream=not args.no_stream
        ),
        toolbox=toolbox,
        system_prompt=system_prompt,
        max_steps=args.max_steps,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m miniagent.server",
        description="Run the agent as a server with one long-lived session.",
    )
    parser.add_argument("--host", default="127.0.0.1",
                        help="Bind address (default 127.0.0.1; no TLS, so keep it local)")
    parser.add_argument("--port", type=int, default=8765, help="0 picks a free port")
    parser.add_argument("--runtime-dir", default=None)
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
    parser.add_argument("--verbose", action="store_true", help="Log every request")
    args = parser.parse_args(argv)

    runtime = Runtime(args.runtime_dir or RUNTIME_DIR)
    try:
        lock = runtime.claim()
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    try:
        agent = build_agent(args)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        lock.close()
        return 2

    session = Session(agent)
    try:
        httpd = make_server(session, args.host, args.port, verbose=args.verbose)
    except OSError as exc:
        print(f"error: cannot bind {args.host}:{args.port}: {exc}", file=sys.stderr)
        lock.close()
        return 1

    host, port = httpd.server_address[:2]
    runtime.write_info(
        host=host,
        port=port,
        token=httpd.token,
        pid=os.getpid(),
        model=getattr(agent.provider, "model", None),
        started_at=session.started_at,
        version=1,
    )
    if host not in {"127.0.0.1", "localhost", "::1"}:
        print(f"warning: bound to {host} without TLS; only do this behind a "
              f"tunnel or a trusted network", file=sys.stderr)
    print(f"ready: http://{host}:{port}  pid {os.getpid()}", flush=True)

    stopping = {"count": 0}

    def on_signal(signum, frame):  # noqa: ARG001
        stopping["count"] += 1
        # A second Ctrl-C means "now": waiting for a slow model is a fair
        # request, but it must also be possible to give up on it.
        now = stopping["count"] > 1
        print(f"{'stopping' if not now else 'stopping now'} (signal {signum})",
              file=sys.stderr, flush=True)
        threading.Thread(target=httpd.stop, args=(now,), daemon=True).start()

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    try:
        httpd.serve_forever(poll_interval=0.2)
    finally:
        session.close()
        httpd.server_close()
        runtime.clear()
        lock.close()  # releases the flock, so the next start can claim it
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
