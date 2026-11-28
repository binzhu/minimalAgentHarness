"""The long-lived session: one conversation that outlives every terminal.

The server process owns an `Agent`. Its `messages` list *is* the session —
there is no persistence layer here, so "the conversation" and "the transcript
in memory" are the same object, and it dies with the process. Clients are
windows onto it, not owners of it.

Only three things live here:

* **ownership** — a lease naming the one client allowed to drive,
* **turns** — run one at a time, and roll back a turn that produced no answer,
* **events** — a sequence number, a replay ring, and fan-out to subscribers.

Deliberately absent: aborting a turn. A turn is seconds long and always ends
with a valid transcript, which is the property that keeps the next request
acceptable to the model server. A takeover during a turn is therefore *refused*
rather than granted, and shutdown waits for the turn instead of killing it.
"""

from __future__ import annotations

import hmac
import secrets
import threading
import time
from collections import deque
from typing import Any

from .harness import Agent

# Seconds without a ping before a lease is considered abandoned. Four client
# heartbeats: long enough to survive a slow request, short enough that a
# laptop that slept is not locked out for long.
#
# TODO: revisit together with observer clients. The end state is a lease per
# event stream plus a cheap liveness probe, which makes read-only watchers free.
LEASE_TTL = 60.0

# Events kept for replay. A reattaching client asks for everything after the
# sequence number it last saw; if that point has aged out it gets a `gap`.
RING_SIZE = 2000

# Per-client backlog before we start dropping the oldest events. A client that
# stops reading must never stall the agent.
CLIENT_BACKLOG = 500


class Subscriber:
    """One connected client's queue of events.

    Bounded on purpose. On overflow the oldest event is dropped and the client
    is told where the replay resumes, so a slow terminal degrades to "you
    missed a burst" instead of to memory growth or a stuck turn.
    """

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._items: deque[dict] = deque()
        self._gap_from: int | None = None
        self.closed = False

    def push(self, record: dict, gap_from: int) -> None:
        with self._cond:
            if self.closed:
                return
            if len(self._items) >= CLIENT_BACKLOG:
                self._items.popleft()
                self._gap_from = gap_from
            self._items.append(record)
            self._cond.notify()

    def close(self) -> None:
        with self._cond:
            self.closed = True
            self._cond.notify_all()

    def next_batch(self, timeout: float) -> tuple[list[dict], int | None]:
        """Block up to `timeout`; return the queued records and a gap marker."""
        with self._cond:
            if not self._items and self._gap_from is None and not self.closed:
                self._cond.wait(timeout)
            items = list(self._items)
            self._items.clear()
            gap, self._gap_from = self._gap_from, None
            return items, gap


class Session:
    """The conversation, its lease, and the clients watching it."""

    def __init__(self, agent: Agent, lease_ttl: float = LEASE_TTL) -> None:
        self.agent = agent
        self.lease_ttl = lease_ttl
        self.started_at = time.time()

        self._lock = threading.RLock()
        self._idle = threading.Condition(self._lock)

        self._seq = 0
        self._ring: deque[dict] = deque(maxlen=RING_SIZE)
        self._subs: set[Subscriber] = set()

        self._lease: str | None = None
        self._lease_seen = 0.0
        self._holder: dict | None = None

        self._busy = False
        self._turn_seq = 0
        self._shutting_down = False

        # The observability seam already existed on Agent; the session is simply
        # another consumer of it, and turns it into the event stream.
        agent.on_event = self._on_agent_event

    # -- events ------------------------------------------------------------- #

    def _on_agent_event(self, event: str, payload: dict[str, Any]) -> None:
        self.emit(event, **payload)

    def emit(self, event: str, **payload: Any) -> None:
        with self._lock:
            self._seq += 1
            record = {"seq": self._seq, "event": event, **payload}
            self._ring.append(record)
            oldest = self._ring[0]["seq"]
            subs = list(self._subs)
        for sub in subs:
            sub.push(record, oldest)

    def subscribe(self, since: int = 0) -> Subscriber:
        """A new event consumer, starting just after `since`."""
        sub = Subscriber()
        with self._lock:
            oldest = self._ring[0]["seq"] if self._ring else 0
            replay = [record for record in self._ring if record["seq"] > since]
            self._subs.add(sub)
        for record in replay:
            sub.push(record, oldest)
        return sub

    def unsubscribe(self, sub: Subscriber) -> None:
        with self._lock:
            self._subs.discard(sub)

    def close(self) -> None:
        """Tell every subscriber the process is going away, then hang up."""
        self._shutting_down = True
        self.emit("shutdown", message="server stopped; session context discarded")
        with self._lock:
            subs = list(self._subs)
            self._subs.clear()
        for sub in subs:
            sub.close()

    # -- ownership ---------------------------------------------------------- #

    def _lease_live(self) -> bool:
        return (
            self._lease is not None
            and (time.monotonic() - self._lease_seen) < self.lease_ttl
        )

    def _holder_info(self) -> dict | None:
        if self._holder is None:
            return None
        info = dict(self._holder)
        info["state"] = "working" if self._busy else "idle"
        info["seen_ago"] = round(time.monotonic() - self._lease_seen, 1)
        return info

    def _grant(self, client: str, host: str) -> dict:
        self._lease = secrets.token_hex(16)
        self._lease_seen = time.monotonic()
        self._holder = {"client": client, "host": host, "since": time.time()}
        # Replaying from the start of the current turn lets a client that
        # reattaches mid-turn see what the agent is doing right now.
        since = self._turn_seq if self._busy else self._seq
        return {"ok": True, "lease": self._lease, "since": since, "busy": self._busy}

    def attach(self, client: str, host: str) -> dict:
        with self._lock:
            if self._shutting_down:
                return {"ok": False, "reason": "shutting_down",
                        "message": "server is shutting down"}
            if self._lease_live():
                return {"ok": False, "reason": "held", "holder": self._holder_info()}
            return self._grant(client, host)

    def takeover(self, client: str, host: str) -> dict:
        """Ask for the session. Granted only while nothing is running.

        A turn is never interrupted: half a turn is not a conversation state
        anyone can resume from, and the model rejects the transcript.
        """
        with self._lock:
            if self._shutting_down:
                return {"ok": False, "reason": "shutting_down",
                        "message": "server is shutting down"}
            if self._lease_live():
                if self._busy:
                    return {
                        "ok": False,
                        "reason": "busy",
                        "message": "turn in progress, please wait",
                        "holder": self._holder_info(),
                    }
                self._detach_all("taken_over")
            return self._grant(client, host)

    def _detach_all(self, reason: str) -> None:
        self.emit("detached", reason=reason)
        with self._lock:
            subs = list(self._subs)
            self._subs.clear()
        for sub in subs:
            sub.close()

    def verify(self, lease: str | None) -> bool:
        """Is this the live lease? Also counts as a sign of life."""
        with self._lock:
            if not self._lease_live() or not lease or not self._lease:
                return False
            if not hmac.compare_digest(lease, self._lease):
                return False
            self._lease_seen = time.monotonic()
            return True

    def release(self, lease: str | None) -> None:
        with self._lock:
            if self._lease and lease and hmac.compare_digest(lease, self._lease):
                self._lease = None
                self._holder = None

    # -- turns -------------------------------------------------------------- #

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._busy

    def start_turn(self, text: str) -> dict:
        with self._lock:
            if self._busy:
                return {"ok": False, "reason": "busy",
                        "message": "turn in progress, please wait"}
            if not text.strip():
                return {"ok": False, "reason": "empty", "message": "empty message"}
            self._busy = True
            self._turn_seq = self._seq
        threading.Thread(target=self._turn, args=(text,), name="turn", daemon=True).start()
        return {"ok": True, "seq": self._turn_seq}

    def _turn(self, text: str) -> None:
        turn_start = len(self.agent.messages)
        ok = True
        stop_reason = "final_answer"
        steps = 0
        error: str | None = None
        try:
            result = self.agent.run(text)
            steps = result.steps
            stop_reason = result.stop_reason
            ok = result.stop_reason == "final_answer"
            error = result.error
        except Exception as exc:  # noqa: BLE001 - a harness bug must not kill the server
            stop_reason = "exception"
            error = f"{type(exc).__name__}: {exc}"
            ok = False
            self.emit("error", message=error)

        if not ok:
            # Rollback invariant: a turn that did not end in an answer is
            # removed whole. Popping only the trailing user message is not
            # enough — a failed multi-step turn can end on tool results, and
            # those belong to a question that is no longer being asked.
            del self.agent.messages[max(turn_start, 0):]

        with self._lock:
            self._busy = False
            self._idle.notify_all()
        self.emit(
            "turn_end",
            ok=ok,
            stop_reason=stop_reason,
            steps=steps,
            messages=len(self.agent.messages),
            error=error,
        )

    def reset(self) -> dict:
        with self._lock:
            if self._busy:
                return {"ok": False, "reason": "busy",
                        "message": "turn in progress, please wait"}
            self.agent.messages.clear()
        self.emit("reset", messages=0)
        return {"ok": True}

    def wait_idle(self, timeout: float | None = None) -> bool:
        with self._idle:
            if not self._busy:
                return True
            return self._idle.wait_for(lambda: not self._busy, timeout)

    # -- reporting ---------------------------------------------------------- #

    def begin_shutdown(self) -> None:
        with self._lock:
            self._shutting_down = True

    def status(self) -> dict:
        with self._lock:
            report = self.agent.context_report()
            return {
                "state": "shutting_down" if self._shutting_down else "running",
                "busy": self._busy,
                "uptime": round(time.time() - self.started_at, 1),
                "messages": len(self.agent.messages),
                "model": getattr(self.agent.provider, "model", None),
                "holder": self._holder_info(),
                "context": {
                    "chars": report.chars,
                    "tokens": report.tokens,
                    "sections": [
                        {"label": section.label, "count": section.count,
                         "chars": section.chars, "tokens": section.tokens}
                        for section in report.sections
                    ],
                },
            }
