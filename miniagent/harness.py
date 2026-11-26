"""The agent harness and the agent loop.

Terminology, since the two get used interchangeably:

* **Agent loop** — the control flow. Ask the model -> it may request tools ->
  run them -> feed results back -> ask again -> stop when it answers.
* **Harness** — everything *around* the loop that makes it usable: the system
  prompt, the message history, tool dispatch, stopping conditions, error
  handling, and observability (events you can print or log).
* **Model** — the policy. It only proposes; the harness decides what happens.

The loop below is deliberately small. Most production agents are this, plus
retries, streaming, persistence, sub-agents, and cost accounting.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from math import ceil
from typing import Any, Callable

from .providers import AssistantMessage, Provider, ProviderError
from .tools import Toolbox


def estimate_tokens(chars: int) -> int:
    """~4 characters per token, the usual rule of thumb for prose and code.

    There is no tokenizer here on purpose: adding one would pin the harness to
    a single model's vocabulary, and the provider is swappable. So `/context`
    reports an estimate and says so, rather than a number it cannot stand behind.
    """
    return ceil(chars / 4)


@dataclass
class ContextSection:
    """One line of the `/context` report: a kind of content and how big it is."""

    label: str
    count: int  # messages, tool schemas, tool calls — whatever this row counts
    chars: int

    @property
    def tokens(self) -> int:
        return estimate_tokens(self.chars)


@dataclass
class ContextReport:
    """What the *next* request would send, split by where it comes from."""

    sections: list[ContextSection]

    @property
    def chars(self) -> int:
        return sum(section.chars for section in self.sections)

    @property
    def tokens(self) -> int:
        return estimate_tokens(self.chars)


@dataclass
class AgentResult:
    text: str
    messages: list[dict]
    steps: int
    stop_reason: str  # "final_answer" | "max_steps" | "provider_error"
    error: str | None = None


@dataclass
class Agent:
    """A model + a toolbox + a loop, with guards."""

    provider: Provider
    toolbox: Toolbox
    # No default on purpose: the prompt is configuration (config/system_prompt.md),
    # and a built-in copy would silently drift from the file you actually edit.
    system_prompt: str
    max_steps: int = 12
    on_event: Callable[[str, dict[str, Any]], None] | None = None
    # The transcript. It is deliberately *not* reset between run() calls — that
    # is what makes run() one turn of a conversation rather than all of it.
    messages: list[dict] = field(default_factory=list)

    # -- observability ------------------------------------------------------ #

    def _emit(self, event: str, **payload: Any) -> None:
        # `event` rather than `kind` so payload keys like kind="content" are free.
        if self.on_event:
            self.on_event(event, payload)

    def _on_delta(self, kind: str, text: str) -> None:
        """Handed to streaming providers: one call per fragment as it arrives.

        This is display only. The loop still waits for the completed
        AssistantMessage, because half a tool call cannot be executed.
        """
        self._emit("delta", kind=kind, text=text)

    # -- context accounting ------------------------------------------------- #

    def context_report(self) -> ContextReport:
        """Where the next request's context goes, one section at a time.

        Counted in characters, because characters are exact; the token figure
        built on top of them is the estimate. Empty sections stay in the report:
        after a long session, "tool results: 0" is part of the answer.
        """
        sizes = {"system": 0, "user": 0, "assistant": 0, "tool_calls": 0, "tool": 0}
        counts = dict.fromkeys(sizes, 0)

        for message in self.messages:
            role = message.get("role")
            content = message.get("content") or ""
            if role in sizes:
                sizes[role] += len(content)
                counts[role] += 1
            # Tool calls are prompt too: the model needs the name and the raw
            # arguments it sent, even though they are not the message content.
            for call in message.get("tool_calls") or []:
                sizes["tool_calls"] += len(json.dumps(call["function"]))
                counts["tool_calls"] += 1

        # A fresh agent has no system message yet — run() seeds it on the first
        # turn — so report the prompt that is about to be sent.
        if not counts["system"]:
            sizes["system"] = len(self.system_prompt)
            counts["system"] = 1

        # Tool schemas never live in `messages`, but they travel with every
        # request, so they are part of what a turn costs.
        schemas = self.toolbox.schemas()
        return ContextReport([
            ContextSection("system prompt", counts["system"], sizes["system"]),
            ContextSection("tool schemas", len(schemas), len(json.dumps(schemas))),
            ContextSection("user messages", counts["user"], sizes["user"]),
            ContextSection("assistant messages", counts["assistant"], sizes["assistant"]),
            ContextSection("assistant tool calls", counts["tool_calls"], sizes["tool_calls"]),
            ContextSection("tool results", counts["tool"], sizes["tool"]),
        ])

    # -- the loop ----------------------------------------------------------- #

    def run(self, task: str) -> AgentResult:
        # The conversation *is* the agent's memory: the whole list goes to the
        # provider on every turn, so the model can stay stateless. Calling run()
        # again continues this transcript instead of starting over.
        if not self.messages:
            self.messages = [{"role": "system", "content": self.system_prompt}]
        self.messages.append({"role": "user", "content": task})
        tools = self.toolbox.schemas()
        self._emit("start", task=task, tools=self.toolbox.names)

        # Loop hygiene: models (especially small local ones) often repeat the
        # exact call they just made instead of reading the result. Remembering
        # signatures lets the harness answer the repeat for free.
        seen: dict[tuple[str, str], str] = {}
        last_signature: tuple[str, str] | None = None

        for step in range(1, self.max_steps + 1):
            # 1. Ask the model what to do next.
            try:
                reply: AssistantMessage = self.provider.complete(
                    self.messages, tools, on_delta=self._on_delta
                )
            except ProviderError as exc:
                self._emit("error", message=str(exc))
                return AgentResult(
                    text="", messages=self.messages, steps=step,
                    stop_reason="provider_error", error=str(exc),
                )

            self.messages.append(reply.to_message())
            self._emit("assistant", step=step, content=reply.content,
                       reasoning=reply.reasoning,
                       tool_calls=[tc.name for tc in reply.tool_calls])

            # 2. No tool calls means the model considers itself finished.
            if not reply.tool_calls:
                # A thinking model may answer only in the reasoning channel.
                text = reply.content or reply.reasoning or ""
                self._emit("final", step=step, content=text)
                return AgentResult(text, self.messages, step, "final_answer")

            # 3. Run every requested tool and append each result. The tool
            #    result must be tied back to its call id, or the model loses
            #    track of which answer belongs to which question.
            for call in reply.tool_calls:
                signature = (call.name, call.arguments)
                if signature == last_signature and signature in seen:
                    result = (
                        f"ERROR: you just called {call.name} with these exact arguments, "
                        f"and already got this result:\n{seen[signature]}\n"
                        "Do not repeat it. Use the result above and continue."
                    )
                    self._emit("tool_skipped", step=step, name=call.name)
                else:
                    result = self.toolbox.call(call.name, call.arguments)
                    seen[signature] = result
                    last_signature = signature

                self.messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": result}
                )
                self._emit("tool_result", step=step, name=call.name, result=result)

        # 4. Ran out of budget. A harness that cannot stop is a bug, so this is
        #    a normal outcome, not an exception.
        self._emit("stop", reason="max_steps", steps=self.max_steps)
        return AgentResult(
            text="", messages=self.messages, steps=self.max_steps, stop_reason="max_steps"
        )
