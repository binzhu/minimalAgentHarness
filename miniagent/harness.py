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

from dataclasses import dataclass, field
from typing import Any, Callable

from .providers import AssistantMessage, Provider, ProviderError
from .tools import Toolbox

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

    # -- the loop ----------------------------------------------------------- #

    def run(self, task: str) -> AgentResult:
        # The conversation *is* the agent's memory. Everything it knows about
        # the task lives in this list, which is why it is sent in full each turn.
        self.messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": task},
        ]
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
