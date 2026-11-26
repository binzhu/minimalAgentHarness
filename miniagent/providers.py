"""Model providers.

A *provider* is the only part of the harness that talks to a model. The harness
never knows whether it is talking to GPT, Llama, or a box on the LAN — it just
calls `complete(messages, tools)` and gets back an `AssistantMessage`.

Messages use the OpenAI chat format, which has become the de-facto shape for
tool calling:

    {"role": "system",    "content": "..."}
    {"role": "user",      "content": "..."}
    {"role": "assistant", "content": "..." | None,
     "tool_calls": [{"id": "...", "type": "function",
                     "function": {"name": "...", "arguments": "<json string>"}}]}
    {"role": "tool", "tool_call_id": "...", "content": "<string result>"}

Streaming is a transport detail, not a different interface: a streaming
provider still returns one finished `AssistantMessage`. It just reports the
fragments on the way there through `on_delta(kind, text)`.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Protocol

DeltaHandler = Callable[[str, str], None]  # (kind: "content"|"reasoning", fragment)


class ProviderError(RuntimeError):
    """Raised when the model backend cannot be reached or returns junk."""


@dataclass
class ToolCall:
    """One tool invocation requested by the model."""

    id: str
    name: str
    arguments: str  # raw JSON string, exactly as the model emitted it


@dataclass
class AssistantMessage:
    """One turn of model output: some text, some tool calls, or both.

    `reasoning` is the separate chain-of-thought channel that thinking models
    (gemma4, gpt-oss, deepseek-r1) emit. It is kept apart from `content` so it is
    never fed back into the transcript as if the model had said it out loud.
    """

    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    reasoning: str | None = None

    def to_message(self) -> dict:
        msg: dict = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": tc.arguments},
                }
                for tc in self.tool_calls
            ]
        return msg


class Provider(Protocol):
    """Anything that can turn a message history into the next assistant turn."""

    def complete(
        self,
        messages: list[dict],
        tools: list[dict],
        on_delta: DeltaHandler | None = None,
    ) -> AssistantMessage: ...


# --------------------------------------------------------------------------- #
# A real model, over plain HTTP (no SDK required)
# --------------------------------------------------------------------------- #


class OpenAICompatibleProvider:
    """Calls any OpenAI-compatible `/chat/completions` endpoint.

    Works with OpenAI, DeepSeek, Groq, Together, vLLM, LM Studio, Ollama
    (`http://localhost:11434/v1`), and friends — only `base_url` changes.
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        timeout: int = 300,
        stream: bool = True,
    ):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        # Generous by default: a 12B model may need to load into VRAM on the
        # first request, and each loop step is a fresh round trip.
        self.timeout = timeout
        self.stream = stream

    # -- public API --------------------------------------------------------- #

    def complete(
        self,
        messages: list[dict],
        tools: list[dict],
        on_delta: DeltaHandler | None = None,
    ) -> AssistantMessage:
        payload: dict = {"model": self.model, "messages": messages}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        if self.stream:
            return self._streamed(payload, on_delta)
        return self._buffered(payload)

    # -- buffered: one response at the end ---------------------------------- #

    def _buffered(self, payload: dict) -> AssistantMessage:
        body = self._post(payload)

        try:
            raw = body["choices"][0]["message"]
        except (KeyError, IndexError) as exc:
            raise ProviderError(f"unexpected response shape: {body}") from exc

        tool_calls = [
            ToolCall(
                id=tc.get("id") or f"call_{i}",
                name=tc["function"]["name"],
                arguments=tc["function"].get("arguments") or "{}",
            )
            for i, tc in enumerate(raw.get("tool_calls") or [])
        ]
        return AssistantMessage(
            content=raw.get("content") or None,
            tool_calls=tool_calls,
            reasoning=raw.get("reasoning") or None,
        )

    # -- streamed: server-sent events, accumulated here --------------------- #

    def _streamed(self, payload: dict, on_delta: DeltaHandler | None) -> AssistantMessage:
        payload["stream"] = True

        text: list[str] = []
        thoughts: list[str] = []
        calls: dict[int, dict] = {}  # index -> {id, name, arguments}

        try:
            with urllib.request.urlopen(self._request(payload), timeout=self.timeout) as response:
                # Each event is a line like `data: {...}`, then a blank line;
                # the stream ends with `data: [DONE]`.
                for raw_line in response:
                    line = raw_line.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:"):].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue  # keep-alive or vendor noise

                    for choice in chunk.get("choices") or []:
                        delta = choice.get("delta") or {}

                        if delta.get("content"):
                            text.append(delta["content"])
                            if on_delta:
                                on_delta("content", delta["content"])
                        if delta.get("reasoning"):
                            thoughts.append(delta["reasoning"])
                            if on_delta:
                                on_delta("reasoning", delta["reasoning"])
                        for fragment in delta.get("tool_calls") or []:
                            self._merge_tool_call(calls, fragment)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:800]
            raise ProviderError(f"HTTP {exc.code} from {self.base_url}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ProviderError(f"stream from {self.base_url} broke: {exc}") from exc

        tool_calls = [
            ToolCall(
                id=slot["id"] or f"call_{index}",
                name=slot["name"],
                arguments=slot["arguments"] or "{}",
            )
            for index, slot in sorted(calls.items())
        ]
        return AssistantMessage(
            content="".join(text) or None,
            tool_calls=tool_calls,
            reasoning="".join(thoughts) or None,
        )

    @staticmethod
    def _merge_tool_call(calls: dict[int, dict], fragment: dict) -> None:
        """Stitch tool-call fragments together.

        A tool call can arrive in pieces: `{"pa`, `th":`, `"."}`. Pieces share an
        `index`, so accumulate per index and parse the JSON only when the turn
        ends — a partial fragment is not valid JSON on its own.
        """
        slot = calls.setdefault(
            fragment.get("index", 0), {"id": None, "name": "", "arguments": ""}
        )
        if fragment.get("id"):
            slot["id"] = fragment["id"]

        function = fragment.get("function") or {}
        name = function.get("name") or ""
        # Names usually arrive whole; some servers repeat them every chunk.
        if name and slot["name"] != name:
            slot["name"] += name
        if function.get("arguments"):
            slot["arguments"] += function["arguments"]

    # -- plumbing ----------------------------------------------------------- #

    def _request(self, payload: dict) -> urllib.request.Request:
        return urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

    def _post(self, payload: dict) -> dict:
        try:
            with urllib.request.urlopen(self._request(payload), timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:800]
            raise ProviderError(f"HTTP {exc.code} from {self.base_url}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise ProviderError(f"could not reach {self.base_url}: {exc}") from exc
