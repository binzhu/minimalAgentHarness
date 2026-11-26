# miniagent — a small agent harness you can read in one sitting

A coding agent is not a model with superpowers. It is a **loop** that lets a
model ask for actions, plus a **harness** that performs them and decides when
to stop. This project is that loop, in about 1,000 lines of dependency-free
Python, with streaming over plain `http.client` and no SDKs.

```
miniagent/
├── providers.py   # the only code that talks to a model (OpenAI-compatible)
├── tools.py       # the agent's hands: Python implementations + dispatch
├── harness.py     # the loop, the message history, the guards
├── config.py      # loads prompts and tool schemas from files
config/
├── system_prompt.md   # the policy text sent as the system message
├── tools.json         # the schemas the model reads to choose a tool
main.py            # CLI: a conversation with the agent
```

---

## 1. The three pieces

| Piece | Role | In an LLM analogy |
| --- | --- | --- |
| **Model** | proposes the next message or tool call | the policy |
| **Tools** | actually read files, write files, run code | the actuators |
| **Harness** | runs the loop, keeps memory, enforces limits | the runtime / OS |

The model never executes anything. It emits JSON saying *"call `write_file`
with these arguments"* and the harness decides whether and how to do that.
That separation is the whole trick: it is also the security boundary.

---

## 2. The agent loop

```
            ┌──────────────────────────────────────────┐
 task ────► │ messages = [system, user(task)]          │
            └───────────────────┬──────────────────────┘
                                ▼
                    ┌───────────────────┐
                    │ model.complete()  │◄──────────────┐
                    └─────────┬─────────┘               │
                              ▼                         │
                 ┌────────────────────────┐             │
                 │ tool calls requested?  │             │
                 └───────┬────────┬───────┘             │
                      no │        │ yes                 │
                         ▼        ▼                     │
                    ┌────────┐  ┌──────────────────┐    │
                    │ answer │  │ run each tool    │    │
                    │  done  │  │ append results   │────┘
                    └────────┘  └──────────────────┘
```

Five details that make it actually work:

1. **The transcript is the memory.** Every turn resends the whole message list:
   the task, each assistant message, each tool call, each tool result. The model
   is stateless; continuity lives in the harness.
2. **Tool results must be linked by id.** A `tool` message carries
   `tool_call_id`. Drop it and the model can't tell which answer belongs to
   which request.
3. **Errors are data.** A failed tool returns `"ERROR: ..."` as text rather than
   raising. The model reads it, fixes its arguments, and continues — that is
   where most of an agent's apparent "intelligence" comes from.
4. **It must be able to stop.** A step budget (`max_steps`) is not optional.
   Loops that can't stop are how agents spend your budget forever.
5. **You need to see it.** `on_event` reports `start / assistant / delta /
   tool_result / tool_skipped / final / stop`. Without this trace you are
   debugging blind.
6. **Repeated calls are free to absorb.** Small models often re-issue the call
   they just made instead of reading its result. The harness remembers each
   `(tool, arguments)` signature and answers an immediate repeat with the cached
   result plus a nudge, instead of running the tool again.

Two channels, not one: thinking models (gemma4, gpt-oss, deepseek-r1) return
chain-of-thought in a separate `reasoning` field. The harness shows it as
`thinking` and keeps it out of the transcript, so the model is never fed its own
internal monologue back as if it had said it out loud.

---

## 3. Run it

No dependencies. The CLI is a conversation: start it, then keep typing. The
agent keeps the transcript and resends it on every turn, so it remembers what
it just did — including the files it read.

```bash
python3 main.py
```

```
you>      list this folder
...
you>      now summarise what those files do
```

`quit`, `exit`, or Ctrl-D ends it. A task given on the command line becomes the
first message, which is the one-shot form — piped stdin ends the loop after it:

```bash
python3 main.py "list this folder and summarise what it does"
```

With a real model (any OpenAI-compatible endpoint):

```bash
export OPENAI_API_KEY=sk-...
python3 main.py "write a prime checker in check.py and test it with 97"
```

### Checking the context budget

Every turn resends the whole transcript, so it grows, while the model's window
does not. `/context` shows what the next request would send:

```
you      /context

context   6 messages, 14,203 chars ~ 3,551 tokens (estimate)

  system prompt             1     1,204 chars  ~   301 tok
  tool schemas              4     3,905 chars  ~   977 tok
  user messages             3        86 chars  ~    22 tok
  assistant messages        2       640 chars  ~   160 tok
  assistant tool calls      2       112 chars  ~    28 tok
  tool results              2     8,256 chars  ~ 2,064 tok
  total                           14,203 chars  ~ 3,551 tok
```

The answer is local: no request is made, no tokens are spent, and it works with
the server down. Sizes are exact character counts; the token column is
`chars / 4`, because this harness has no tokenizer and the provider is
swappable — good enough to see *which* section is eating the window, not exact.
Tool results dominate most long sessions, so that is the row to watch.

### Default: the Ollama box on the LAN

`--provider ollama` is the default and points at `http://10.20.30.99:11434/v1`
with model `gemma4:12b`. Ollama serves an OpenAI-compatible API under `/v1`,
which is why the same provider class drives it and OpenAI alike:

```bash
python3 main.py "list this folder and summarise what it does"
python3 main.py --model qwen2.5-coder:14b "write fizzbuzz.py and run it"
```

Override with `--base-url`, `--model`, `--api-key`, or the `MINIAGENT_BASE_URL` /
`MINIAGENT_MODEL` / `MINIAGENT_API_KEY` env vars. Ollama ignores the key, but the
`Authorization` header must be present.

A model only works here if it was loaded with tool support. Check what the
server reports:

```bash
curl -s http://10.20.30.99:11434/api/tags | grep -o '"capabilities":[^]]*]'
```

`gemma4:12b` lists `completion, vision, audio, tools, thinking`; the `tools`
entry is the one the harness needs. `llama2:7b` lists only `completion` and will
never call a tool, however the prompt is worded. Thinking models also return a
separate `reasoning` channel — the harness prints it as `thinking` and keeps it
out of the transcript.

### Other endpoints

Only the URL changes for any OpenAI-compatible server:

```bash
# LM Studio: --provider openai --base-url http://localhost:1234/v1 --api-key lm-studio
# DeepSeek:   --provider openai --base-url https://api.deepseek.com/v1 --model deepseek-chat
```

Useful flags: `--max-steps` (budget), `--workspace` (tool sandbox root),
`--no-stream` (buffer each turn instead of streaming it), `--system-prompt` and
`--tools` (override the config files below).

### Configuration files

Two things are configuration, not logic, and live in `config/`:

| File | What it is | Why it is a file |
| --- | --- | --- |
| `system_prompt.md` | the policy text sent as the system message | you tune it constantly; diffs should be readable |
| `tools.json` | name, description, parameters for each tool | this is the JSON the model reads when choosing |

Both are loaded at startup and passed explicitly into `Agent` / `build_toolbox`.
A missing, empty, or malformed file is a **startup error** (`ConfigError`, exit
2), never a silent fall back to built-in defaults — otherwise you spend an hour
wondering why your prompt edits do nothing.

```bash
python3 main.py --system-prompt /tmp/strict.md "..."
python3 main.py --tools /tmp/extra_tools.json "..."
```

The prompt and the tool list are separate for a reason: `tools.json` is
capability (what the model *can* do), `system_prompt.md` is policy (what it
*should* do). The tool descriptions are prompt text too — they are read at the
moment of choosing, which is why they often steer behavior more than the system
message does.

---

## 4. Add a tool

Two edits in two files: the schema the model reads, and the code that runs.

```json
// config/tools.json — append to "tools"
{
  "name": "word_count",
  "description": "Count words in a text file.",
  "parameters": {
    "type": "object",
    "properties": {"path": {"type": "string", "description": "File path."}},
    "required": ["path"]
  }
}
```

```python
# tools.py — inside _implementations()
def word_count(path: str) -> str:
    return str(len(box._resolve(path).read_text().split()))

return {
    "list_dir": list_dir,
    "read_file": read_file,
    "write_file": write_file,
    "run_python": run_python,
    "word_count": word_count,      # <- the name is the join key
}
```

The name is the link between the two halves. A schema with no implementation
stops the run at startup with a message naming the missing tool, so you cannot
ship a tool the model can see but nobody can run. Schemas are sent to the model
every turn, so nothing else needs registering.

Write the description for a competent stranger who cannot see your code — the
model is exactly that stranger.

---

## 5. Streaming

By default the provider streams: it sends `"stream": true` and reads the
response as server-sent events, one `data: {...}` line per chunk and a final
`data: [DONE]`. Each chunk carries a `delta` (only the new fragment) instead of
a complete `message`.

Measured on the LAN box, for a two-word answer: **first fragment at 5.17s, whole
turn done at 6.22s, 65 fragments.** Same work, but you watch it happen instead
of staring at a blank prompt — and on a long answer the gap is much wider.

Two things make this harder than it looks:

- **Tool-call arguments arrive in pieces.** You get `{"pa`, then `th":`, then
  `"."}` — each invalid JSON alone. Fragments share an `index`, so the provider
  accumulates per index and parses only when the turn ends. See
  `_merge_tool_call` in `providers.py`.
- **The loop must not act early.** Half a tool call cannot be executed, so
  streaming is display only: `complete()` still returns one finished
  `AssistantMessage`. That is why `harness.py` needed no changes beyond
  forwarding `delta` events, and why buffering vs streaming is a `--no-stream`
  flag rather than a second code path through the agent.

Switch it off to compare:

```bash
python3 main.py --no-stream "say what this project is"   # one blob at the end
```

---

## 6. What this deliberately leaves out

You now have the skeleton. Production harnesses add:

- **Retries / backoff** around provider errors, and per-call cost accounting
- **Context management** — summarizing or truncating history when it grows
- **Sub-agents** — a tool whose implementation is *another* `Agent.run(...)`
- **Approval gates** for destructive tools (`write_file` outside a scratch dir)
- **Real sandboxing** — `run_python` here runs in your workspace with a timeout,
  not in a container. Don't hand it untrusted models or untrusted code.

The interesting engineering is almost entirely in item 3: an agent is a
context-management problem wearing a tool-use hat.
