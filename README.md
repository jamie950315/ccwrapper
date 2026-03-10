# Claude Code OpenAI Wrapper

Drop-in OpenAI API server backed by Claude. Point any OpenAI-compatible client at this wrapper and it talks to Claude via the official [Claude Agent SDK](https://docs.anthropic.com/en/docs/claude-code/sdk).

## Why

Many tools — Open WebUI, Continue, Cursor, custom apps — speak the OpenAI `/v1/chat/completions` protocol. This wrapper translates that protocol into Claude Agent SDK calls, so you can use Claude everywhere OpenAI is expected without changing client code.

## Quick Start

```bash
git clone https://github.com/jamie950315/ccwrapper.git
cd ccwrapper
poetry install

# Authenticate (pick one)
export ANTHROPIC_API_KEY=sk-ant-...
# or: claude auth login

# Run
poetry run uvicorn src.main:app --port 8000
```

Test it:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/v1/models
```

## Usage

### OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="any")

response = client.chat.completions.create(
    model="claude-sonnet-4-6",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

### Streaming

```python
stream = client.chat.completions.create(
    model="claude-sonnet-4-6",
    messages=[{"role": "user", "content": "Explain recursion"}],
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### Enable Tools

Tools (file read/write, bash, etc.) are **disabled by default** for speed. Enable them per-request:

```python
response = client.chat.completions.create(
    model="claude-sonnet-4-6",
    messages=[{"role": "user", "content": "List files in the current directory"}],
    extra_body={"enable_tools": True},
)
```

### Sessions

Maintain conversation context across requests with `session_id`:

```python
# First message
client.chat.completions.create(
    model="claude-sonnet-4-6",
    messages=[{"role": "user", "content": "My name is Alice."}],
    extra_body={"session_id": "chat-1"},
)

# Follow-up — Claude remembers the context
client.chat.completions.create(
    model="claude-sonnet-4-6",
    messages=[{"role": "user", "content": "What's my name?"}],
    extra_body={"session_id": "chat-1"},
)
```

Sessions expire after 1 hour of inactivity. Each session holds up to 200 messages. Expired sessions are cleaned up automatically every 5 minutes.

### curl

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-ant-..." \  # only required if API_KEY is set in .env
  -d '{
    "model": "claude-sonnet-4-6",
    "messages": [{"role": "user", "content": "What is 2+2?"}]
  }'
```

### Anthropic Messages API

A native `/v1/messages` endpoint is also available for clients that speak the Anthropic protocol directly.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/chat/completions` | OpenAI-compatible chat (streaming & non-streaming) |
| `POST` | `/v1/messages` | Anthropic-compatible messages |
| `GET` | `/v1/models` | List available models |
| `GET` | `/v1/sessions` | List active sessions |
| `GET` | `/v1/sessions/{id}` | Get session details |
| `DELETE` | `/v1/sessions/{id}` | Delete a session |
| `GET` | `/v1/sessions/stats` | Session statistics |
| `GET` | `/v1/tools` | List available tools with metadata |
| `GET` | `/v1/tools/config` | Get tool configuration |
| `POST` | `/v1/tools/config` | Update tool configuration |
| `GET` | `/v1/auth/status` | Authentication info |
| `POST` | `/v1/compatibility` | OpenAI compatibility report |
| `POST` | `/v1/debug/request` | Request inspection |
| `GET` | `/health` | Health check (reports SDK readiness + auth validity) |
| `GET` | `/version` | Server version |
| `GET` | `/` | Interactive API explorer |

## Supported Models

| Model | Notes |
|-------|-------|
| `claude-sonnet-4-6` | **Default.** Best balance of speed and capability. |
| `claude-opus-4-6` | Most capable. Best for agents and complex tasks. |
| `claude-haiku-4-5-20251001` | Fastest and cheapest. |
| `claude-opus-4-5-20250929` | Legacy Opus 4.5 |
| `claude-sonnet-4-5-20250929` | Legacy Sonnet 4.5 |
| `claude-opus-4-1-20250805` | Legacy Opus 4.1 |
| `claude-opus-4-20250514` | Legacy Opus 4.0 |
| `claude-sonnet-4-20250514` | Legacy Sonnet 4.0 |

Claude 3.x models are **not supported** by the Claude Agent SDK.

## Configuration

Copy `.env.example` to `.env` and edit as needed. All settings have sensible defaults.

| Variable | Description | Default |
|----------|-------------|---------|
| `ANTHROPIC_API_KEY` | API key for Claude | — |
| `CLAUDE_AUTH_METHOD` | Force auth method: `cli`, `api_key`, `bedrock`, `vertex` | auto-detect |
| `API_KEY` | Protect wrapper endpoints with Bearer token | — |
| `DEFAULT_MODEL` | Default model for requests | `claude-sonnet-4-6` |
| `PORT` | Server port | `8000` |
| `CLAUDE_CWD` | Working directory for Claude | isolated temp dir |
| `MAX_TIMEOUT` | Request timeout (ms) | `600000` |
| `MAX_CONCURRENT_QUERIES` | Concurrent SDK query limit | `2` |
| `CLIENT_RECYCLE_REQUESTS` | Recycle persistent client after N requests | `200` |
| `MAX_STANDBY_CLIENTS` | Number of pre-warmed standby clients | `2` |
| `RATE_LIMIT_ENABLED` | Enable per-IP rate limiting | `true` |
| `RATE_LIMIT_CHAT_PER_MINUTE` | Chat endpoint rate limit | `10` |
| `CORS_ORIGINS` | Allowed CORS origins | `["*"]` |

### Authentication

The wrapper auto-detects your auth method, or you can set `CLAUDE_AUTH_METHOD` explicitly.

| Method | How |
|--------|-----|
| **API Key** (recommended) | `export ANTHROPIC_API_KEY=sk-ant-...` |
| **CLI Auth** | `claude auth login` (uses your Pro/Max subscription) |
| **AWS Bedrock** | Standard AWS credential chain |
| **Google Vertex AI** | Standard GCP credential chain |

### API Key Protection

Optionally protect the wrapper's endpoints with a Bearer token:

```bash
# Set in environment
export API_KEY=my-secret-key

# Or let the server generate one interactively on startup
poetry run python main.py
```

Clients then include `Authorization: Bearer my-secret-key` in requests.

### Working Directory

By default, Claude runs in an isolated temp directory so it can't access the wrapper's source code. Override with `CLAUDE_CWD`:

```bash
export CLAUDE_CWD=/path/to/your/project
```

## Architecture

```
Client (OpenAI SDK / curl / Open WebUI)
  │
  ▼
FastAPI server (src/main.py)
  │
  ├─ parameter_validator.py   → validate & map OpenAI params
  ├─ message_adapter.py       → convert message formats
  ├─ session_manager.py       → inject conversation history
  │
  ▼
claude_cli.py → Claude Agent SDK
  │
  ├─ Fast path: triple-buffered ClaudeSDKClient (zero-latency swap, 2 standbys)
  └─ Slow path: stateless query() (tools, sessions, multi-turn)
```

Key modules:

- **`claude_cli.py`** — Dual-path SDK wrapper with triple-buffered session isolation. Active client serves the request; up to 2 standby clients are pre-warmed in background via cascade pattern for instant swap. If a standby is mid-preparation when needed, the request waits via `asyncio.Event` rather than cold-starting a duplicate. SIGKILL fallback ensures subprocess cleanup. `asyncio.Lock` enforces exclusive fast-path access; `asyncio.timeout()` guards against subprocess hangs; 30s backpressure on semaphore rejects overload early.
- **`message_adapter.py`** — Converts between OpenAI and Claude formats. Strips thinking blocks, extracts `attempt_completion` results, CJK-aware token estimation.
- **`session_manager.py`** — In-memory session store. 1-hour TTL, 200-message cap, background cleanup, thread-safe.
- **`auth.py`** — Multi-provider auth with auto-detection.
- **`tool_manager.py`** — Tool metadata and per-session allow/deny lists.
- **`rate_limiter.py`** — Per-IP rate limiting via slowapi.
- **`mcp_client.py`** — Optional Model Context Protocol integration.

## Docker

```bash
docker build -t ccwrapper .
docker run -d -p 8000:8000 -e ANTHROPIC_API_KEY=sk-ant-... ccwrapper
```

With Docker Compose:

```bash
docker-compose up -d
```

Mount `~/.claude` for CLI auth, or set `ANTHROPIC_API_KEY` in the environment. Set `CLAUDE_CWD` and mount a volume if you need Claude to access a specific directory.

## Development

```bash
# Install with dev dependencies
poetry install --with dev

# Dev server with hot reload
poetry run uvicorn src.main:app --reload --port 8000

# Tests
poetry run pytest tests/ -v

# Format
poetry run black src tests

# Type check
poetry run mypy src --ignore-missing-imports

# Security scan
poetry run bandit -r src/ -ll -x tests
```

## Current Limitations

- Image content in messages is converted to text placeholders
- OpenAI parameters `temperature`, `top_p`, `max_tokens`, `logit_bias`, `presence_penalty`, `frequency_penalty` are accepted but not mapped
- `n > 1` (multiple completions) not supported
- Function calling / tool_choice not supported (use `enable_tools` instead)

## Terms Compliance

This wrapper requires your own Claude subscription or API access. It does not provide, share, or pool credentials. It is a format translator — converting OpenAI-shaped requests into Claude Agent SDK calls using your own authentication.

- Uses the official Claude Agent SDK
- Each user authenticates individually
- No credential sharing, reselling, or data harvesting

For programmatic/commercial use, `ANTHROPIC_API_KEY` is recommended (explicitly permitted under Anthropic's Commercial Terms). CLI auth via Pro/Max subscription is appropriate for personal use at moderate scale.

See Anthropic's [Usage Policy](https://www.anthropic.com/legal/aup), [Consumer Terms](https://www.anthropic.com/legal/consumer-terms), and [Commercial Terms](https://www.anthropic.com/legal/commercial-terms).