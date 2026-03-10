@localClaude.md

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

OpenAI API-compatible wrapper for Claude Code. Translates OpenAI `/v1/chat/completions` requests into Claude Agent SDK calls, enabling any OpenAI client library to use Claude models. Also exposes a native Anthropic `/v1/messages` endpoint. Built with FastAPI, managed with Poetry.

## Commands

### Development
```bash
poetry install                          # Install all deps (add --with dev for dev tools)
poetry run uvicorn src.main:app --reload --port 8000  # Dev server with hot reload
poetry run python main.py               # Production server (no reload)
```

### Testing
```bash
poetry run pytest tests/ -v             # Full test suite
poetry run pytest tests/test_foo.py -v  # Single test file
poetry run pytest tests/ -v --cov=src --cov-report=term-missing  # With coverage
```

### Linting & Quality
```bash
poetry run black --check src tests      # Check formatting (CI uses this)
poetry run black src tests              # Auto-format
poetry run mypy src --ignore-missing-imports  # Type checking (non-blocking in CI)
poetry run bandit -r src/ -ll -x tests  # Security scan
```

### Docker
```bash
docker build -t claude-wrapper:latest .
docker-compose up -d
```

## Architecture

**Entry point:** `src/main.py` — FastAPI app defining all endpoints. This is a large file (~79KB) containing route handlers, streaming logic, and the interactive landing page.

**Request flow (OpenAI-compatible):**
1. Client sends OpenAI-format request to `POST /v1/chat/completions`
2. `parameter_validator.py` validates and maps OpenAI params to Claude equivalents
3. `message_adapter.py` converts OpenAI message format → Claude prompt format
4. `session_manager.py` injects conversation history if `session_id` is present
5. `claude_cli.py` calls Claude Agent SDK — dual-path architecture:
   - **Fast path**: Triple-buffered persistent `ClaudeSDKClient` with zero-latency swap (for simple requests: no tools, no sessions, max_turns=1). Up to 2 standby clients pre-warmed in background via cascade pattern.
   - **Slow path**: Stateless `query()` call (for requests needing custom tools, sessions, or multi-turn)
   - Concurrent requests: fast path is exclusive (one at a time); overflow goes to slow path automatically
6. Response converted back to OpenAI format (streaming SSE or JSON)

**Key modules:**
- `auth.py` — Multi-provider auth (CLI, API key, Bedrock, Vertex). Auto-detects method or uses `CLAUDE_AUTH_METHOD` env var.
- `claude_cli.py` — Dual-path SDK wrapper with triple-buffered session isolation. Active client serves the current request; up to 2 standby clients are pre-warmed in background via cascade pattern (each task creates ONE client then spawns the next, avoiding event-loop blocking). Swap is instant (~0s); if a standby is still warming when needed, the request waits via `asyncio.Event` instead of cold-starting a duplicate. After each request the used client is killed (SIGKILL fallback for SDK disconnect bugs). Fast path exclusive access enforced via `asyncio.Lock` (`_fast_path_lock`); concurrent requests that can't acquire the lock fall back to slow path. `asyncio.timeout()` guards against subprocess hangs (replaces manual `time.monotonic()` deadline checks). Semaphore acquire has 30s backpressure timeout — rejects with "server busy" instead of queueing indefinitely. Manages working directory, max_turns, tool config, and automatic client recycling after N requests (configurable via `CLIENT_RECYCLE_REQUESTS` env, default 200). Standby pool size configurable via `MAX_STANDBY_CLIENTS` (default 2).
- `message_adapter.py` — Bidirectional format conversion. Strips thinking blocks, handles `attempt_completion`, CJK-aware token estimation (~1.5 chars/token for CJK, ~4 chars/token for ASCII). Single source of truth for `estimate_tokens()` (used by both `message_adapter.py` and `claude_cli.py`).
- `session_manager.py` — In-memory conversation history with 1-hour TTL, background cleanup every 5 min, lock-based thread safety, 200-message cap per session.
- `tool_manager.py` — Tool metadata and allow/deny lists. Tools disabled by default for speed; enabled per-request via `enable_tools` parameter.
- `models.py` — Pydantic request/response models for both OpenAI and Anthropic formats.
- `constants.py` — Single source of truth for model lists, tool names, rate limits, defaults.
- `rate_limiter.py` — Per-IP rate limiting via slowapi, configurable per endpoint through env vars.
- `mcp_client.py` — Optional Model Context Protocol client for external tool servers.

## Configuration

All config via environment variables (see `.env.example`). Key ones:
- `ANTHROPIC_API_KEY` — Direct API key auth
- `API_KEY` — Protects wrapper endpoints with Bearer token
- `CLAUDE_CWD` — Working directory for Claude (default: isolated temp dir)
- `DEFAULT_MODEL` — Default model (default: `claude-sonnet-4-6`)
- `MAX_CONCURRENT_QUERIES` — Concurrent SDK queries limit (default: 2)
- `CLIENT_RECYCLE_REQUESTS` — Recycle persistent client after N requests (default: 200)
- `MAX_STANDBY_CLIENTS` — Number of pre-warmed standby clients (default: 2)
- `RATE_LIMIT_ENABLED` — Toggle rate limiting (default: true)

## Code Style

- **Formatter:** Black with 100-char line length, targeting Python 3.10+ (`pyproject.toml [tool.black]`)
- **Python:** 3.10+ required (CI tests 3.10, 3.11, 3.12)
- **Packages:** `src/` directory as the package root (`packages = [{include = "src"}]`)

## Notes

- Claude Agent SDK only supports Claude 4+ models, not Claude 3.x
- Default model is `claude-sonnet-4-6` (Claude 4.6 family)
- Memory safety details for constrained deployments are in `localClaude.md`
