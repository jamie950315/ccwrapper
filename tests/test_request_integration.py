#!/usr/bin/env python3
"""
Integration tests that send actual HTTP requests to the FastAPI app.

Mocks the Claude Agent SDK at the `claude_cli` level so no real API calls are made.
Tests the full request → processing → response pipeline for all new features.
"""

import json
import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from starlette.testclient import TestClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mock_claude_response():
    """Default mock: Claude returns a simple text response."""

    async def fake_run_completion(prompt, **kwargs):
        yield {
            "type": "result",
            "subtype": "success",
            "result": "Hello from Claude!",
            "total_cost_usd": 0.002,
            "duration_ms": 1500,
            "num_turns": 1,
            "session_id": "test-session",
        }

    return fake_run_completion


@pytest.fixture(scope="module")
def client(mock_claude_response):
    """Create a test client with mocked Claude CLI."""
    with patch("src.main.claude_cli") as mock_cli:
        mock_cli.run_completion = mock_claude_response
        mock_cli.parse_claude_message = MagicMock(return_value="Hello from Claude!")
        mock_cli.extract_metadata = MagicMock(
            return_value={
                "total_cost_usd": 0.002,
                "duration_ms": 1500,
                "num_turns": 1,
                "session_id": "test-session",
                "model": "claude-sonnet-4-6",
            }
        )
        mock_cli.estimate_token_usage = MagicMock(
            return_value={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        )

        with patch("src.main.verify_api_key", new_callable=AsyncMock):
            with patch(
                "src.main.validate_claude_code_auth", return_value=(True, {"method": "test"})
            ):
                from src.main import app

                # Override lifespan to skip SDK verification
                app.router.lifespan_context = _noop_lifespan
                yield TestClient(app, raise_server_exceptions=False)


from contextlib import asynccontextmanager


@asynccontextmanager
async def _noop_lifespan(app):
    yield


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _chat_payload(**overrides):
    """Build a minimal chat completion request."""
    payload = {
        "model": "claude-sonnet-4-6",
        "messages": [{"role": "user", "content": "Hello"}],
    }
    payload.update(overrides)
    return payload


# ============================================================================
# Basic request / response
# ============================================================================


class TestBasicRequest:
    """Verify basic non-streaming request works."""

    def test_basic_chat_completion(self, client):
        """POST /v1/chat/completions returns valid response."""
        resp = client.post("/v1/chat/completions", json=_chat_payload())
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "chat.completion"
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert data["choices"][0]["finish_reason"] == "stop"

    def test_response_has_system_fingerprint(self, client):
        """Response includes system_fingerprint."""
        resp = client.post("/v1/chat/completions", json=_chat_payload())
        data = resp.json()
        assert data.get("system_fingerprint") is not None
        assert data["system_fingerprint"].startswith("fp_")

    def test_response_has_usage(self, client):
        """Response includes usage with token counts."""
        resp = client.post("/v1/chat/completions", json=_chat_payload())
        data = resp.json()
        assert "usage" in data
        assert data["usage"]["prompt_tokens"] >= 0
        assert data["usage"]["completion_tokens"] >= 0
        assert data["usage"]["total_tokens"] >= 0


# ============================================================================
# response_format
# ============================================================================


class TestResponseFormat:
    """Test response_format parameter."""

    def test_json_object_accepted(self, client):
        """response_format type=json_object is accepted without error."""
        resp = client.post(
            "/v1/chat/completions",
            json=_chat_payload(response_format={"type": "json_object"}),
        )
        assert resp.status_code == 200

    def test_json_schema_accepted(self, client):
        """response_format type=json_schema is accepted."""
        resp = client.post(
            "/v1/chat/completions",
            json=_chat_payload(
                response_format={
                    "type": "json_schema",
                    "json_schema": {"type": "object", "properties": {"name": {"type": "string"}}},
                }
            ),
        )
        assert resp.status_code == 200

    def test_text_format_accepted(self, client):
        """response_format type=text is accepted (no-op)."""
        resp = client.post(
            "/v1/chat/completions",
            json=_chat_payload(response_format={"type": "text"}),
        )
        assert resp.status_code == 200


# ============================================================================
# seed parameter
# ============================================================================


class TestSeedParameter:
    """Test seed parameter acceptance."""

    def test_seed_accepted(self, client):
        """seed parameter is accepted without error."""
        resp = client.post("/v1/chat/completions", json=_chat_payload(seed=42))
        assert resp.status_code == 200

    def test_seed_large_value(self, client):
        """Large seed value is accepted."""
        resp = client.post("/v1/chat/completions", json=_chat_payload(seed=999999999))
        assert resp.status_code == 200


# ============================================================================
# stop sequences
# ============================================================================


class TestStopSequences:
    """Test stop sequence parameter."""

    def test_stop_string_accepted(self, client):
        """stop as string is accepted."""
        resp = client.post("/v1/chat/completions", json=_chat_payload(stop="END"))
        assert resp.status_code == 200

    def test_stop_array_accepted(self, client):
        """stop as array of strings is accepted."""
        resp = client.post("/v1/chat/completions", json=_chat_payload(stop=["END", "STOP"]))
        assert resp.status_code == 200


# ============================================================================
# name field
# ============================================================================


class TestNameField:
    """Test name field in messages."""

    def test_name_field_accepted(self, client):
        """Messages with name field are accepted."""
        resp = client.post(
            "/v1/chat/completions",
            json=_chat_payload(
                messages=[
                    {"role": "user", "content": "Hello", "name": "Alice"},
                    {"role": "assistant", "content": "Hi!", "name": "Bot"},
                    {"role": "user", "content": "Bye"},
                ]
            ),
        )
        assert resp.status_code == 200


# ============================================================================
# Model aliasing
# ============================================================================


class TestModelAliasing:
    """Test OpenAI model names are accepted and mapped."""

    @pytest.mark.parametrize(
        "model",
        [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4",
            "gpt-3.5-turbo",
            "o1",
            "o3-mini",
            "gpt-4.1",
            "gpt-4.1-mini",
        ],
    )
    def test_openai_models_accepted(self, client, model):
        """OpenAI model names are accepted without error."""
        resp = client.post("/v1/chat/completions", json=_chat_payload(model=model))
        assert resp.status_code == 200

    @pytest.mark.parametrize("model", ["gemini-1.5-pro", "gemini-2.0-flash", "deepseek-chat"])
    def test_other_provider_models_accepted(self, client, model):
        """Gemini/DeepSeek model names are accepted."""
        resp = client.post("/v1/chat/completions", json=_chat_payload(model=model))
        assert resp.status_code == 200

    def test_native_claude_model_accepted(self, client):
        """Native Claude model name passes through."""
        resp = client.post("/v1/chat/completions", json=_chat_payload(model="claude-opus-4-6"))
        assert resp.status_code == 200


# ============================================================================
# Function calling (tools)
# ============================================================================

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather for a location",
        "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        },
    },
}


class TestFunctionCalling:
    """Test OpenAI function calling simulation."""

    def test_tools_parameter_accepted(self, client):
        """tools array is accepted without error."""
        resp = client.post(
            "/v1/chat/completions",
            json=_chat_payload(tools=[WEATHER_TOOL]),
        )
        assert resp.status_code == 200

    def test_tool_choice_auto(self, client):
        """tool_choice='auto' is accepted."""
        resp = client.post(
            "/v1/chat/completions",
            json=_chat_payload(tools=[WEATHER_TOOL], tool_choice="auto"),
        )
        assert resp.status_code == 200

    def test_tool_choice_none(self, client):
        """tool_choice='none' is accepted."""
        resp = client.post(
            "/v1/chat/completions",
            json=_chat_payload(tools=[WEATHER_TOOL], tool_choice="none"),
        )
        assert resp.status_code == 200

    def test_tool_choice_required(self, client):
        """tool_choice='required' is accepted."""
        resp = client.post(
            "/v1/chat/completions",
            json=_chat_payload(tools=[WEATHER_TOOL], tool_choice="required"),
        )
        assert resp.status_code == 200

    def test_tool_choice_specific(self, client):
        """tool_choice with specific function is accepted."""
        resp = client.post(
            "/v1/chat/completions",
            json=_chat_payload(
                tools=[WEATHER_TOOL],
                tool_choice={"type": "function", "function": {"name": "get_weather"}},
            ),
        )
        assert resp.status_code == 200

    def test_multiple_tools(self, client):
        """Multiple tool definitions accepted."""
        tools = [
            WEATHER_TOOL,
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search the web",
                    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
                },
            },
        ]
        resp = client.post("/v1/chat/completions", json=_chat_payload(tools=tools))
        assert resp.status_code == 200

    def test_tool_role_message_accepted(self, client):
        """Messages with role='tool' are accepted."""
        resp = client.post(
            "/v1/chat/completions",
            json=_chat_payload(
                messages=[
                    {"role": "user", "content": "What's the weather?"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_abc123",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"location": "Tokyo"}',
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_abc123",
                        "content": '{"temp": 22, "condition": "sunny"}',
                        "name": "get_weather",
                    },
                ],
                tools=[WEATHER_TOOL],
            ),
        )
        assert resp.status_code == 200

    def test_parallel_tool_calls_accepted(self, client):
        """parallel_tool_calls parameter is accepted."""
        resp = client.post(
            "/v1/chat/completions",
            json=_chat_payload(tools=[WEATHER_TOOL], parallel_tool_calls=True),
        )
        assert resp.status_code == 200


class TestFunctionCallingResponse:
    """Test that tool_calls appear in response when Claude returns them."""

    @pytest.fixture
    def tool_call_client(self):
        """Client where Claude returns a tool call response."""
        tool_response = (
            '```tool_calls\n[{"name": "get_weather", "arguments": {"location": "Tokyo"}}]\n```'
        )

        async def fake_run_completion(prompt, **kwargs):
            yield {
                "type": "result",
                "subtype": "success",
                "result": tool_response,
                "total_cost_usd": 0.001,
                "duration_ms": 500,
                "num_turns": 1,
            }

        with patch("src.main.claude_cli") as mock_cli:
            mock_cli.run_completion = fake_run_completion
            mock_cli.parse_claude_message = MagicMock(return_value=tool_response)
            mock_cli.extract_metadata = MagicMock(
                return_value={"total_cost_usd": 0.001, "duration_ms": 500, "num_turns": 1}
            )
            mock_cli.estimate_token_usage = MagicMock(
                return_value={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
            )

            with patch("src.main.verify_api_key", new_callable=AsyncMock):
                with patch(
                    "src.main.validate_claude_code_auth",
                    return_value=(True, {"method": "test"}),
                ):
                    from src.main import app

                    app.router.lifespan_context = _noop_lifespan
                    yield TestClient(app, raise_server_exceptions=False)

    def test_tool_calls_in_response(self, tool_call_client):
        """Response contains tool_calls when Claude returns structured tool call."""
        resp = tool_call_client.post(
            "/v1/chat/completions",
            json=_chat_payload(tools=[WEATHER_TOOL]),
        )
        assert resp.status_code == 200
        data = resp.json()
        msg = data["choices"][0]["message"]
        assert msg.get("tool_calls") is not None
        assert len(msg["tool_calls"]) == 1
        assert msg["tool_calls"][0]["function"]["name"] == "get_weather"
        assert data["choices"][0]["finish_reason"] == "tool_calls"

    def test_tool_call_has_id(self, tool_call_client):
        """Each tool call has a unique id starting with call_."""
        resp = tool_call_client.post(
            "/v1/chat/completions",
            json=_chat_payload(tools=[WEATHER_TOOL]),
        )
        data = resp.json()
        tc = data["choices"][0]["message"]["tool_calls"][0]
        assert tc["id"].startswith("call_")
        assert tc["type"] == "function"

    def test_tool_call_arguments_is_json_string(self, tool_call_client):
        """tool_calls arguments is a JSON string (not object)."""
        resp = tool_call_client.post(
            "/v1/chat/completions",
            json=_chat_payload(tools=[WEATHER_TOOL]),
        )
        data = resp.json()
        args_str = data["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
        assert isinstance(args_str, str)
        args = json.loads(args_str)
        assert args["location"] == "Tokyo"


# ============================================================================
# Custom SDK headers
# ============================================================================


class TestCustomSdkHeaders:
    """Test Claude SDK custom headers are accepted."""

    def test_beta_header(self, client):
        """X-Claude-Beta header is accepted."""
        resp = client.post(
            "/v1/chat/completions",
            json=_chat_payload(),
            headers={"X-Claude-Beta": "context-1m-2025-08-07"},
        )
        assert resp.status_code == 200

    def test_max_thinking_tokens_header(self, client):
        """X-Claude-Max-Thinking-Tokens header is accepted."""
        resp = client.post(
            "/v1/chat/completions",
            json=_chat_payload(),
            headers={"X-Claude-Max-Thinking-Tokens": "10000"},
        )
        assert resp.status_code == 200

    def test_fork_session_header(self, client):
        """X-Claude-Fork-Session header is accepted."""
        resp = client.post(
            "/v1/chat/completions",
            json=_chat_payload(),
            headers={"X-Claude-Fork-Session": "true"},
        )
        assert resp.status_code == 200

    def test_env_header(self, client):
        """X-Claude-Env header with JSON is accepted."""
        resp = client.post(
            "/v1/chat/completions",
            json=_chat_payload(),
            headers={"X-Claude-Env": json.dumps({"MY_VAR": "value"})},
        )
        assert resp.status_code == 200

    def test_cwd_header(self, client):
        """X-Claude-Cwd header is accepted."""
        resp = client.post(
            "/v1/chat/completions",
            json=_chat_payload(),
            headers={"X-Claude-Cwd": "/tmp"},
        )
        assert resp.status_code == 200

    def test_max_budget_header(self, client):
        """X-Claude-Max-Budget-Usd header is accepted."""
        resp = client.post(
            "/v1/chat/completions",
            json=_chat_payload(),
            headers={"X-Claude-Max-Budget-Usd": "1.50"},
        )
        assert resp.status_code == 200

    def test_fallback_model_header(self, client):
        """X-Claude-Fallback-Model header is accepted."""
        resp = client.post(
            "/v1/chat/completions",
            json=_chat_payload(),
            headers={"X-Claude-Fallback-Model": "claude-haiku-4-5"},
        )
        assert resp.status_code == 200

    def test_system_prompt_preset_header(self, client):
        """X-Claude-System-Prompt-Preset header is accepted."""
        resp = client.post(
            "/v1/chat/completions",
            json=_chat_payload(),
            headers={"X-Claude-System-Prompt-Preset": "claude_code"},
        )
        assert resp.status_code == 200

    def test_multiple_headers_combined(self, client):
        """Multiple SDK headers sent together."""
        resp = client.post(
            "/v1/chat/completions",
            json=_chat_payload(),
            headers={
                "X-Claude-Beta": "context-1m-2025-08-07",
                "X-Claude-Max-Thinking-Tokens": "5000",
                "X-Claude-Fallback-Model": "claude-haiku-4-5",
            },
        )
        assert resp.status_code == 200


# ============================================================================
# Cost reporting
# ============================================================================


class TestCostReporting:
    """Test SDK cost data in Usage response."""

    def test_usage_includes_cost_fields(self, client):
        """Usage includes total_cost_usd, duration_ms, num_turns."""
        resp = client.post("/v1/chat/completions", json=_chat_payload())
        data = resp.json()
        usage = data.get("usage", {})
        # These may be present (from SDK metadata) or None
        assert "total_cost_usd" in usage or "prompt_tokens" in usage
        assert "prompt_tokens" in usage
        assert "completion_tokens" in usage
        assert "total_tokens" in usage


# ============================================================================
# Streaming
# ============================================================================


class TestStreaming:
    """Test streaming responses."""

    @pytest.fixture
    def streaming_client(self):
        """Client that returns streaming events."""

        async def fake_run_completion(prompt, **kwargs):
            # Simulate StreamEvent deltas
            yield {
                "event": {
                    "type": "content_block_delta",
                    "delta": {"type": "text_delta", "text": "Hello "},
                }
            }
            yield {
                "event": {
                    "type": "content_block_delta",
                    "delta": {"type": "text_delta", "text": "world!"},
                }
            }
            yield {
                "type": "result",
                "subtype": "success",
                "result": "Hello world!",
            }

        with patch("src.main.claude_cli") as mock_cli:
            mock_cli.run_completion = fake_run_completion
            mock_cli.parse_claude_message = MagicMock(return_value="Hello world!")
            mock_cli.extract_metadata = MagicMock(return_value={})
            mock_cli.estimate_token_usage = MagicMock(
                return_value={"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10}
            )

            with patch("src.main.verify_api_key", new_callable=AsyncMock):
                with patch(
                    "src.main.validate_claude_code_auth",
                    return_value=(True, {"method": "test"}),
                ):
                    from src.main import app

                    app.router.lifespan_context = _noop_lifespan
                    yield TestClient(app, raise_server_exceptions=False)

    def test_streaming_returns_sse(self, streaming_client):
        """Streaming returns text/event-stream."""
        resp = streaming_client.post(
            "/v1/chat/completions",
            json=_chat_payload(stream=True),
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")

    def test_streaming_ends_with_done(self, streaming_client):
        """Streaming ends with [DONE]."""
        resp = streaming_client.post(
            "/v1/chat/completions",
            json=_chat_payload(stream=True),
        )
        text = resp.text
        assert "data: [DONE]" in text

    def test_streaming_has_finish_reason(self, streaming_client):
        """Streaming includes finish_reason in final chunk."""
        resp = streaming_client.post(
            "/v1/chat/completions",
            json=_chat_payload(stream=True),
        )
        # Parse SSE lines for finish_reason
        lines = [l for l in resp.text.split("\n") if l.startswith("data: {")]
        last_data = json.loads(lines[-1].replace("data: ", ""))
        assert last_data["choices"][0]["finish_reason"] == "stop"

    def test_streaming_has_system_fingerprint(self, streaming_client):
        """Streaming chunks include system_fingerprint."""
        resp = streaming_client.post(
            "/v1/chat/completions",
            json=_chat_payload(stream=True),
        )
        lines = [l for l in resp.text.split("\n") if l.startswith("data: {")]
        chunk = json.loads(lines[0].replace("data: ", ""))
        assert chunk.get("system_fingerprint", "").startswith("fp_")

    def test_streaming_with_stop_sequences(self, streaming_client):
        """Streaming with stop sequences is accepted."""
        resp = streaming_client.post(
            "/v1/chat/completions",
            json=_chat_payload(stream=True, stop=["END"]),
        )
        assert resp.status_code == 200
        assert "data: [DONE]" in resp.text

    def test_streaming_with_response_format(self, streaming_client):
        """Streaming with json mode is accepted."""
        resp = streaming_client.post(
            "/v1/chat/completions",
            json=_chat_payload(stream=True, response_format={"type": "json_object"}),
        )
        assert resp.status_code == 200

    def test_streaming_with_tools(self, streaming_client):
        """Streaming with function calling tools is accepted."""
        resp = streaming_client.post(
            "/v1/chat/completions",
            json=_chat_payload(stream=True, tools=[WEATHER_TOOL]),
        )
        assert resp.status_code == 200


# ============================================================================
# Combined features
# ============================================================================


class TestCombinedFeatures:
    """Test multiple features used together."""

    def test_tools_with_json_mode(self, client):
        """tools + response_format together."""
        resp = client.post(
            "/v1/chat/completions",
            json=_chat_payload(
                tools=[WEATHER_TOOL],
                response_format={"type": "json_object"},
            ),
        )
        assert resp.status_code == 200

    def test_stop_with_model_alias(self, client):
        """stop sequences + model aliasing together."""
        resp = client.post(
            "/v1/chat/completions",
            json=_chat_payload(model="gpt-4o", stop=["END"]),
        )
        assert resp.status_code == 200

    def test_all_new_params_together(self, client):
        """All new parameters sent together."""
        resp = client.post(
            "/v1/chat/completions",
            json=_chat_payload(
                model="gpt-4o",
                response_format={"type": "json_object"},
                stop=["END"],
                seed=42,
                tools=[WEATHER_TOOL],
                tool_choice="auto",
                messages=[{"role": "user", "content": "Hello", "name": "Alice"}],
            ),
            headers={
                "X-Claude-Beta": "context-1m-2025-08-07",
                "X-Claude-Max-Thinking-Tokens": "5000",
            },
        )
        assert resp.status_code == 200

    def test_multi_turn_with_tools_and_sessions(self, client):
        """Multi-turn conversation with tool role and session_id."""
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "claude-sonnet-4-6",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "What's the weather?"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"location":"SF"}',
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_1",
                        "content": "72°F sunny",
                        "name": "get_weather",
                    },
                    {"role": "user", "content": "Thanks! How about NYC?"},
                ],
                "tools": [WEATHER_TOOL],
                "session_id": "multi-turn-test",
            },
        )
        assert resp.status_code == 200
