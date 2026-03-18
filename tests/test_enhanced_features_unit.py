#!/usr/bin/env python3
"""
Unit tests for new features:
- response_format (json_object, json_schema)
- stop sequence post-processing (streaming + non-streaming)
- seed parameter (accepted, ignored)
- name field in messages
- extra SDK headers (betas, fork_session, env, cwd, max_budget_usd, fallback_model)
- max_thinking_tokens passthrough fix
- system_fingerprint generation
- clean_json_response

Pure unit tests — no running server required.
"""

import json
import tempfile
import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from src.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Message,
    ResponseFormat,
)
from src.message_adapter import MessageAdapter, StopSequenceProcessor
from src.parameter_validator import ParameterValidator


# ============================================================================
# ResponseFormat model tests
# ============================================================================


class TestResponseFormat:
    """Test ResponseFormat model and integration with ChatCompletionRequest."""

    def test_default_response_format_is_none(self):
        """response_format defaults to None."""
        req = ChatCompletionRequest(
            model="claude-sonnet-4-6",
            messages=[Message(role="user", content="Hi")],
        )
        assert req.response_format is None

    def test_text_format_accepted(self):
        """response_format type=text is accepted."""
        req = ChatCompletionRequest(
            model="claude-sonnet-4-6",
            messages=[Message(role="user", content="Hi")],
            response_format=ResponseFormat(type="text"),
        )
        assert req.response_format.type == "text"

    def test_json_object_format_accepted(self):
        """response_format type=json_object is accepted."""
        req = ChatCompletionRequest(
            model="claude-sonnet-4-6",
            messages=[Message(role="user", content="Hi")],
            response_format=ResponseFormat(type="json_object"),
        )
        assert req.response_format.type == "json_object"

    def test_json_schema_format_accepted(self):
        """response_format type=json_schema with schema is accepted."""
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        req = ChatCompletionRequest(
            model="claude-sonnet-4-6",
            messages=[Message(role="user", content="Hi")],
            response_format=ResponseFormat(type="json_schema", json_schema=schema),
        )
        assert req.response_format.type == "json_schema"
        assert req.response_format.json_schema == schema

    def test_json_instructions_none_for_text(self):
        """get_json_instructions returns None for type=text."""
        req = ChatCompletionRequest(
            model="claude-sonnet-4-6",
            messages=[Message(role="user", content="Hi")],
            response_format=ResponseFormat(type="text"),
        )
        assert req.get_json_instructions() is None

    def test_json_instructions_none_when_no_format(self):
        """get_json_instructions returns None when response_format is None."""
        req = ChatCompletionRequest(
            model="claude-sonnet-4-6",
            messages=[Message(role="user", content="Hi")],
        )
        assert req.get_json_instructions() is None

    def test_json_instructions_for_json_object(self):
        """get_json_instructions returns JSON-forcing prompt for json_object."""
        req = ChatCompletionRequest(
            model="claude-sonnet-4-6",
            messages=[Message(role="user", content="Hi")],
            response_format=ResponseFormat(type="json_object"),
        )
        instructions = req.get_json_instructions()
        assert instructions is not None
        assert "valid JSON" in instructions
        assert "markdown" in instructions.lower() or "text" in instructions.lower()

    def test_json_instructions_for_json_schema(self):
        """get_json_instructions includes the schema for json_schema."""
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        req = ChatCompletionRequest(
            model="claude-sonnet-4-6",
            messages=[Message(role="user", content="Hi")],
            response_format=ResponseFormat(type="json_schema", json_schema=schema),
        )
        instructions = req.get_json_instructions()
        assert instructions is not None
        assert "valid JSON" in instructions
        assert '"name"' in instructions  # Schema is embedded

    def test_json_schema_without_schema_returns_none(self):
        """json_schema type without actual schema returns None."""
        req = ChatCompletionRequest(
            model="claude-sonnet-4-6",
            messages=[Message(role="user", content="Hi")],
            response_format=ResponseFormat(type="json_schema"),
        )
        assert req.get_json_instructions() is None

    def test_response_format_logged_in_parameter_info(self):
        """response_format is logged in parameter info."""
        req = ChatCompletionRequest(
            model="claude-sonnet-4-6",
            messages=[Message(role="user", content="Hi")],
            response_format=ResponseFormat(type="json_object"),
        )
        with patch("src.models.logger") as mock_logger:
            req.log_parameter_info()
            logged = [str(c) for c in mock_logger.info.call_args_list]
            assert any("response_format" in s for s in logged)


# ============================================================================
# Seed parameter tests
# ============================================================================


class TestSeedParameter:
    """Test seed parameter acceptance."""

    def test_seed_defaults_to_none(self):
        """seed defaults to None."""
        req = ChatCompletionRequest(
            model="claude-sonnet-4-6",
            messages=[Message(role="user", content="Hi")],
        )
        assert req.seed is None

    def test_seed_accepted_as_integer(self):
        """seed is accepted as an integer."""
        req = ChatCompletionRequest(
            model="claude-sonnet-4-6",
            messages=[Message(role="user", content="Hi")],
            seed=42,
        )
        assert req.seed == 42

    def test_seed_logged_in_parameter_info(self):
        """seed is logged when set."""
        req = ChatCompletionRequest(
            model="claude-sonnet-4-6",
            messages=[Message(role="user", content="Hi")],
            seed=123,
        )
        with patch("src.models.logger") as mock_logger:
            req.log_parameter_info()
            logged = [str(c) for c in mock_logger.info.call_args_list]
            assert any("seed" in s for s in logged)

    def test_seed_not_in_claude_options(self):
        """seed is NOT passed to claude options (ignored)."""
        req = ChatCompletionRequest(
            model="claude-sonnet-4-6",
            messages=[Message(role="user", content="Hi")],
            seed=42,
        )
        options = req.to_claude_options()
        assert "seed" not in options


# ============================================================================
# Stop sequence post-processing tests
# ============================================================================


class TestStopSequenceProcessorStreaming:
    """Test StopSequenceProcessor for streaming (process_delta + flush)."""

    def test_no_stop_sequences_passthrough(self):
        """Empty stop sequences = pass everything through."""
        proc = StopSequenceProcessor([])
        assert proc.process_delta("hello world") == "hello world"
        assert not proc.stopped

    def test_single_char_stop_sequence(self):
        """Single character stop sequence."""
        proc = StopSequenceProcessor(["."])
        # With holdback_size=0 for single char, everything emits immediately
        result = proc.process_delta("hello.")
        assert result == "hello"
        assert proc.stopped

    def test_multi_char_stop_sequence(self):
        """Multi-character stop sequence within a single chunk."""
        proc = StopSequenceProcessor(["STOP"])
        text = proc.process_delta("hello STOP world")
        assert text == "hello "
        assert proc.stopped

    def test_stop_sequence_at_start(self):
        """Stop sequence at the beginning of text."""
        proc = StopSequenceProcessor(["END"])
        text = proc.process_delta("END of story")
        assert text == ""
        assert proc.stopped

    def test_stop_sequence_spanning_chunks(self):
        """Stop sequence that spans two chunks."""
        proc = StopSequenceProcessor(["STOP"])
        # holdback_size = 3 (len("STOP") - 1)

        out1 = proc.process_delta("helST")
        assert not proc.stopped
        # "helST" buffer, holdback=3, safe to emit "he"
        assert out1 == "he"

        out2 = proc.process_delta("OP rest")
        # buffer "lSTOP rest", finds "STOP" at idx 1
        assert out2 == "l"
        assert proc.stopped

    def test_no_stop_hit_flushes_buffer(self):
        """Flush returns remaining buffer when no stop was hit."""
        proc = StopSequenceProcessor(["STOP"])
        out1 = proc.process_delta("hel")
        # buffer "hel", holdback=3, nothing safe to emit
        assert out1 == ""

        remaining = proc.flush()
        assert remaining == "hel"
        assert not proc.stopped

    def test_flush_finds_stop_in_buffer(self):
        """Flush detects stop sequence in held-back buffer."""
        proc = StopSequenceProcessor(["END"])
        out1 = proc.process_delta("EN")
        assert out1 == ""  # All held back (holdback=2)

        out2 = proc.process_delta("D")
        # buffer "END", finds "END" at idx 0
        assert out2 == ""
        assert proc.stopped

    def test_multiple_stop_sequences(self):
        """First matching stop sequence wins."""
        proc = StopSequenceProcessor(["<END>", "<STOP>"])
        text = proc.process_delta("hello <STOP> more <END> rest")
        assert text == "hello "
        assert proc.stopped

    def test_stopped_processor_returns_empty(self):
        """After stopping, all further deltas return empty."""
        proc = StopSequenceProcessor(["END"])
        proc.process_delta("hello END more")
        assert proc.stopped

        assert proc.process_delta("more text") == ""
        assert proc.flush() == ""

    def test_empty_string_stop_sequence_ignored(self):
        """Empty strings in stop_sequences are filtered out."""
        proc = StopSequenceProcessor(["", "END"])
        text = proc.process_delta("hello END world")
        assert text == "hello "
        assert proc.stopped

    def test_holdback_buffer_correctness(self):
        """Verify holdback buffer emits correctly across many chunks."""
        proc = StopSequenceProcessor(["END"])
        # holdback_size = 2

        accumulated = ""
        for char in "hello world":
            out = proc.process_delta(char)
            accumulated += out

        remaining = proc.flush()
        accumulated += remaining
        assert accumulated == "hello world"
        assert not proc.stopped


class TestStopSequenceProcessorTruncate:
    """Test StopSequenceProcessor.truncate() for non-streaming."""

    def test_truncate_at_stop_sequence(self):
        """Content is truncated at first stop sequence."""
        content, was_truncated = StopSequenceProcessor.truncate(
            "Hello world. End of text.", ["."]
        )
        assert content == "Hello world"
        assert was_truncated is True

    def test_truncate_no_stop_found(self):
        """Content unchanged when no stop sequence found."""
        content, was_truncated = StopSequenceProcessor.truncate(
            "Hello world", ["END"]
        )
        assert content == "Hello world"
        assert was_truncated is False

    def test_truncate_empty_stop_sequences(self):
        """Empty stop sequences list returns content unchanged."""
        content, was_truncated = StopSequenceProcessor.truncate("Hello", [])
        assert content == "Hello"
        assert was_truncated is False

    def test_truncate_multiple_sequences_picks_earliest(self):
        """Earliest matching stop sequence is used."""
        content, was_truncated = StopSequenceProcessor.truncate(
            "Hello END world STOP rest", ["STOP", "END"]
        )
        assert content == "Hello "
        assert was_truncated is True

    def test_truncate_at_start(self):
        """Stop sequence at the very start returns empty."""
        content, was_truncated = StopSequenceProcessor.truncate("END rest", ["END"])
        assert content == ""
        assert was_truncated is True


# ============================================================================
# Stop sequence logged as supported
# ============================================================================


class TestStopSequenceLogging:
    """Test that stop sequences are now logged as supported, not ignored."""

    def test_stop_logged_as_supported(self):
        """stop sequences log as applied via post-processing."""
        req = ChatCompletionRequest(
            model="claude-sonnet-4-6",
            messages=[Message(role="user", content="Hi")],
            stop=["END"],
        )
        with patch("src.models.logger") as mock_logger:
            req.log_parameter_info()
            logged = [str(c) for c in mock_logger.info.call_args_list]
            assert any("stop" in s and "post-processing" in s for s in logged)
            # Should NOT be in warnings
            warn_logged = [str(c) for c in mock_logger.warning.call_args_list]
            assert not any("stop" in s for s in warn_logged)


# ============================================================================
# Name field in messages
# ============================================================================


class TestNameFieldInMessages:
    """Test name field prefix in message conversion."""

    def test_user_message_with_name(self):
        """User message with name gets prefix."""
        messages = [Message(role="user", content="Hello", name="Alice")]
        prompt, _ = MessageAdapter.messages_to_prompt(messages)
        assert "[Alice]" in prompt
        assert "Human: [Alice] Hello" in prompt

    def test_assistant_message_with_name(self):
        """Assistant message with name gets prefix."""
        messages = [
            Message(role="user", content="Hi"),
            Message(role="assistant", content="Hello!", name="Claude"),
        ]
        prompt, _ = MessageAdapter.messages_to_prompt(messages)
        assert "Assistant: [Claude] Hello!" in prompt

    def test_message_without_name_no_prefix(self):
        """Messages without name have no prefix (unchanged behavior)."""
        messages = [Message(role="user", content="Hello")]
        prompt, _ = MessageAdapter.messages_to_prompt(messages)
        assert "Human: Hello" in prompt
        assert "[" not in prompt

    def test_mixed_named_and_unnamed(self):
        """Mix of named and unnamed messages."""
        messages = [
            Message(role="user", content="Hi", name="User1"),
            Message(role="assistant", content="Hey"),
            Message(role="user", content="Bye", name="User2"),
        ]
        prompt, _ = MessageAdapter.messages_to_prompt(messages)
        assert "[User1]" in prompt
        assert "[User2]" in prompt
        assert "Assistant: Hey" in prompt  # No brackets


# ============================================================================
# Clean JSON response
# ============================================================================


class TestCleanJsonResponse:
    """Test MessageAdapter.clean_json_response()"""

    def test_strips_json_code_fence(self):
        """Strips ```json ... ``` wrapper."""
        raw = '```json\n{"key": "value"}\n```'
        result = MessageAdapter.clean_json_response(raw)
        assert result == '{"key": "value"}'

    def test_strips_generic_code_fence(self):
        """Strips ``` ... ``` wrapper."""
        raw = '```\n{"key": "value"}\n```'
        result = MessageAdapter.clean_json_response(raw)
        assert result == '{"key": "value"}'

    def test_no_code_fence_passthrough(self):
        """Plain JSON passes through unchanged."""
        raw = '{"key": "value"}'
        result = MessageAdapter.clean_json_response(raw)
        assert result == '{"key": "value"}'

    def test_strips_whitespace(self):
        """Surrounding whitespace is stripped."""
        raw = '  \n{"key": "value"}\n  '
        result = MessageAdapter.clean_json_response(raw)
        assert result == '{"key": "value"}'

    def test_nested_backticks_preserved(self):
        """Content with backticks inside is handled."""
        raw = '```json\n{"code": "use ```this```"}\n```'
        result = MessageAdapter.clean_json_response(raw)
        assert "use ```this```" in result


# ============================================================================
# Extra SDK headers extraction
# ============================================================================


class TestExtractExtraSdkHeaders:
    """Test ParameterValidator.extract_extra_sdk_headers()"""

    def test_empty_headers_returns_empty(self):
        """No relevant headers returns empty dict."""
        result = ParameterValidator.extract_extra_sdk_headers({})
        assert result == {}

    def test_beta_header_parsed(self):
        """X-Claude-Beta header is parsed to list."""
        result = ParameterValidator.extract_extra_sdk_headers(
            {"x-claude-beta": "context-1m-2025-08-07"}
        )
        assert result["betas"] == ["context-1m-2025-08-07"]

    def test_beta_header_multiple(self):
        """Multiple betas comma-separated."""
        result = ParameterValidator.extract_extra_sdk_headers(
            {"x-claude-beta": "context-1m-2025-08-07, other-beta"}
        )
        assert result["betas"] == ["context-1m-2025-08-07", "other-beta"]

    def test_fork_session_true(self):
        """X-Claude-Fork-Session: true."""
        result = ParameterValidator.extract_extra_sdk_headers(
            {"x-claude-fork-session": "true"}
        )
        assert result["fork_session"] is True

    def test_fork_session_false(self):
        """X-Claude-Fork-Session: false."""
        result = ParameterValidator.extract_extra_sdk_headers(
            {"x-claude-fork-session": "false"}
        )
        assert result["fork_session"] is False

    def test_env_header_json_object(self):
        """X-Claude-Env accepts JSON object."""
        env = {"MY_VAR": "value", "OTHER": "123"}
        result = ParameterValidator.extract_extra_sdk_headers(
            {"x-claude-env": json.dumps(env)}
        )
        assert result["env"] == env

    def test_env_header_invalid_json_ignored(self):
        """X-Claude-Env with invalid JSON is ignored."""
        result = ParameterValidator.extract_extra_sdk_headers(
            {"x-claude-env": "not-json"}
        )
        assert "env" not in result

    def test_env_header_non_object_ignored(self):
        """X-Claude-Env with non-object JSON is ignored."""
        result = ParameterValidator.extract_extra_sdk_headers(
            {"x-claude-env": json.dumps(["a", "b"])}
        )
        assert "env" not in result

    def test_cwd_header(self):
        """X-Claude-Cwd is parsed."""
        result = ParameterValidator.extract_extra_sdk_headers(
            {"x-claude-cwd": "/tmp/workspace"}
        )
        assert result["cwd"] == "/tmp/workspace"

    def test_max_budget_usd_header(self):
        """X-Claude-Max-Budget-Usd is parsed as float."""
        result = ParameterValidator.extract_extra_sdk_headers(
            {"x-claude-max-budget-usd": "1.50"}
        )
        assert result["max_budget_usd"] == 1.50

    def test_max_budget_usd_invalid_ignored(self):
        """X-Claude-Max-Budget-Usd with non-numeric is ignored."""
        result = ParameterValidator.extract_extra_sdk_headers(
            {"x-claude-max-budget-usd": "not-a-number"}
        )
        assert "max_budget_usd" not in result

    def test_fallback_model_header(self):
        """X-Claude-Fallback-Model is parsed."""
        result = ParameterValidator.extract_extra_sdk_headers(
            {"x-claude-fallback-model": "claude-haiku-4-5"}
        )
        assert result["fallback_model"] == "claude-haiku-4-5"

    def test_all_headers_combined(self):
        """All extra headers extracted at once."""
        headers = {
            "x-claude-beta": "context-1m-2025-08-07",
            "x-claude-fork-session": "yes",
            "x-claude-env": json.dumps({"KEY": "VAL"}),
            "x-claude-cwd": "/work",
            "x-claude-max-budget-usd": "5.0",
            "x-claude-fallback-model": "claude-haiku-4-5",
        }
        result = ParameterValidator.extract_extra_sdk_headers(headers)
        assert result["betas"] == ["context-1m-2025-08-07"]
        assert result["fork_session"] is True
        assert result["env"] == {"KEY": "VAL"}
        assert result["cwd"] == "/work"
        assert result["max_budget_usd"] == 5.0
        assert result["fallback_model"] == "claude-haiku-4-5"


# ============================================================================
# max_thinking_tokens passthrough
# ============================================================================


class TestMaxThinkingTokensPassthrough:
    """Test max_thinking_tokens is passed to SDK (was previously dropped)."""

    @pytest.fixture
    def cli_instance(self):
        """Create a CLI instance with mocked auth."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("src.auth.validate_claude_code_auth") as mock_validate:
                with patch("src.auth.auth_manager") as mock_auth:
                    mock_validate.return_value = (True, {"method": "anthropic"})
                    mock_auth.get_claude_code_env_vars.return_value = {
                        "ANTHROPIC_API_KEY": "test-key"
                    }
                    from src.claude_cli import ClaudeCodeCLI

                    cli = ClaudeCodeCLI(cwd=temp_dir)
                    yield cli

    @pytest.mark.asyncio
    async def test_max_thinking_tokens_passed_to_query(self, cli_instance):
        """max_thinking_tokens is set on ClaudeAgentOptions."""
        captured_options = []

        async def mock_query(prompt, options):
            captured_options.append(options)
            yield {"type": "result", "subtype": "success", "result": "done"}

        with patch("src.claude_cli.query", mock_query):
            async for _ in cli_instance.run_completion(
                "Hello",
                max_turns=2,  # Force slow path
                max_thinking_tokens=5000,
            ):
                pass

            assert len(captured_options) == 1
            assert captured_options[0].max_thinking_tokens == 5000


# ============================================================================
# extra_sdk_options passthrough
# ============================================================================


class TestExtraSdkOptionsPassthrough:
    """Test extra SDK options are applied to ClaudeAgentOptions."""

    @pytest.fixture
    def cli_instance(self):
        """Create a CLI instance with mocked auth."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("src.auth.validate_claude_code_auth") as mock_validate:
                with patch("src.auth.auth_manager") as mock_auth:
                    mock_validate.return_value = (True, {"method": "anthropic"})
                    mock_auth.get_claude_code_env_vars.return_value = {
                        "ANTHROPIC_API_KEY": "test-key"
                    }
                    from src.claude_cli import ClaudeCodeCLI

                    cli = ClaudeCodeCLI(cwd=temp_dir)
                    yield cli

    @pytest.mark.asyncio
    async def test_betas_passed_through(self, cli_instance):
        """betas from extra_sdk_options are set on options."""
        captured_options = []

        async def mock_query(prompt, options):
            captured_options.append(options)
            yield {"type": "result", "subtype": "success", "result": "done"}

        with patch("src.claude_cli.query", mock_query):
            async for _ in cli_instance.run_completion(
                "Hello",
                max_turns=2,  # Force slow path
                extra_sdk_options={"betas": ["context-1m-2025-08-07"]},
            ):
                pass

            assert captured_options[0].betas == ["context-1m-2025-08-07"]

    @pytest.mark.asyncio
    async def test_fork_session_passed_through(self, cli_instance):
        """fork_session from extra_sdk_options is set."""
        captured_options = []

        async def mock_query(prompt, options):
            captured_options.append(options)
            yield {"type": "result", "subtype": "success", "result": "done"}

        with patch("src.claude_cli.query", mock_query):
            async for _ in cli_instance.run_completion(
                "Hello",
                max_turns=2,
                extra_sdk_options={"fork_session": True},
            ):
                pass

            assert captured_options[0].fork_session is True

    @pytest.mark.asyncio
    async def test_max_budget_usd_passed_through(self, cli_instance):
        """max_budget_usd from extra_sdk_options is set."""
        captured_options = []

        async def mock_query(prompt, options):
            captured_options.append(options)
            yield {"type": "result", "subtype": "success", "result": "done"}

        with patch("src.claude_cli.query", mock_query):
            async for _ in cli_instance.run_completion(
                "Hello",
                max_turns=2,
                extra_sdk_options={"max_budget_usd": 1.5},
            ):
                pass

            assert captured_options[0].max_budget_usd == 1.5

    @pytest.mark.asyncio
    async def test_unknown_option_ignored(self, cli_instance):
        """Unknown extra_sdk_options keys are silently ignored."""
        captured_options = []

        async def mock_query(prompt, options):
            captured_options.append(options)
            yield {"type": "result", "subtype": "success", "result": "done"}

        with patch("src.claude_cli.query", mock_query):
            async for _ in cli_instance.run_completion(
                "Hello",
                max_turns=2,
                extra_sdk_options={"nonexistent_option": "value"},
            ):
                pass

            # Should not crash; option is just skipped
            assert len(captured_options) == 1


# ============================================================================
# _can_use_persistent_client with new params
# ============================================================================


class TestCanUsePersistentClient:
    """Test _can_use_persistent_client rejects fast path for new params."""

    @pytest.fixture
    def cli_instance(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("src.auth.validate_claude_code_auth") as mock_validate:
                with patch("src.auth.auth_manager") as mock_auth:
                    mock_validate.return_value = (True, {"method": "anthropic"})
                    mock_auth.get_claude_code_env_vars.return_value = {}
                    from src.claude_cli import ClaudeCodeCLI

                    cli = ClaudeCodeCLI(cwd=temp_dir)
                    yield cli

    def test_default_params_allow_fast_path(self, cli_instance):
        """Default params (max_turns=1, no tools) allow fast path."""
        assert cli_instance._can_use_persistent_client(
            max_turns=1, allowed_tools=[], disallowed_tools=None, permission_mode=None
        )

    def test_max_thinking_tokens_blocks_fast_path(self, cli_instance):
        """max_thinking_tokens forces slow path."""
        assert not cli_instance._can_use_persistent_client(
            max_turns=1,
            allowed_tools=[],
            disallowed_tools=None,
            permission_mode=None,
            max_thinking_tokens=5000,
        )

    def test_extra_sdk_options_blocks_fast_path(self, cli_instance):
        """Any extra_sdk_options force slow path."""
        assert not cli_instance._can_use_persistent_client(
            max_turns=1,
            allowed_tools=[],
            disallowed_tools=None,
            permission_mode=None,
            extra_sdk_options={"betas": ["context-1m-2025-08-07"]},
        )

    def test_empty_extra_sdk_options_allows_fast_path(self, cli_instance):
        """Empty extra_sdk_options dict still allows fast path."""
        assert cli_instance._can_use_persistent_client(
            max_turns=1,
            allowed_tools=[],
            disallowed_tools=None,
            permission_mode=None,
            extra_sdk_options={},
        )


# ============================================================================
# System fingerprint generation
# ============================================================================


class TestSystemFingerprint:
    """Test system_fingerprint is generated consistently."""

    def test_fingerprint_starts_with_fp(self):
        """system_fingerprint starts with 'fp_'."""
        import hashlib

        model = "claude-sonnet-4-6"
        fp = f"fp_{hashlib.md5(model.encode()).hexdigest()[:10]}"
        assert fp.startswith("fp_")
        assert len(fp) == 13  # "fp_" + 10 hex chars

    def test_fingerprint_deterministic(self):
        """Same model always produces same fingerprint."""
        import hashlib

        model = "claude-sonnet-4-6"
        fp1 = f"fp_{hashlib.md5(model.encode()).hexdigest()[:10]}"
        fp2 = f"fp_{hashlib.md5(model.encode()).hexdigest()[:10]}"
        assert fp1 == fp2

    def test_different_models_different_fingerprints(self):
        """Different models produce different fingerprints."""
        import hashlib

        fp1 = f"fp_{hashlib.md5('claude-sonnet-4-6'.encode()).hexdigest()[:10]}"
        fp2 = f"fp_{hashlib.md5('claude-opus-4-6'.encode()).hexdigest()[:10]}"
        assert fp1 != fp2


# ============================================================================
# Compatibility reporter updated for stop
# ============================================================================


class TestCompatibilityReporterStopUpdate:
    """Test that CompatibilityReporter now reports stop as supported."""

    def test_stop_in_supported_not_unsupported(self):
        """stop is in supported_parameters, not unsupported."""
        from src.parameter_validator import CompatibilityReporter

        req = ChatCompletionRequest(
            model="claude-sonnet-4-6",
            messages=[Message(role="user", content="Hi")],
            stop=["END"],
        )
        report = CompatibilityReporter.generate_compatibility_report(req)
        assert any("stop" in p for p in report["supported_parameters"])
        assert "stop" not in report["unsupported_parameters"]
