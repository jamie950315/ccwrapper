#!/usr/bin/env python3
"""
Unit tests for function calling simulation.
"""

import json
import pytest

from src.function_calling import (
    build_tools_system_prompt,
    parse_tool_calls,
    convert_tool_messages,
)
from src.models import Message, ToolDef, FunctionDef, ToolCall, ToolCallFunction
from src.message_adapter import MessageAdapter
from src.parameter_validator import ParameterValidator
from src.constants import MODEL_ALIASES


class TestBuildToolsSystemPrompt:
    """Test system prompt generation from tool definitions."""

    def test_basic_tool_definition(self):
        """Single tool definition is included in prompt."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                    },
                },
            }
        ]
        prompt = build_tools_system_prompt(tools)
        assert "get_weather" in prompt
        assert "Get weather for a location" in prompt
        assert "location" in prompt
        assert "tool_calls" in prompt  # Format instructions

    def test_multiple_tools(self):
        """Multiple tools are all included."""
        tools = [
            {"type": "function", "function": {"name": "func_a", "parameters": {}}},
            {"type": "function", "function": {"name": "func_b", "parameters": {}}},
        ]
        prompt = build_tools_system_prompt(tools)
        assert "func_a" in prompt
        assert "func_b" in prompt

    def test_tool_choice_none(self):
        """tool_choice='none' adds do-not-call instruction."""
        tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}]
        prompt = build_tools_system_prompt(tools, tool_choice="none")
        assert "Do NOT call" in prompt

    def test_tool_choice_required(self):
        """tool_choice='required' adds must-call instruction."""
        tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}]
        prompt = build_tools_system_prompt(tools, tool_choice="required")
        assert "MUST call" in prompt

    def test_tool_choice_specific(self):
        """tool_choice with specific function forces that function."""
        tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}]
        choice = {"type": "function", "function": {"name": "f"}}
        prompt = build_tools_system_prompt(tools, tool_choice=choice)
        assert "MUST call the function 'f'" in prompt

    def test_empty_tools(self):
        """Empty tools list returns empty string."""
        assert build_tools_system_prompt([]) == ""


class TestParseToolCalls:
    """Test parsing tool calls from Claude's response."""

    def test_parse_structured_format(self):
        """Parses ```tool_calls block."""
        content = '```tool_calls\n[{"name": "get_weather", "arguments": {"location": "SF"}}]\n```'
        calls, remaining = parse_tool_calls(content)
        assert calls is not None
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "get_weather"
        args = json.loads(calls[0]["function"]["arguments"])
        assert args["location"] == "SF"
        assert remaining is None

    def test_parse_with_surrounding_text(self):
        """Parses tool_calls with text before the block."""
        content = 'Let me check that.\n```tool_calls\n[{"name": "f", "arguments": {}}]\n```'
        calls, remaining = parse_tool_calls(content)
        assert calls is not None
        assert remaining == "Let me check that."

    def test_parse_multiple_calls(self):
        """Parses multiple tool calls."""
        content = '```tool_calls\n[{"name": "a", "arguments": {}}, {"name": "b", "arguments": {"x": 1}}]\n```'
        calls, remaining = parse_tool_calls(content)
        assert calls is not None
        assert len(calls) == 2
        assert calls[0]["function"]["name"] == "a"
        assert calls[1]["function"]["name"] == "b"

    def test_no_tool_calls(self):
        """Normal text returns None for calls."""
        calls, remaining = parse_tool_calls("Hello, how can I help?")
        assert calls is None
        assert remaining == "Hello, how can I help?"

    def test_empty_content(self):
        """Empty string returns None."""
        calls, remaining = parse_tool_calls("")
        assert calls is None

    def test_tool_call_ids_generated(self):
        """Each tool call gets a unique id starting with 'call_'."""
        content = '```tool_calls\n[{"name": "f", "arguments": {}}]\n```'
        calls, _ = parse_tool_calls(content)
        assert calls[0]["id"].startswith("call_")

    def test_fallback_bare_json(self):
        """Falls back to bare JSON array parsing."""
        content = '[{"name": "get_weather", "arguments": {"city": "NYC"}}]'
        calls, remaining = parse_tool_calls(content)
        assert calls is not None
        assert calls[0]["function"]["name"] == "get_weather"


class TestToolModels:
    """Test Pydantic models for function calling."""

    def test_tool_def_model(self):
        """ToolDef model validates correctly."""
        td = ToolDef(
            function=FunctionDef(
                name="test",
                description="A test function",
                parameters={"type": "object", "properties": {}},
            )
        )
        assert td.type == "function"
        assert td.function.name == "test"

    def test_tool_call_model(self):
        """ToolCall model validates correctly."""
        tc = ToolCall(
            id="call_abc123",
            function=ToolCallFunction(name="test", arguments='{"key": "value"}'),
        )
        assert tc.id == "call_abc123"
        assert tc.type == "function"
        assert tc.function.name == "test"

    def test_message_with_tool_calls(self):
        """Message model accepts tool_calls."""
        msg = Message(
            role="assistant",
            content=None,
            tool_calls=[
                ToolCall(
                    id="call_1",
                    function=ToolCallFunction(name="f", arguments="{}"),
                )
            ],
        )
        assert msg.tool_calls is not None
        assert len(msg.tool_calls) == 1

    def test_tool_role_message(self):
        """Message with role='tool' is accepted."""
        msg = Message(
            role="tool",
            content="Result data",
            tool_call_id="call_1",
            name="get_weather",
        )
        assert msg.role == "tool"
        assert msg.tool_call_id == "call_1"

    def test_chat_request_with_tools(self):
        """ChatCompletionRequest accepts tools parameter."""
        from src.models import ChatCompletionRequest

        req = ChatCompletionRequest(
            model="claude-sonnet-4-6",
            messages=[Message(role="user", content="Hi")],
            tools=[ToolDef(function=FunctionDef(name="test", parameters={"type": "object"}))],
            tool_choice="auto",
        )
        assert req.tools is not None
        assert len(req.tools) == 1
        assert req.tool_choice == "auto"


class TestToolMessageConversion:
    """Test tool role messages in MessageAdapter."""

    def test_tool_result_converted_to_user_message(self):
        """Tool result message converted to Human: message."""
        messages = [
            Message(role="user", content="What's the weather?"),
            Message(
                role="assistant",
                content=None,
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        function=ToolCallFunction(
                            name="get_weather",
                            arguments='{"location": "SF"}',
                        ),
                    )
                ],
            ),
            Message(
                role="tool",
                content='{"temp": 72, "condition": "sunny"}',
                tool_call_id="call_1",
                name="get_weather",
            ),
        ]
        prompt, _ = MessageAdapter.messages_to_prompt(messages)
        assert "Function result" in prompt
        assert "get_weather" in prompt
        assert "72" in prompt

    def test_assistant_tool_calls_in_prompt(self):
        """Assistant message with tool_calls includes call info."""
        messages = [
            Message(role="user", content="Check weather"),
            Message(
                role="assistant",
                content=None,
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        function=ToolCallFunction(
                            name="get_weather",
                            arguments='{"city": "NYC"}',
                        ),
                    )
                ],
            ),
        ]
        prompt, _ = MessageAdapter.messages_to_prompt(messages)
        assert "Called function: get_weather" in prompt
