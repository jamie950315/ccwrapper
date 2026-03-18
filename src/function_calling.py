"""
OpenAI Function Calling simulation for Claude.

Translates OpenAI tool definitions → system prompt instructions, and
parses Claude's response to extract tool_calls in OpenAI format.

Since the Claude Agent SDK doesn't support custom tool definitions natively,
we simulate function calling by:
1. Injecting tool schemas into the system prompt
2. Asking Claude to respond in a structured format when calling tools
3. Parsing the structured response into OpenAI tool_calls format
"""

import json
import re
import uuid
import logging
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

# The format Claude should use when it wants to call a tool
_TOOL_CALL_FORMAT = """When you want to call a function, respond with ONLY a JSON object in this exact format (no other text):
```tool_calls
[
  {
    "name": "<function_name>",
    "arguments": { ... }
  }
]
```

Rules:
- You may call one or multiple functions in a single response.
- If you want to call a function, output ONLY the tool_calls block above — no other text.
- If you do NOT need to call a function, respond normally with text.
- The "arguments" field must be a valid JSON object matching the function's parameters schema."""

# Regex to extract tool_calls blocks from Claude's response
_TOOL_CALLS_PATTERN = re.compile(r"```tool_calls\s*\n(.*?)\n\s*```", re.DOTALL)
# Fallback: bare JSON array starting with [{"name":
_TOOL_CALLS_FALLBACK = re.compile(r'\[\s*\{\s*"name"\s*:', re.DOTALL)


def build_tools_system_prompt(tools: List[Dict[str, Any]], tool_choice: Any = "auto") -> str:
    """Convert OpenAI tool definitions to a system prompt section.

    Args:
        tools: OpenAI-format tool definitions
        tool_choice: "auto", "none", "required", or {"type":"function","function":{"name":"..."}}
    """
    if not tools:
        return ""

    lines = ["# Available Functions\n"]
    lines.append("You have access to the following functions:\n")

    for tool in tools:
        if tool.get("type") != "function":
            continue
        func = tool.get("function", {})
        name = func.get("name", "unknown")
        desc = func.get("description", "")
        params = func.get("parameters", {})

        lines.append(f"## {name}")
        if desc:
            lines.append(f"{desc}\n")
        lines.append("Parameters:")
        lines.append(f"```json\n{json.dumps(params, indent=2)}\n```\n")

    lines.append(_TOOL_CALL_FORMAT)

    # Add tool_choice guidance
    if tool_choice == "none":
        lines.append("\nIMPORTANT: Do NOT call any functions. Respond with text only.")
    elif tool_choice == "required":
        lines.append("\nIMPORTANT: You MUST call at least one function in your response.")
    elif isinstance(tool_choice, dict):
        forced_name = tool_choice.get("function", {}).get("name")
        if forced_name:
            lines.append(f"\nIMPORTANT: You MUST call the function '{forced_name}'.")

    return "\n".join(lines)


def parse_tool_calls(content: str) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
    """Parse Claude's response for tool call blocks.

    Returns:
        (tool_calls, remaining_text)
        - tool_calls: list of OpenAI-format tool_call dicts, or None if no calls found
        - remaining_text: any text outside the tool_calls block, or None
    """
    if not content:
        return None, content

    # Try structured format first: ```tool_calls ... ```
    match = _TOOL_CALLS_PATTERN.search(content)
    if match:
        try:
            calls_data = json.loads(match.group(1))
            if isinstance(calls_data, list):
                tool_calls = _format_tool_calls(calls_data)
                # Extract text outside the block
                remaining = content[: match.start()].strip()
                if remaining:
                    return tool_calls, remaining
                return tool_calls, None
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Failed to parse tool_calls block: {e}")

    # Fallback: try to find raw JSON array with tool calls
    fallback_match = _TOOL_CALLS_FALLBACK.search(content)
    if fallback_match:
        # Try to extract from the match position to the end of the array
        json_start = fallback_match.start()
        try:
            # Find the matching closing bracket
            bracket_depth = 0
            for i in range(json_start, len(content)):
                if content[i] == "[":
                    bracket_depth += 1
                elif content[i] == "]":
                    bracket_depth -= 1
                    if bracket_depth == 0:
                        candidate = content[json_start : i + 1]
                        calls_data = json.loads(candidate)
                        if isinstance(calls_data, list) and all(
                            isinstance(c, dict) and "name" in c for c in calls_data
                        ):
                            tool_calls = _format_tool_calls(calls_data)
                            remaining = content[:json_start].strip()
                            if remaining:
                                return tool_calls, remaining
                            return tool_calls, None
                        break
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

    return None, content


def _format_tool_calls(calls_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Format parsed tool call data into OpenAI tool_calls format."""
    tool_calls = []
    for call in calls_data:
        name = call.get("name", "")
        arguments = call.get("arguments", {})
        tool_calls.append(
            {
                "id": f"call_{uuid.uuid4().hex[:12]}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": (
                        json.dumps(arguments) if isinstance(arguments, dict) else str(arguments)
                    ),
                },
            }
        )
    return tool_calls


def convert_tool_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert messages containing tool role and tool_calls to Claude-compatible format.

    OpenAI format has:
    - assistant messages with tool_calls array
    - tool role messages with tool_call_id and content

    We convert these to plain text messages Claude can understand.
    """
    converted = []
    for msg in messages:
        role = msg.get("role", "")

        if role == "assistant" and "tool_calls" in msg:
            # Convert tool_calls to text showing what was called
            tc = msg["tool_calls"]
            text_parts = []
            if msg.get("content"):
                text_parts.append(msg["content"])
            for call in tc:
                func = call.get("function", {})
                name = func.get("name", "unknown")
                args = func.get("arguments", "{}")
                text_parts.append(f"[Called function: {name}({args})]")
            converted.append({"role": "assistant", "content": "\n".join(text_parts)})

        elif role == "tool":
            # Convert tool result to user message
            tool_call_id = msg.get("tool_call_id", "unknown")
            content = msg.get("content", "")
            name = msg.get("name", "")
            label = (
                f"[Function result for {name or tool_call_id}]" if name else f"[Function result]"
            )
            converted.append({"role": "user", "content": f"{label}\n{content}"})

        else:
            converted.append(msg)

    return converted
