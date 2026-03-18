from typing import List, Optional, Dict, Any
from src.models import Message
import re


class MessageAdapter:
    """Converts between OpenAI message format and Claude Code prompts."""

    @staticmethod
    def messages_to_prompt(messages: List[Message]) -> tuple[str, Optional[str]]:
        """
        Convert OpenAI messages to Claude Code prompt format.
        Returns (prompt, system_prompt)
        """
        system_prompt = None
        conversation_parts = []

        for message in messages:
            if message.role == "system":
                # Use the last system message as the system prompt
                system_prompt = message.content
            elif message.role == "user":
                prefix = f"[{message.name}] " if message.name else ""
                conversation_parts.append(f"Human: {prefix}{message.content}")
            elif message.role == "assistant":
                prefix = f"[{message.name}] " if message.name else ""
                parts = []
                if message.content:
                    parts.append(f"{prefix}{message.content}")
                # Include tool_calls as text for Claude's context
                if hasattr(message, "tool_calls") and message.tool_calls:
                    for tc in message.tool_calls:
                        func = tc.function if hasattr(tc, "function") else tc.get("function", {})
                        name = func.name if hasattr(func, "name") else func.get("name", "")
                        args = (
                            func.arguments
                            if hasattr(func, "arguments")
                            else func.get("arguments", "")
                        )
                        parts.append(f"[Called function: {name}({args})]")
                conversation_parts.append(f"Assistant: {''.join(parts) if parts else ''}")
            elif message.role == "tool":
                # Tool result → user message for Claude
                content = message.content or ""
                label = (
                    f"[Function result for {message.name}]" if message.name else "[Function result]"
                )
                conversation_parts.append(f"Human: {label}\n{content}")

        # Join conversation parts
        prompt = "\n\n".join(conversation_parts)

        # If the last message wasn't from the user or tool, add a prompt for assistant
        if messages and messages[-1].role not in ("user", "tool"):
            prompt += "\n\nHuman: Please continue."

        return prompt, system_prompt

    @staticmethod
    def filter_content(content: str) -> str:
        """
        Filter content for unsupported features and tool usage.
        Remove thinking blocks, tool calls, and image references.
        """
        if not content:
            return content

        # Remove thinking blocks (common when tools are disabled but Claude tries to think)
        thinking_pattern = r"<thinking>.*?</thinking>"
        content = re.sub(thinking_pattern, "", content, flags=re.DOTALL)

        # Extract content from attempt_completion blocks (these contain the actual user response)
        attempt_completion_pattern = r"<attempt_completion>(.*?)</attempt_completion>"
        attempt_matches = re.findall(attempt_completion_pattern, content, flags=re.DOTALL)
        if attempt_matches:
            # Use the content from the attempt_completion block
            extracted_content = attempt_matches[0].strip()

            # If there's a <result> tag inside, extract from that
            result_pattern = r"<result>(.*?)</result>"
            result_matches = re.findall(result_pattern, extracted_content, flags=re.DOTALL)
            if result_matches:
                extracted_content = result_matches[0].strip()

            if extracted_content:
                content = extracted_content
        else:
            # Remove other tool usage blocks (when tools are disabled but Claude tries to use them)
            tool_patterns = [
                r"<read_file>.*?</read_file>",
                r"<write_file>.*?</write_file>",
                r"<bash>.*?</bash>",
                r"<search_files>.*?</search_files>",
                r"<str_replace_editor>.*?</str_replace_editor>",
                r"<args>.*?</args>",
                r"<ask_followup_question>.*?</ask_followup_question>",
                r"<attempt_completion>.*?</attempt_completion>",
                r"<question>.*?</question>",
                r"<follow_up>.*?</follow_up>",
                r"<suggest>.*?</suggest>",
            ]

            for pattern in tool_patterns:
                content = re.sub(pattern, "", content, flags=re.DOTALL)

        # Pattern to match image references or base64 data
        image_pattern = r"\[Image:.*?\]|data:image/.*?;base64,.*?(?=\s|$)"

        def replace_image(match):
            return "[Image: Content not supported by Claude Code]"

        content = re.sub(image_pattern, replace_image, content)

        # Clean up extra whitespace and newlines
        content = re.sub(r"\n\s*\n\s*\n", "\n\n", content)  # Multiple newlines to double
        content = content.strip()

        # Return empty string if content was fully stripped; callers handle this.
        if not content or content.isspace():
            return ""

        return content

    @staticmethod
    def format_claude_response(
        content: str, model: str, finish_reason: str = "stop"
    ) -> Dict[str, Any]:
        """Format Claude response for OpenAI compatibility."""
        return {
            "role": "assistant",
            "content": content,
            "finish_reason": finish_reason,
            "model": model,
        }

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Estimate token count with CJK awareness.

        English: ~4 chars per token.  CJK: ~1.5 chars per token.
        """
        if not text:
            return 0
        cjk = sum(
            1
            for c in text
            if "\u4e00" <= c <= "\u9fff" or "\u3400" <= c <= "\u4dbf" or "\uf900" <= c <= "\ufaff"
        )
        ascii_chars = len(text) - cjk
        return max(1, int(cjk / 1.5 + ascii_chars / 4))

    @staticmethod
    def clean_json_response(content: str) -> str:
        """Strip markdown code fences from JSON responses and validate."""
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        # Validate JSON; if invalid, try to extract JSON object/array
        import json as _json

        try:
            _json.loads(content)
        except (ValueError, TypeError):
            # Try to find a JSON object or array in the text
            for start_char, end_char in [("{", "}"), ("[", "]")]:
                start = content.find(start_char)
                end = content.rfind(end_char)
                if start != -1 and end > start:
                    candidate = content[start : end + 1]
                    try:
                        _json.loads(candidate)
                        return candidate
                    except (ValueError, TypeError):
                        pass
        return content


class JsonFenceStripper:
    """Strips markdown code fences from streaming JSON responses.

    Buffers the first few characters to detect and remove ``json\\n or ```\\n
    at the start, and holds back the last 4 characters to strip trailing
    \\n``` at the end.  Call flush() after the last delta.
    """

    _OPEN_FENCES = ("```json\n", "```json\r\n", "```\n", "```\r\n")
    _CLOSE_FENCE = "\n```"
    _MAX_OPEN_LEN = 10  # max("```json\r\n") = 10
    _CLOSE_LEN = 4  # len("\n```")

    def __init__(self):
        self._head_buffer = ""
        self._head_stripped = False
        self._tail_buffer = ""

    def process_delta(self, text: str) -> str:
        """Process a streaming delta. Returns text safe to emit."""
        # Phase 1: detect and strip opening fence
        if not self._head_stripped:
            self._head_buffer += text
            # Check if buffer matches any fence exactly (or starts with one)
            for fence in self._OPEN_FENCES:
                if self._head_buffer.startswith(fence):
                    self._head_stripped = True
                    self._head_buffer = self._head_buffer[len(fence) :]
                    text = self._head_buffer
                    self._head_buffer = ""
                    break
            if not self._head_stripped:
                if len(self._head_buffer) >= self._MAX_OPEN_LEN:
                    # Exceeded max fence length with no match — not a fence
                    self._head_stripped = True
                    text = self._head_buffer
                    self._head_buffer = ""
                else:
                    # Check if buffer could still match a fence prefix
                    could_match = any(f.startswith(self._head_buffer) for f in self._OPEN_FENCES)
                    if not could_match:
                        self._head_stripped = True
                        text = self._head_buffer
                        self._head_buffer = ""
                    else:
                        return ""  # Still accumulating

        # Phase 2: hold back last _CLOSE_LEN chars for closing fence
        self._tail_buffer += text
        if len(self._tail_buffer) > self._CLOSE_LEN:
            emit = self._tail_buffer[: -self._CLOSE_LEN]
            self._tail_buffer = self._tail_buffer[-self._CLOSE_LEN :]
            return emit
        return ""

    def flush(self) -> str:
        """Flush remaining buffer. Call at end of stream."""
        remaining = self._head_buffer + self._tail_buffer
        self._head_buffer = ""
        self._tail_buffer = ""
        self._head_stripped = True
        # Strip closing fence if present
        if remaining.endswith("```"):
            remaining = remaining[:-3]
            if remaining.endswith("\n"):
                remaining = remaining[:-1]
        return remaining


class StopSequenceProcessor:
    """Post-processes streaming/non-streaming output to apply stop sequences.

    For streaming: call process_delta() per chunk, flush() at end.
    Uses a hold-back buffer of (max_stop_len - 1) chars to catch
    stop sequences that span chunk boundaries.

    For non-streaming: use the static truncate() method.
    """

    def __init__(self, stop_sequences: List[str]):
        self.stop_sequences = [s for s in stop_sequences if s]
        self._holdback_size = max((len(s) for s in self.stop_sequences), default=0)
        if self._holdback_size > 0:
            self._holdback_size -= 1
        self._buffer = ""
        self._stopped = False

    @property
    def stopped(self) -> bool:
        return self._stopped

    def process_delta(self, text: str) -> str:
        """Process a streaming text delta.

        Returns text safe to emit. May return "" if buffering.
        Check .stopped after calling to see if a stop sequence was hit.
        """
        if self._stopped or not self.stop_sequences:
            return "" if self._stopped else text

        self._buffer += text

        # Find the earliest stop sequence occurrence in the buffer
        earliest_idx = -1
        for seq in self.stop_sequences:
            idx = self._buffer.find(seq)
            if idx != -1 and (earliest_idx == -1 or idx < earliest_idx):
                earliest_idx = idx
        if earliest_idx != -1:
            self._stopped = True
            emit = self._buffer[:earliest_idx]
            self._buffer = ""
            return emit

        # No stop found — emit the safe portion (before holdback window)
        if len(self._buffer) > self._holdback_size:
            if self._holdback_size > 0:
                emit = self._buffer[: -self._holdback_size]
                self._buffer = self._buffer[-self._holdback_size :]
            else:
                emit = self._buffer
                self._buffer = ""
            return emit

        return ""  # Everything held back

    def flush(self) -> str:
        """Flush remaining buffer at end of stream. Check .stopped after."""
        if self._stopped:
            return ""
        remaining = self._buffer
        self._buffer = ""
        for seq in self.stop_sequences:
            idx = remaining.find(seq)
            if idx != -1:
                self._stopped = True
                return remaining[:idx]
        return remaining

    @staticmethod
    def truncate(content: str, stop_sequences: List[str]) -> tuple:
        """Truncate content at first stop sequence (non-streaming).

        Returns (truncated_content, was_truncated).
        """
        if not stop_sequences:
            return content, False
        earliest_idx = len(content)
        found = False
        for seq in stop_sequences:
            if not seq:
                continue
            idx = content.find(seq)
            if idx != -1 and idx < earliest_idx:
                earliest_idx = idx
                found = True
        if found:
            return content[:earliest_idx], True
        return content, False
