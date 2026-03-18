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
                conversation_parts.append(f"Assistant: {prefix}{message.content}")

        # Join conversation parts
        prompt = "\n\n".join(conversation_parts)

        # If the last message wasn't from the user, add a prompt for assistant
        if messages and messages[-1].role != "user":
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
        """Strip markdown code fences from JSON responses."""
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        return content.strip()


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
