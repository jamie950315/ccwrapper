import os
import asyncio
import signal
import time
import uuid
import tempfile
import atexit
import shutil
from typing import AsyncGenerator, Dict, Any, Optional, List
from pathlib import Path
import logging

from claude_agent_sdk import query, ClaudeAgentOptions, ClaudeSDKClient

logger = logging.getLogger(__name__)

# Default system prompt used when no custom prompt is provided
_DEFAULT_SYSTEM_PROMPT = "You are a helpful AI assistant."

# Limit concurrent SDK queries to prevent memory exhaustion.
MAX_CONCURRENT_QUERIES = int(os.getenv("MAX_CONCURRENT_QUERIES", "2"))
_query_semaphore = asyncio.Semaphore(MAX_CONCURRENT_QUERIES)


def _to_dict(message: Any) -> Dict[str, Any]:
    """Convert an SDK message object to a dict for consistent handling."""
    if isinstance(message, dict):
        return message
    if hasattr(message, "__dict__"):
        return {k: v for k, v in vars(message).items() if not k.startswith("_")}
    return message


class ClaudeCodeCLI:
    def __init__(self, timeout: int = 600000, cwd: Optional[str] = None):
        self.timeout = timeout / 1000  # Convert ms to seconds
        self.temp_dir = None

        # If cwd is provided (from CLAUDE_CWD env var), use it
        # Otherwise create an isolated temp directory
        if cwd:
            self.cwd = Path(cwd)
            if not self.cwd.exists():
                logger.error(f"ERROR: Specified working directory does not exist: {self.cwd}")
                logger.error(
                    "Please create the directory first or unset CLAUDE_CWD to use a temporary directory"
                )
                raise ValueError(f"Working directory does not exist: {self.cwd}")
            else:
                logger.info(f"Using CLAUDE_CWD: {self.cwd}")
        else:
            self.temp_dir = tempfile.mkdtemp(prefix="claude_code_workspace_")
            self.cwd = Path(self.temp_dir)
            logger.info(f"Using temporary isolated workspace: {self.cwd}")
            atexit.register(self._cleanup_temp_dir)

        # Import auth manager
        from src.auth import auth_manager, validate_claude_code_auth

        # Validate authentication
        is_valid, auth_info = validate_claude_code_auth()
        if not is_valid:
            logger.warning(f"Claude Code authentication issues detected: {auth_info['errors']}")
        else:
            logger.info(f"Claude Code authentication method: {auth_info.get('method', 'unknown')}")

        # Set auth environment variables once for the process lifetime.
        # This avoids per-request env manipulation which is not thread-safe.
        self.claude_env_vars = auth_manager.get_claude_code_env_vars()
        for key, value in self.claude_env_vars.items():
            os.environ[key] = value

        # Persistent SDK client — one subprocess reused across all requests.
        self._client: Optional[ClaudeSDKClient] = None
        self._client_lock = asyncio.Lock()
        self._client_request_count = 0
        # Recycle the persistent client every N requests to prevent subprocess memory growth.
        self._client_max_requests = int(os.getenv("CLIENT_RECYCLE_REQUESTS", "200"))
        # Flag: recycle client before next request for session isolation.
        self._needs_recycle = False

    def _get_client_pid(self) -> Optional[int]:
        """Extract subprocess PID from the SDK client (best-effort)."""
        try:
            transport = getattr(self._client, "_transport", None)
            if transport is None:
                query = getattr(self._client, "_query", None)
                if query:
                    transport = getattr(query, "transport", None)
            if transport:
                proc = getattr(transport, "_process", None)
                if proc and hasattr(proc, "pid"):
                    return proc.pid
        except Exception:
            pass
        return None

    async def _kill_client(self):
        """Disconnect client and ensure subprocess is dead.

        The SDK's disconnect() chain has suppress(Exception) calls that can
        silently fail to terminate the subprocess. We track the PID and
        send SIGKILL as a fallback.
        """
        if self._client is None:
            return
        pid = self._get_client_pid()
        try:
            await self._client.disconnect()
        except Exception:
            pass
        self._client = None
        # Fallback: ensure subprocess is dead
        if pid:
            try:
                os.kill(pid, signal.SIGKILL)
                logger.debug(f"SIGKILL sent to subprocess {pid}")
            except ProcessLookupError:
                pass  # Already dead — good
            except Exception as e:
                logger.debug(f"Failed to kill subprocess {pid}: {e}")

    async def _ensure_client(self) -> ClaudeSDKClient:
        """Get or create the persistent ClaudeSDKClient.

        Automatically recycles the client after _client_max_requests to
        prevent subprocess memory growth over long uptimes.
        """
        async with self._client_lock:
            # Recycle if flagged for session isolation or request count exceeded
            if self._client is not None and (
                self._needs_recycle
                or self._client_request_count >= self._client_max_requests
            ):
                reason = "session isolation" if self._needs_recycle else f"{self._client_request_count} requests"
                logger.info(f"Recycling persistent client ({reason})")
                await self._kill_client()
                self._client_request_count = 0
                self._needs_recycle = False

            if self._client is None:
                logger.info("Creating persistent ClaudeSDKClient...")
                opts = ClaudeAgentOptions(
                    max_turns=1,
                    cwd=self.cwd,
                    allowed_tools=[],
                    include_partial_messages=True,
                    system_prompt={"type": "text", "text": _DEFAULT_SYSTEM_PROMPT},
                )
                self._client = ClaudeSDKClient(options=opts)
                await self._client.connect()
                self._client_request_count = 0
                logger.info("✅ Persistent ClaudeSDKClient connected (~400MB, reused)")
            return self._client

    async def _reset_client(self):
        """Reset client on error so next request creates a fresh one."""
        async with self._client_lock:
            if self._client is not None:
                await self._kill_client()
                self._client_request_count = 0
                logger.info("Persistent client reset")

    async def shutdown(self):
        """Disconnect persistent client on server shutdown."""
        await self._reset_client()
        logger.info("ClaudeCodeCLI shutdown complete")

    def _can_use_persistent_client(
        self,
        max_turns: int,
        allowed_tools: Optional[List[str]],
        disallowed_tools: Optional[List[str]],
        permission_mode: Optional[str],
    ) -> bool:
        """Check if request can use the persistent client (fast path).

        The persistent client is configured with allowed_tools=[], max_turns=1.
        Custom system prompts are supported by embedding them into the user
        prompt (see _run_via_client), so they don't block the fast path.

        We allow allowed_tools=None here because main.py always passes []
        explicitly for the default (no-tools) path; None would only arrive
        from direct callers that didn't specify tools, which is compatible
        with the client's empty-tools configuration.
        """
        if max_turns != 1:
            return False
        if allowed_tools is not None and allowed_tools != []:
            return False
        if disallowed_tools is not None:
            return False
        if permission_mode is not None:
            return False
        return True

    async def verify_cli(self) -> bool:
        """Verify Claude Agent SDK is working and authenticated.

        Also establishes the persistent client for subsequent requests.
        """
        try:
            logger.info("Testing Claude Agent SDK...")
            client = await self._ensure_client()

            sid = uuid.uuid4().hex[:8]
            await client.query("Hello", session_id=sid)

            got_response = False
            async for message in client.receive_messages():
                name = type(message).__name__
                if name in ("AssistantMessage", "StreamEvent"):
                    got_response = True
                if name == "ResultMessage":
                    break

            if got_response:
                logger.info("✅ Claude Agent SDK verified successfully")
                return True
            else:
                logger.warning("⚠️ Claude Agent SDK test returned no messages")
                return False

        except Exception as e:
            logger.error(f"Claude Agent SDK verification failed: {e}")
            await self._reset_client()
            logger.warning("Please ensure Claude Code is installed and authenticated:")
            logger.warning("  1. Install: npm install -g @anthropic-ai/claude-code")
            logger.warning("  2. Set ANTHROPIC_API_KEY environment variable")
            logger.warning("  3. Test: claude --print 'Hello'")
            return False

    async def run_completion(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        stream: bool = True,
        max_turns: int = 10,
        allowed_tools: Optional[List[str]] = None,
        disallowed_tools: Optional[List[str]] = None,
        session_id: Optional[str] = None,
        continue_session: bool = False,
        permission_mode: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Run Claude Agent using the Python SDK and yield response chunks."""

        try:
            use_persistent = self._can_use_persistent_client(
                max_turns, allowed_tools, disallowed_tools, permission_mode,
            )

            async with _query_semaphore:
                if use_persistent and not session_id and not continue_session:
                    async for msg in self._run_via_client(prompt, system_prompt):
                        yield msg
                else:
                    async for msg in self._run_via_query(
                        prompt, system_prompt, model, stream, max_turns,
                        allowed_tools, disallowed_tools, session_id,
                        continue_session, permission_mode,
                    ):
                        yield msg

        except Exception as e:
            logger.error(f"Claude Agent SDK error: {e}")
            yield {
                "type": "result",
                "subtype": "error_during_execution",
                "is_error": True,
                "error_message": str(e),
            }

    def _get_project_dir(self) -> Path:
        """Get Claude Code's project directory for the current workspace.

        Claude Code stores session history in ~/.claude/projects/<encoded-cwd>/
        where <encoded-cwd> is the cwd path with '/' replaced by '-'.
        """
        encoded = str(self.cwd).replace("/", "-")
        return Path.home() / ".claude" / "projects" / encoded

    def _cleanup_session_files(self):
        """Remove session .jsonl files to prevent cross-session context leakage.

        Claude Code persists conversation history as .jsonl files in the
        project directory. Without cleanup, the subprocess may inject
        previous conversation context into new sessions.
        """
        project_dir = self._get_project_dir()
        if not project_dir.exists():
            return
        removed = 0
        for f in project_dir.glob("*.jsonl"):
            try:
                f.unlink()
                removed += 1
            except Exception:
                pass
        if removed:
            logger.debug(f"Cleaned up {removed} session file(s) from {project_dir}")

    async def _run_via_client(
        self, prompt: str, system_prompt: Optional[str] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Fast path: reuse persistent client with a fresh session_id per request.

        When a custom system_prompt is provided, it is embedded at the top of
        the user prompt so the persistent client (which has a fixed system
        prompt) can still honour per-request system instructions without
        falling back to the slow query() path.
        """
        try:
            client = await self._ensure_client()
            self._client_request_count += 1
            sid = uuid.uuid4().hex[:8]

            # Embed system prompt into user prompt for the persistent client
            if system_prompt and system_prompt != _DEFAULT_SYSTEM_PROMPT:
                prompt = f"<system_instructions>\n{system_prompt}\n</system_instructions>\n\n{prompt}"

            await client.query(prompt, session_id=sid)

            deadline = time.monotonic() + self.timeout
            async for message in client.receive_messages():
                if time.monotonic() > deadline:
                    logger.error(f"Timeout after {self.timeout}s waiting for SDK response")
                    await self._reset_client()
                    yield {
                        "type": "result",
                        "subtype": "error_during_execution",
                        "is_error": True,
                        "error_message": f"Request timed out after {self.timeout}s",
                    }
                    return
                logger.debug(f"Raw SDK message type: {type(message)}")
                yield _to_dict(message)
                if type(message).__name__ == "ResultMessage":
                    self._cleanup_session_files()
                    # Flag client for recycle on next request — the subprocess
                    # retains in-memory context across session_ids.
                    self._needs_recycle = True
                    break

        except Exception as e:
            logger.warning(f"Persistent client error, resetting: {e}")
            await self._reset_client()
            yield {
                "type": "result",
                "subtype": "error_during_execution",
                "is_error": True,
                "error_message": f"Persistent client error (will auto-recover on next request): {e}",
            }

    async def _run_via_query(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        stream: bool = True,
        max_turns: int = 1,
        allowed_tools: Optional[List[str]] = None,
        disallowed_tools: Optional[List[str]] = None,
        session_id: Optional[str] = None,
        continue_session: bool = False,
        permission_mode: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Slow path: stateless query() with full option flexibility."""
        options = ClaudeAgentOptions(
            max_turns=max_turns,
            cwd=self.cwd,
            include_partial_messages=stream,
        )

        if model:
            options.model = model

        if system_prompt:
            options.system_prompt = {"type": "text", "text": system_prompt}
        else:
            options.system_prompt = {"type": "text", "text": _DEFAULT_SYSTEM_PROMPT}

        if allowed_tools is not None:
            options.allowed_tools = allowed_tools
        if disallowed_tools is not None:
            options.disallowed_tools = disallowed_tools

        if permission_mode:
            options.permission_mode = permission_mode

        if continue_session:
            options.continue_session = True
        elif session_id:
            options.resume = session_id

        deadline = time.monotonic() + self.timeout
        async for message in query(prompt=prompt, options=options):
            if time.monotonic() > deadline:
                logger.error(f"Timeout after {self.timeout}s in query()")
                yield {
                    "type": "result",
                    "subtype": "error_during_execution",
                    "is_error": True,
                    "error_message": f"Request timed out after {self.timeout}s",
                }
                return
            logger.debug(f"Raw SDK message type: {type(message)}")
            yield _to_dict(message)

    def parse_claude_message(self, messages: List[Dict[str, Any]]) -> Optional[str]:
        """Extract the assistant message from Claude Agent SDK messages."""
        for message in messages:
            if message.get("subtype") == "success" and "result" in message:
                return message["result"]

        last_text = None
        for message in messages:
            if "content" in message and isinstance(message["content"], list):
                text_parts = []
                for block in message["content"]:
                    if hasattr(block, "text"):
                        text_parts.append(block.text)
                    elif isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif isinstance(block, str):
                        text_parts.append(block)
                if text_parts:
                    last_text = "\n".join(text_parts)

            elif message.get("type") == "assistant" and "message" in message:
                sdk_message = message["message"]
                if isinstance(sdk_message, dict) and "content" in sdk_message:
                    content = sdk_message["content"]
                    if isinstance(content, list) and len(content) > 0:
                        text_parts = []
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                text_parts.append(block.get("text", ""))
                        if text_parts:
                            last_text = "\n".join(text_parts)
                    elif isinstance(content, str):
                        last_text = content

        return last_text

    def extract_metadata(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract metadata like costs, tokens, and session info from SDK messages."""
        metadata = {
            "session_id": None,
            "total_cost_usd": 0.0,
            "duration_ms": 0,
            "num_turns": 0,
            "model": None,
        }

        for message in messages:
            if message.get("subtype") == "success" and "total_cost_usd" in message:
                metadata.update(
                    {
                        "total_cost_usd": message.get("total_cost_usd", 0.0),
                        "duration_ms": message.get("duration_ms", 0),
                        "num_turns": message.get("num_turns", 0),
                        "session_id": message.get("session_id"),
                    }
                )
            elif message.get("subtype") == "init" and "data" in message:
                data = message["data"]
                metadata.update({"session_id": data.get("session_id"), "model": data.get("model")})
            elif message.get("type") == "result":
                metadata.update(
                    {
                        "total_cost_usd": message.get("total_cost_usd", 0.0),
                        "duration_ms": message.get("duration_ms", 0),
                        "num_turns": message.get("num_turns", 0),
                        "session_id": message.get("session_id"),
                    }
                )
            elif message.get("type") == "system" and message.get("subtype") == "init":
                metadata.update(
                    {"session_id": message.get("session_id"), "model": message.get("model")}
                )

        return metadata

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Estimate token count with CJK awareness.

        English: ~4 chars per token.  CJK: ~1.5 chars per token.
        """
        if not text:
            return 1
        cjk = sum(1 for c in text if "\u4e00" <= c <= "\u9fff"
                   or "\u3400" <= c <= "\u4dbf" or "\uf900" <= c <= "\ufaff")
        ascii_chars = len(text) - cjk
        return max(1, int(cjk / 1.5 + ascii_chars / 4))

    def estimate_token_usage(
        self, prompt: str, completion: str, model: Optional[str] = None
    ) -> Dict[str, int]:
        """Estimate token usage with CJK-aware heuristic."""
        prompt_tokens = self._estimate_tokens(prompt)
        completion_tokens = self._estimate_tokens(completion)
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

    def _cleanup_temp_dir(self):
        """Clean up temporary directory on exit."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
                logger.info(f"Cleaned up temporary workspace: {self.temp_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean up temp directory {self.temp_dir}: {e}")
