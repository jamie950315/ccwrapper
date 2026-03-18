"""
Constants and configuration for Claude Code OpenAI Wrapper.

Single source of truth for tool names, models, and other configuration values.

Usage Examples:
    # Check if a model is supported
    from src.constants import CLAUDE_MODELS
    if model_name in CLAUDE_MODELS:
        # proceed with request

    # Get default allowed tools
    from src.constants import DEFAULT_ALLOWED_TOOLS
    options = {"allowed_tools": DEFAULT_ALLOWED_TOOLS}

    # Use rate limits in FastAPI
    from src.constants import RATE_LIMIT_CHAT
    @limiter.limit(f"{RATE_LIMIT_CHAT}/minute")
    async def chat_endpoint(): ...

Note:
    - Tool configurations are managed by ToolManager (see tool_manager.py)
    - Model validation uses graceful degradation (warns but allows unknown models)
    - Rate limits can be overridden via environment variables
"""

import os

# Claude Agent SDK Tool Names
# These are the built-in tools available in the Claude Agent SDK
# See: https://docs.anthropic.com/en/docs/claude-code/sdk
CLAUDE_TOOLS = [
    "Task",  # Launch agents for complex tasks
    "Bash",  # Execute bash commands
    "Glob",  # File pattern matching
    "Grep",  # Search file contents
    "Read",  # Read files
    "Edit",  # Edit files
    "Write",  # Write files
    "NotebookEdit",  # Edit Jupyter notebooks
    "WebFetch",  # Fetch web content
    "TodoWrite",  # Manage todo lists
    "WebSearch",  # Search the web
    "BashOutput",  # Get bash output
    "KillShell",  # Kill bash shells
    "Skill",  # Execute skills
    "SlashCommand",  # Execute slash commands
]

# Default tools to allow when tools are enabled
# Subset of CLAUDE_TOOLS that are safe and commonly used
DEFAULT_ALLOWED_TOOLS = [
    "Read",
    "Glob",
    "Grep",
    "Bash",
    "Write",
    "Edit",
]

# Tools to disallow by default (potentially dangerous or slow)
DEFAULT_DISALLOWED_TOOLS = [
    "Task",  # Can spawn sub-agents
    "WebFetch",  # External network access
    "WebSearch",  # External network access
]

# Claude Models
# Models supported by Claude Agent SDK (as of March 2026)
# NOTE: Claude Agent SDK only supports Claude 4+ models, not Claude 3.x
CLAUDE_MODELS = [
    # Claude 4.6 Family (Latest)
    "claude-opus-4-6",  # Most capable - best for agents and coding
    "claude-sonnet-4-6",  # Best balance of speed and intelligence
    # Claude 4.5 Family (Legacy)
    "claude-opus-4-5-20250929",
    "claude-sonnet-4-5-20250929",
    "claude-haiku-4-5-20251001",  # Fastest, near-frontier intelligence
    "claude-haiku-4-5",  # Alias for claude-haiku-4-5-20251001
    # Claude 4.1 (Legacy)
    "claude-opus-4-1-20250805",
    # Claude 4.0 Family (Legacy)
    "claude-opus-4-20250514",
    "claude-sonnet-4-20250514",
    # Claude 3.x Family - NOT SUPPORTED by Claude Agent SDK
    # These models work with Anthropic API but NOT with Claude Code
    # Uncomment only if using direct Anthropic API (not Claude Agent SDK)
    # "claude-3-7-sonnet-20250219",
    # "claude-3-5-sonnet-20241022",
    # "claude-3-5-haiku-20241022",
]

# Default model (recommended for most use cases)
# Can be overridden via DEFAULT_MODEL environment variable
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "claude-sonnet-4-6")

# Fast model (for speed/cost optimization)
FAST_MODEL = "claude-haiku-4-5-20251001"

# Model Alias Mapping: OpenAI / other model names → Claude equivalents
# Enables drop-in compatibility: clients can send gpt-4o and it maps to Claude
MODEL_ALIASES = {
    # OpenAI GPT-4o family → Sonnet (balanced)
    "gpt-4o": "claude-sonnet-4-6",
    "gpt-4o-2024-05-13": "claude-sonnet-4-6",
    "gpt-4o-2024-08-06": "claude-sonnet-4-6",
    "gpt-4o-2024-11-20": "claude-sonnet-4-6",
    "chatgpt-4o-latest": "claude-sonnet-4-6",
    # OpenAI GPT-4o-mini → Haiku (fast/cheap)
    "gpt-4o-mini": "claude-haiku-4-5",
    "gpt-4o-mini-2024-07-18": "claude-haiku-4-5",
    # OpenAI GPT-4 family → Opus (premium)
    "gpt-4": "claude-opus-4-6",
    "gpt-4-0613": "claude-opus-4-6",
    "gpt-4-turbo": "claude-sonnet-4-6",
    "gpt-4-turbo-preview": "claude-sonnet-4-6",
    "gpt-4-turbo-2024-04-09": "claude-sonnet-4-6",
    # OpenAI GPT-4.1 family (2025)
    "gpt-4.1": "claude-sonnet-4-6",
    "gpt-4.1-2025-04-14": "claude-sonnet-4-6",
    "gpt-4.1-mini": "claude-haiku-4-5",
    "gpt-4.1-mini-2025-04-14": "claude-haiku-4-5",
    "gpt-4.1-nano": "claude-haiku-4-5",
    "gpt-4.1-nano-2025-04-14": "claude-haiku-4-5",
    # OpenAI GPT-3.5 → Haiku (fast/cheap)
    "gpt-3.5-turbo": "claude-haiku-4-5",
    "gpt-3.5-turbo-16k": "claude-haiku-4-5",
    "gpt-3.5-turbo-0125": "claude-haiku-4-5",
    # OpenAI reasoning models
    "o1": "claude-opus-4-6",
    "o1-2024-12-17": "claude-opus-4-6",
    "o1-preview": "claude-opus-4-6",
    "o1-preview-2024-09-12": "claude-opus-4-6",
    "o1-mini": "claude-sonnet-4-6",
    "o1-mini-2024-09-12": "claude-sonnet-4-6",
    "o3": "claude-opus-4-6",
    "o3-mini": "claude-sonnet-4-6",
    "o4-mini": "claude-sonnet-4-6",
    # Google Gemini (some clients send these)
    "gemini-pro": "claude-sonnet-4-6",
    "gemini-1.5-pro": "claude-sonnet-4-6",
    "gemini-1.5-flash": "claude-haiku-4-5",
    "gemini-2.0-flash": "claude-haiku-4-5",
    # DeepSeek
    "deepseek-chat": "claude-sonnet-4-6",
    "deepseek-reasoner": "claude-opus-4-6",
}

# System Prompt Types
SYSTEM_PROMPT_TYPE_TEXT = "text"
SYSTEM_PROMPT_TYPE_PRESET = "preset"

# System Prompt Presets
SYSTEM_PROMPT_PRESET_CLAUDE_CODE = "claude_code"

# API Configuration
DEFAULT_MAX_TURNS = 10
DEFAULT_TIMEOUT_MS = 600000  # 10 minutes
DEFAULT_PORT = 8000

# Session Management
SESSION_CLEANUP_INTERVAL_MINUTES = 5
SESSION_MAX_AGE_MINUTES = 60

# Rate Limiting (requests per minute)
RATE_LIMIT_DEFAULT = 60
RATE_LIMIT_CHAT = 30
RATE_LIMIT_MODELS = 100
RATE_LIMIT_HEALTH = 200
