"""
Security Configuration for AI Agent
Supports external YAML configuration and environment variables
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def load_yaml_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"

    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def parse_list_from_env(env_value: Optional[str], delimiter: str = ",") -> List[str]:
    """Parse comma-separated environment variable into list"""
    if not env_value:
        return []
    return [item.strip() for item in env_value.split(delimiter) if item.strip()]


def get_env_or_config(env_var: str, config_value: Any, cast_type: type = str) -> Any:
    """Get value from environment variable or fall back to config"""
    env_value = os.getenv(env_var)
    if env_value is not None:
        if cast_type == bool:
            return env_value.lower() in ("true", "1", "yes", "on")
        elif cast_type == int:
            try:
                return int(env_value)
            except ValueError:
                pass
        elif cast_type == list:
            return parse_list_from_env(env_value)
        elif cast_type == float:
            try:
                return float(env_value)
            except ValueError:
                pass
        return env_value
    return config_value


# Load YAML configuration
_yaml_config = load_yaml_config()

# Base directory (where script is run from)
BASE_DIR = Path(get_env_or_config("BASE_DIR", Path.cwd(), Path))

# Security Settings
_security_config = _yaml_config.get("security", {})
SECURITY = {
    "max_file_size_mb": get_env_or_config(
        "MAX_FILE_SIZE_MB", _security_config.get("max_file_size_mb", 10), int
    ),
    "allowed_extensions": get_env_or_config(
        "ALLOWED_EXTENSIONS",
        _security_config.get(
            "allowed_extensions",
            [
                ".txt",
                ".json",
                ".yaml",
                ".yml",
                ".md",
                ".csv",
                ".py",
                ".js",
                ".html",
                ".css",
                ".xml",
                ".log",
            ],
        ),
        list,
    ),
    "blocked_extensions": get_env_or_config(
        "BLOCKED_EXTENSIONS",
        _security_config.get(
            "blocked_extensions",
            [".exe", ".dll", ".so", ".dylib", ".bat", ".sh", ".ps1", ".app", ".deb", ".rpm"],
        ),
        list,
    ),
    "allowed_directories": [
        BASE_DIR,
        BASE_DIR / "workspace",
        BASE_DIR / "logs",
        BASE_DIR / "data",
    ],
    "blocked_patterns": get_env_or_config(
        "BLOCKED_PATTERNS",
        _security_config.get(
            "blocked_patterns",
            [
                "..",
                "~",
                "/etc",
                "/sys",
                "/proc",
                "C:\\Windows",
                "C:\\Program Files",
                "/bin",
                "/sbin",
                "/usr/bin",
            ],
        ),
        list,
    ),
}

# Dangerous commands that require confirmation
_dangerous_config = _yaml_config.get("dangerous_commands", [])
DANGEROUS_COMMANDS = _dangerous_config or [
    "rm",
    "del",
    "rmdir",
    "format",
    "fdisk",
    "kill",
    "shutdown",
    "reboot",
    "dd",
]

# Command whitelist (only these can run without confirmation)
_safe_config = _yaml_config.get("safe_commands", [])
SAFE_COMMANDS = _safe_config or [
    "ls",
    "dir",
    "pwd",
    "cd",
    "echo",
    "cat",
    "head",
    "tail",
    "grep",
    "find",
    "wc",
]

# Timeout Settings
_timeouts_config = _yaml_config.get("timeouts", {})
COMMAND_TIMEOUT = get_env_or_config(
    "COMMAND_TIMEOUT_SECONDS", _timeouts_config.get("command", 30), int
)
API_TIMEOUT = get_env_or_config("API_TIMEOUT_SECONDS", _timeouts_config.get("api", 120), int)
API_RETRY_TIMEOUT = get_env_or_config(
    "API_RETRY_TIMEOUT_SECONDS", _timeouts_config.get("api_retry", 10), int
)

# Conversation Settings
_conversation_config = _yaml_config.get("conversation", {})
MAX_CONVERSATION_MESSAGES = get_env_or_config(
    "MAX_CONVERSATION_MESSAGES", _conversation_config.get("max_messages", 20), int
)
MAX_TOOL_ITERATIONS = get_env_or_config(
    "MAX_TOOL_ITERATIONS", _conversation_config.get("max_tool_iterations", 5), int
)

# AI Provider Settings
AI_PROVIDERS = _yaml_config.get("ai_providers", {})

# Logging Settings
_logging_config = _yaml_config.get("logging", {})
LOG_LEVEL = get_env_or_config("LOG_LEVEL", _logging_config.get("level", "INFO")).upper()
ENABLE_AUDIT_LOGGING = get_env_or_config(
    "ENABLE_AUDIT_LOGGING", _logging_config.get("enable_audit", True), bool
)

# Audit log location
_audit_log_path = _logging_config.get("audit_log_path", "logs/audit.log")
AUDIT_LOG = BASE_DIR / _audit_log_path

# Debug mode
_app_config = _yaml_config.get("app", {})
DEBUG = get_env_or_config("DEBUG", _app_config.get("debug", False), bool)


def ensure_directories():
    """Create required directories if they don't exist"""
    # Ensure workspace exists
    (BASE_DIR / "workspace").mkdir(parents=True, exist_ok=True)

    # Ensure logs directory exists
    AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)

    # Ensure data directory exists
    (BASE_DIR / "data").mkdir(parents=True, exist_ok=True)
