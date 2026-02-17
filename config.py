"""
Security Configuration for AI Agent
"""
import os
from pathlib import Path

# Base directory (where script is run from)
BASE_DIR = Path.cwd()

# Security Settings
SECURITY = {
    "max_file_size_mb": 10,  # Maximum file size for operations
    "allowed_extensions": [
        ".txt", ".json", ".yaml", ".yml", ".md", ".csv", 
        ".py", ".js", ".html", ".css", ".xml", ".log"
    ],
    "blocked_extensions": [
        ".exe", ".dll", ".so", ".dylib", ".bat", ".sh", 
        ".ps1", ".app", ".deb", ".rpm"
    ],
    "allowed_directories": [
        BASE_DIR,                # Allow access to project root
        BASE_DIR / "workspace",  # Keep explicit workspace reference
        BASE_DIR / "logs",
        BASE_DIR / "data",
    ],
    "blocked_patterns": [
        "..", "~", "/etc", "/sys", "/proc", "C:\\Windows",
        "C:\\Program Files", "/bin", "/sbin", "/usr/bin"
    ],
}

# Dangerous commands that require confirmation
DANGEROUS_COMMANDS = [
    "rm", "del", "rmdir", "format", "fdisk", 
    "kill", "shutdown", "reboot", "dd"
]

# Command whitelist (only these can run without confirmation)
SAFE_COMMANDS = [
    "ls", "dir", "pwd", "cd", "echo", "cat", 
    "head", "tail", "grep", "find", "wc"
]

# AI Provider Settings
AI_PROVIDERS = {
    "openai": {
        "supports_functions": True,
        "models": ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"],
    },
    "anthropic": {
        "supports_functions": True,
        "models": ["claude-3-5-sonnet-20241022", "claude-3-opus-20240229"],
    },
    "groq": {
        "supports_functions": True,
        "models": ["llama3-groq-70b-8192-tool-use-preview"],
    },
}

# Audit log location
AUDIT_LOG = BASE_DIR / "logs" / "audit.log"

def ensure_directories():
    """Create required directories if they don't exist"""
    # Only ensure workspace exists by default to keep root clean
    (BASE_DIR / "workspace").mkdir(parents=True, exist_ok=True)

    pass
