"""
Secure Tool Implementations for AI Agent
"""
import os
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import hashlib

from config import SECURITY, DANGEROUS_COMMANDS, SAFE_COMMANDS, AUDIT_LOG, BASE_DIR


class SecurityError(Exception):
    """Raised when a security violation is detected"""
    pass


class AuditLogger:
    """Fast, thread-safe audit logger"""

    @staticmethod
    def log(action: str, details: Dict, status: str = "success"):
        """Log actions with timestamp"""
        try:
            entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "action": action,
                "details": details,
                "status": status,
            }
            with open(AUDIT_LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            pass  # Don't break execution on logging failure


class SecureFileOps:
    """Secure file operations with validation"""

    @staticmethod
    def validate_path(path: str) -> Path:
        """Validate and resolve path safely"""
        try:
            # Convert to Path object and resolve
            resolved = Path(path).resolve()

            # Check if path is within allowed directories
            allowed = False
            for allowed_dir in SECURITY["allowed_directories"]:
                try:
                    resolved.relative_to(allowed_dir.resolve())
                    allowed = True
                    break
                except ValueError:
                    continue

            if not allowed:
                raise SecurityError(
                    f"Access denied: Path must be within workspace directories"
                )

            # Check for blocked patterns
            path_str = str(resolved)
            for pattern in SECURITY["blocked_patterns"]:
                if pattern in path_str:
                    raise SecurityError(f"Blocked pattern detected: {pattern}")

            return resolved

        except Exception as e:
            raise SecurityError(f"Invalid path: {e}")

    @staticmethod
    def validate_extension(path: Path, operation: str = "read") -> None:
        """Check file extension against whitelist/blacklist"""
        ext = path.suffix.lower()

        if ext in SECURITY["blocked_extensions"]:
            raise SecurityError(f"Blocked file type: {ext}")

        if operation in ("write", "create") and ext not in SECURITY["allowed_extensions"]:
            if ext:  # Has extension but not in allowed list
                raise SecurityError(
                    f"File type {ext} not in allowed list. "
                    f"Allowed: {', '.join(SECURITY['allowed_extensions'])}"
                )

    @staticmethod
    def check_file_size(path: Path) -> None:
        """Ensure file size is within limits"""
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            if size_mb > SECURITY["max_file_size_mb"]:
                raise SecurityError(
                    f"File too large: {size_mb:.2f}MB "
                    f"(max: {SECURITY['max_file_size_mb']}MB)"
                )

    @classmethod
    def create_file(cls, path: str, content: str) -> Dict:
        """Create a new file with content"""
        try:
            file_path = cls.validate_path(path)
            cls.validate_extension(file_path, "create")

            if file_path.exists():
                return {
                    "success": False,
                    "error": f"File already exists: {file_path.name}",
                    "suggestion": "Use update_file to modify existing files"
                }

            # Check content size
            content_size_mb = len(content.encode('utf-8')) / (1024 * 1024)
            if content_size_mb > SECURITY["max_file_size_mb"]:
                return {
                    "success": False,
                    "error": f"Content too large: {content_size_mb:.2f}MB"
                }

            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding="utf-8")

            AuditLogger.log("create_file", {"path": str(file_path), "size": len(content)})

            return {
                "success": True,
                "path": str(file_path),
                "size_bytes": len(content),
                "message": f"Created {file_path.name}"
            }

        except SecurityError as e:
            AuditLogger.log("create_file", {"path": path, "error": str(e)}, "denied")
            return {"success": False, "error": str(e)}
        except Exception as e:
            return {"success": False, "error": f"Failed to create file: {e}"}

    @classmethod
    def read_file(cls, path: str, start_line: int = 1, end_line: int = -1) -> Dict:
        """Read file content with optional line range"""
        try:
            file_path = cls.validate_path(path)
            cls.validate_extension(file_path, "read")
            cls.check_file_size(file_path)

            if not file_path.exists():
                return {"success": False, "error": "File not found"}

            if not file_path.is_file():
                return {"success": False, "error": "Path is not a file"}

            content = file_path.read_text(encoding="utf-8")
            
            # Line-based reading
            lines = content.splitlines()
            total_lines = len(lines)
            
            # Default to all lines if end_line is -1
            if end_line == -1:
                end_line = total_lines
                
            # Clamp values
            start_line = max(1, start_line)
            end_line = min(total_lines, end_line)
            
            if start_line > end_line:
                if total_lines == 0:
                    return {
                        "success": True, 
                        "path": str(file_path), 
                        "content": "", 
                        "size_bytes": 0, 
                        "lines": 0
                    }
                return {"success": False, "error": f"Invalid line range: {start_line} to {end_line} (File has {total_lines} lines)"}

            # Extract lines (0-indexed slice)
            selected_lines = lines[start_line-1:end_line]
            result_content = "\n".join(selected_lines)

            AuditLogger.log("read_file", {
                "path": str(file_path), 
                "lines": f"{start_line}-{end_line}",
                "size": len(result_content)
            })

            return {
                "success": True,
                "path": str(file_path),
                "content": result_content,
                "size_bytes": len(result_content),
                "lines": len(selected_lines),
                "total_lines": total_lines,
                "start_line": start_line,
                "end_line": end_line
            }

        except SecurityError as e:
            AuditLogger.log("read_file", {"path": path, "error": str(e)}, "denied")
            return {"success": False, "error": str(e)}
        except UnicodeDecodeError:
            return {"success": False, "error": "File is not text (binary file)"}
        except Exception as e:
            return {"success": False, "error": f"Failed to read file: {e}"}

    @classmethod
    def update_file(cls, path: str, content: str) -> Dict:
        """Update existing file content"""
        try:
            file_path = cls.validate_path(path)
            cls.validate_extension(file_path, "write")

            if not file_path.exists():
                return {
                    "success": False,
                    "error": "File not found",
                    "suggestion": "Use create_file for new files"
                }

            # Backup old content
            old_content = file_path.read_text(encoding="utf-8")
            backup_path = file_path.with_suffix(file_path.suffix + ".backup")
            backup_path.write_text(old_content, encoding="utf-8")

            file_path.write_text(content, encoding="utf-8")

            AuditLogger.log("update_file", {
                "path": str(file_path),
                "old_size": len(old_content),
                "new_size": len(content)
            })

            return {
                "success": True,
                "path": str(file_path),
                "backup": str(backup_path),
                "message": f"Updated {file_path.name}"
            }

        except SecurityError as e:
            AuditLogger.log("update_file", {"path": path, "error": str(e)}, "denied")
            return {"success": False, "error": str(e)}
        except Exception as e:
            return {"success": False, "error": f"Failed to update file: {e}"}

    @classmethod
    def delete_file(cls, path: str) -> Dict:
        """Delete a file (requires confirmation)"""
        try:
            file_path = cls.validate_path(path)

            if not file_path.exists():
                return {"success": False, "error": "File not found"}

            if not file_path.is_file():
                return {"success": False, "error": "Path is not a file"}

            size = file_path.stat().st_size
            file_path.unlink()

            AuditLogger.log("delete_file", {"path": str(file_path), "size": size})

            return {
                "success": True,
                "path": str(file_path),
                "message": f"Deleted {file_path.name}"
            }

        except SecurityError as e:
            AuditLogger.log("delete_file", {"path": path, "error": str(e)}, "denied")
            return {"success": False, "error": str(e)}
        except Exception as e:
            return {"success": False, "error": f"Failed to delete file: {e}"}

    @classmethod
    def list_directory(cls, path: str = ".") -> Dict:
        """List directory contents"""
        try:
            dir_path = cls.validate_path(path)

            if not dir_path.exists():
                return {"success": False, "error": "Directory not found"}

            if not dir_path.is_dir():
                return {"success": False, "error": "Path is not a directory"}

            items = []
            for item in sorted(dir_path.iterdir()):
                try:
                    stat = item.stat()
                    items.append({
                        "name": item.name,
                        "type": "dir" if item.is_dir() else "file",
                        "size": stat.st_size if item.is_file() else None,
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    })
                except Exception:
                    continue

            AuditLogger.log("list_directory", {"path": str(dir_path), "items": len(items)})

            return {
                "success": True,
                "path": str(dir_path),
                "items": items,
                "count": len(items)
            }

        except SecurityError as e:
            AuditLogger.log("list_directory", {"path": path, "error": str(e)}, "denied")
            return {"success": False, "error": str(e)}
        except Exception as e:
            return {"success": False, "error": f"Failed to list directory: {e}"}


class SecureCommandRunner:
    """Secure command execution with validation"""

    @staticmethod
    def validate_command(command: str) -> Tuple[bool, str]:
        """Check if command is safe to run"""
        cmd_lower = command.lower().strip()

        # Check for dangerous commands
        for dangerous in DANGEROUS_COMMANDS:
            if dangerous in cmd_lower.split():
                return False, f"Dangerous command detected: {dangerous}"

        # Check for command injection patterns
        injection_patterns = [";", "&&", "||", "|", "`", "$(", ">", "<", "&"]
        for pattern in injection_patterns:
            if pattern in command:
                return False, f"Potential command injection detected: {pattern}"

        return True, "OK"

    @classmethod
    def execute_command(cls, command: str, require_confirmation: bool = False) -> Dict:
        """Execute shell command with security checks"""
        try:
            is_safe, reason = cls.validate_command(command)

            if not is_safe:
                AuditLogger.log("execute_command", {
                    "command": command,
                    "error": reason
                }, "blocked")
                return {
                    "success": False,
                    "error": f"Command blocked: {reason}",
                    "requires_confirmation": True
                }

            # Execute command
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30,  # 30 second timeout
                cwd=str(SECURITY["allowed_directories"][0])  # Run in workspace
            )

            AuditLogger.log("execute_command", {
                "command": command,
                "returncode": result.returncode
            })

            return {
                "success": result.returncode == 0,
                "command": command,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }

        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Command timeout (30s)"}
        except Exception as e:
            return {"success": False, "error": f"Command execution failed: {e}"}


# Tool definitions for AI function calling
TOOLS_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "create_file",
            "description": "Create a new file with content. File must not already exist.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to workspace (e.g., 'test.txt' or 'folder/file.json')"
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write to the file"
                    }
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read content from an existing file. Supports reading specific lines.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to read"
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "Start line number (1-based, default: 1)"
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "End line number (default: -1 for end of file)"
                    }
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "update_file",
            "description": "Update content of an existing file. Creates backup automatically.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to update"
                    },
                    "content": {
                        "type": "string",
                        "description": "New content for the file"
                    }
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "delete_file",
            "description": "Delete a file. Use with caution.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to delete"
                    }
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List all files and folders in a directory. Use '.' for current directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path to list (default: '.')"
                    }
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "execute_command",
            "description": "Execute a safe shell command. Dangerous commands are blocked.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to execute"
                    }
                },
                "required": ["command"]
            }
        }
    }
]


def execute_tool(tool_name: str, arguments: Dict) -> Dict:
    """Execute a tool and return results"""
    tools = {
        "create_file": SecureFileOps.create_file,
        "read_file": SecureFileOps.read_file,
        "update_file": SecureFileOps.update_file,
        "delete_file": SecureFileOps.delete_file,
        "list_directory": SecureFileOps.list_directory,
        "execute_command": SecureCommandRunner.execute_command,
    }

    if tool_name not in tools:
        return {"success": False, "error": f"Unknown tool: {tool_name}"}

    # Ensure list_directory has a default path if not provided
    if tool_name == "list_directory" and "path" not in arguments:
        arguments["path"] = "."

    try:
        return tools[tool_name](**arguments)
    except TypeError as e:
        return {"success": False, "error": f"Invalid arguments for {tool_name}: {e}"}
