"""
Tests for agent_tools.py module
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestSecurityError:
    """Tests for SecurityError exception"""

    def test_security_error_is_exception(self):
        """Test SecurityError is an Exception subclass"""
        from agent_tools import SecurityError

        assert issubclass(SecurityError, Exception)

    def test_security_error_message(self):
        """Test SecurityError accepts message"""
        from agent_tools import SecurityError

        error = SecurityError("Test error message")
        assert str(error) == "Test error message"


class TestAuditLogger:
    """Tests for AuditLogger class"""

    def test_log_creates_valid_json(self, tmp_path):
        """Test log creates valid JSON entry"""
        from agent_tools import AuditLogger

        log_file = tmp_path / "test_audit.log"

        with patch("agent_tools.AUDIT_LOG", log_file):
            AuditLogger.log("test_action", {"key": "value"})

            content = log_file.read_text()
            entry = json.loads(content.strip())

            assert entry["action"] == "test_action"
            assert entry["details"]["key"] == "value"
            assert entry["status"] == "success"
            assert "timestamp" in entry

    def test_log_handles_failure_gracefully(self):
        """Test log doesn't break execution on failure"""
        from agent_tools import AuditLogger

        # Should not raise exception even if logging fails
        with patch("builtins.open", side_effect=OSError("Cannot write")):
            result = AuditLogger.log("test_action", {})
            # Function should complete without exception
            assert result is None


class TestSecureFileOpsValidatePath:
    """Tests for SecureFileOps.validate_path"""

    def test_valid_path_within_workspace(self, tmp_path):
        """Test valid path within allowed directory"""
        from agent_tools import SecureFileOps

        test_file = tmp_path / "workspace" / "test.txt"
        test_file.parent.mkdir(parents=True, exist_ok=True)

        with patch("config.SECURITY") as mock_security:
            mock_security.__getitem__ = lambda s, k: (
                [tmp_path] if k == "allowed_directories" else []
            )

            result = SecureFileOps.validate_path(str(test_file))
            assert isinstance(result, Path)

    def test_blocked_pattern_rejected(self, tmp_path):
        """Test path with blocked pattern is rejected"""
        from agent_tools import SecureFileOps, SecurityError

        with patch("config.SECURITY") as mock_security:
            mock_security.__getitem__ = lambda s, k: (
                [tmp_path]
                if k == "allowed_directories"
                else [".."] if k == "blocked_patterns" else []
            )

            with pytest.raises(SecurityError):
                SecureFileOps.validate_path("../etc/passwd")

    def test_path_outside_allowed_directories(self):
        """Test path outside allowed directories is rejected"""
        from agent_tools import SecureFileOps, SecurityError

        with patch("config.SECURITY") as mock_security:
            mock_security.__getitem__ = lambda s, k: (
                [Path("/allowed/path")] if k == "allowed_directories" else []
            )

            with pytest.raises(SecurityError):
                SecureFileOps.validate_path("/outside/path/file.txt")


class TestSecureFileOpsValidateExtension:
    """Tests for SecureFileOps.validate_extension"""

    def test_blocked_extension_rejected(self):
        """Test blocked file extension is rejected"""
        from agent_tools import SecureFileOps, SecurityError

        with patch("config.SECURITY") as mock_security:
            mock_security.__getitem__ = lambda s, k: ([".exe"] if k == "blocked_extensions" else [])

            with pytest.raises(SecurityError):
                SecureFileOps.validate_extension(Path("test.exe"), "read")

    def test_non_allowed_extension_for_write_rejected(self):
        """Test non-allowed extension for write operation"""
        from agent_tools import SecureFileOps, SecurityError

        with patch("config.SECURITY") as mock_security:
            mock_security.__getitem__ = lambda s, k: (
                [".xyz"]
                if k == "blocked_extensions"
                else [".txt"] if k == "allowed_extensions" else []
            )

            with pytest.raises(SecurityError):
                SecureFileOps.validate_extension(Path("test.xyz"), "write")

    def test_allowed_extension_accepted(self):
        """Test allowed file extension is accepted"""
        from agent_tools import SecureFileOps

        with patch("config.SECURITY") as mock_security:
            mock_security.__getitem__ = lambda s, k: ([".txt"] if k == "allowed_extensions" else [])

            # Should not raise exception
            SecureFileOps.validate_extension(Path("test.txt"), "write")


class TestSecureFileOpsCheckFileSize:
    """Tests for SecureFileOps.check_file_size"""

    def test_oversized_file_rejected(self, tmp_path):
        """Test file exceeding max size is rejected"""
        from agent_tools import SecureFileOps, SecurityError

        test_file = tmp_path / "large.txt"
        # Create a file larger than 10MB
        test_file.write_bytes(b"x" * (11 * 1024 * 1024))

        with patch("config.SECURITY") as mock_security:
            mock_security.__getitem__ = lambda s, k: 10 if k == "max_file_size_mb" else None

            with pytest.raises(SecurityError):
                SecureFileOps.check_file_size(test_file)

    def test_valid_size_file_accepted(self, tmp_path):
        """Test file within size limit is accepted"""
        from agent_tools import SecureFileOps

        test_file = tmp_path / "small.txt"
        test_file.write_text("Small content")

        with patch("config.SECURITY") as mock_security:
            mock_security.__getitem__ = lambda s, k: 10 if k == "max_file_size_mb" else None

            # Should not raise exception
            SecureFileOps.check_file_size(test_file)


class TestSecureFileOpsCreateFile:
    """Tests for SecureFileOps.create_file"""

    def test_create_new_file_success(self, tmp_path):
        """Test creating a new file successfully"""
        from agent_tools import SecureFileOps

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        test_file = workspace / "test.txt"

        with patch("config.BASE_DIR", tmp_path), patch("config.SECURITY") as mock_security:
            mock_security.__getitem__ = lambda s, k: (
                [tmp_path]
                if k == "allowed_directories"
                else [".txt"] if k == "allowed_extensions" else []
            )

            result = SecureFileOps.create_file(str(test_file), "Test content")

            assert result["success"] is True
            assert test_file.exists()
            assert test_file.read_text() == "Test content"

    def test_create_existing_file_fails(self, tmp_path):
        """Test creating file that already exists fails"""
        from agent_tools import SecureFileOps

        test_file = tmp_path / "test.txt"
        test_file.write_text("Existing content")

        with patch("config.SECURITY") as mock_security:
            mock_security.__getitem__ = lambda s, k: (
                [tmp_path] if k == "allowed_directories" else []
            )

            result = SecureFileOps.create_file(str(test_file), "New content")

            assert result["success"] is False
            assert "already exists" in result["error"].lower()

    def test_create_file_security_error(self):
        """Test create_file handles security errors"""
        from agent_tools import SecureFileOps, SecurityError

        with patch.object(
            SecureFileOps, "validate_path", side_effect=SecurityError("Access denied")
        ):
            result = SecureFileOps.create_file("/invalid/path.txt", "content")

            assert result["success"] is False
            assert "Access denied" in result["error"]


class TestSecureFileOpsReadFile:
    """Tests for SecureFileOps.read_file"""

    def test_read_file_success(self, tmp_path):
        """Test reading a file successfully"""
        from agent_tools import SecureFileOps

        test_file = tmp_path / "test.txt"
        test_file.write_text("Line 1\nLine 2\nLine 3")

        with patch("config.SECURITY") as mock_security:
            mock_security.__getitem__ = lambda s, k: (
                [tmp_path]
                if k == "allowed_directories"
                else (
                    [".txt"] if k == "allowed_extensions" else 10 if k == "max_file_size_mb" else []
                )
            )

            result = SecureFileOps.read_file(str(test_file))

            assert result["success"] is True
            assert result["content"] == "Line 1\nLine 2\nLine 3"
            assert result["lines"] == 3

    def test_read_file_line_range(self, tmp_path):
        """Test reading specific line range"""
        from agent_tools import SecureFileOps

        test_file = tmp_path / "test.txt"
        test_file.write_text("Line 1\nLine 2\nLine 3\nLine 4\nLine 5")

        with patch("config.SECURITY") as mock_security:
            mock_security.__getitem__ = lambda s, k: (
                [tmp_path]
                if k == "allowed_directories"
                else (
                    [".txt"] if k == "allowed_extensions" else 10 if k == "max_file_size_mb" else []
                )
            )

            result = SecureFileOps.read_file(str(test_file), start_line=2, end_line=4)

            assert result["success"] is True
            assert result["content"] == "Line 2\nLine 3\nLine 4"
            assert result["lines"] == 3

    def test_read_nonexistent_file(self):
        """Test reading non-existent file fails"""
        from agent_tools import SecureFileOps

        with patch("config.SECURITY") as mock_security:
            mock_security.__getitem__ = lambda s, k: (
                [Path("/tmp")]
                if k == "allowed_directories"
                else (
                    [".txt"] if k == "allowed_extensions" else 10 if k == "max_file_size_mb" else []
                )
            )

            result = SecureFileOps.read_file("/tmp/nonexistent.txt")

            assert result["success"] is False
            assert "not found" in result["error"].lower()


class TestSecureFileOpsUpdateFile:
    """Tests for SecureFileOps.update_file"""

    def test_update_file_success(self, tmp_path):
        """Test updating a file successfully"""
        from agent_tools import SecureFileOps

        test_file = tmp_path / "test.txt"
        test_file.write_text("Old content")

        with patch("config.SECURITY") as mock_security:
            mock_security.__getitem__ = lambda s, k: (
                [tmp_path]
                if k == "allowed_directories"
                else [".txt"] if k == "allowed_extensions" else []
            )

            result = SecureFileOps.update_file(str(test_file), "New content")

            assert result["success"] is True
            assert test_file.read_text() == "New content"
            # Check backup was created
            assert (tmp_path / "test.txt.backup").exists()

    def test_update_nonexistent_file_fails(self):
        """Test updating non-existent file fails"""
        from agent_tools import SecureFileOps

        with patch("config.SECURITY") as mock_security:
            mock_security.__getitem__ = lambda s, k: (
                [Path("/tmp")] if k == "allowed_directories" else []
            )

            result = SecureFileOps.update_file("/tmp/nonexistent.txt", "content")

            assert result["success"] is False
            assert "not found" in result["error"].lower()


class TestSecureFileOpsDeleteFile:
    """Tests for SecureFileOps.delete_file"""

    def test_delete_file_success(self, tmp_path):
        """Test deleting a file successfully"""
        from agent_tools import SecureFileOps

        test_file = tmp_path / "test.txt"
        test_file.write_text("Content to delete")

        with patch("config.SECURITY") as mock_security:
            mock_security.__getitem__ = lambda s, k: (
                [tmp_path] if k == "allowed_directories" else []
            )

            result = SecureFileOps.delete_file(str(test_file))

            assert result["success"] is True
            assert not test_file.exists()

    def test_delete_nonexistent_file_fails(self):
        """Test deleting non-existent file fails"""
        from agent_tools import SecureFileOps

        with patch("config.SECURITY") as mock_security:
            mock_security.__getitem__ = lambda s, k: (
                [Path("/tmp")] if k == "allowed_directories" else []
            )

            result = SecureFileOps.delete_file("/tmp/nonexistent.txt")

            assert result["success"] is False
            assert "not found" in result["error"].lower()


class TestSecureFileOpsListDirectory:
    """Tests for SecureFileOps.list_directory"""

    def test_list_directory_success(self, tmp_path):
        """Test listing directory contents"""
        from agent_tools import SecureFileOps

        # Create test files
        (tmp_path / "file1.txt").write_text("content1")
        (tmp_path / "file2.txt").write_text("content2")
        (tmp_path / "subdir").mkdir()

        with patch("config.SECURITY") as mock_security:
            mock_security.__getitem__ = lambda s, k: (
                [tmp_path] if k == "allowed_directories" else []
            )

            result = SecureFileOps.list_directory(str(tmp_path))

            assert result["success"] is True
            assert result["count"] == 3

            names = [item["name"] for item in result["items"]]
            assert "file1.txt" in names
            assert "file2.txt" in names
            assert "subdir" in names

    def test_list_nonexistent_directory_fails(self):
        """Test listing non-existent directory fails"""
        from agent_tools import SecureFileOps

        with patch("config.SECURITY") as mock_security:
            mock_security.__getitem__ = lambda s, k: (
                [Path("/tmp")] if k == "allowed_directories" else []
            )

            result = SecureFileOps.list_directory("/tmp/nonexistent_dir")

            assert result["success"] is False
            assert "not found" in result["error"].lower()


class TestSecureCommandRunnerValidateCommand:
    """Tests for SecureCommandRunner.validate_command"""

    def test_dangerous_command_blocked(self):
        """Test dangerous commands are blocked"""
        from agent_tools import SecureCommandRunner

        with patch("config.DANGEROUS_COMMANDS", ["rm", "del"]):
            is_safe, reason = SecureCommandRunner.validate_command("rm -rf /")

            assert is_safe is False
            assert "dangerous" in reason.lower()

    def test_injection_pattern_blocked(self):
        """Test command injection patterns are blocked"""
        from agent_tools import SecureCommandRunner

        is_safe, reason = SecureCommandRunner.validate_command("ls ; rm -rf /")

        assert is_safe is False
        assert "injection" in reason.lower()

    def test_safe_command_allowed(self):
        """Test safe commands are allowed"""
        from agent_tools import SecureCommandRunner

        with patch("config.DANGEROUS_COMMANDS", ["rm"]):
            is_safe, reason = SecureCommandRunner.validate_command("ls -la")

            assert is_safe is True
            assert reason == "OK"


class TestSecureCommandRunnerExecuteCommand:
    """Tests for SecureCommandRunner.execute_command"""

    @patch("agent_tools.subprocess.run")
    def test_execute_successful_command(self, mock_run):
        """Test executing a successful command"""
        from agent_tools import SecureCommandRunner

        mock_run.return_value = MagicMock(returncode=0, stdout="output", stderr="")

        with patch("config.SECURITY") as mock_security:
            mock_security.__getitem__ = lambda s, k: (
                [Path("/tmp")] if k == "allowed_directories" else []
            )

            result = SecureCommandRunner.execute_command("echo test")

            assert result["success"] is True
            assert result["stdout"] == "output"
            assert result["returncode"] == 0

    @patch("agent_tools.subprocess.run")
    def test_execute_blocked_command(self, mock_run):
        """Test blocked command returns error"""
        from agent_tools import SecureCommandRunner

        with patch("config.DANGEROUS_COMMANDS", ["rm"]):
            with patch("config.SECURITY") as mock_security:
                mock_security.__getitem__ = lambda s, k: (
                    [Path("/tmp")] if k == "allowed_directories" else []
                )

                result = SecureCommandRunner.execute_command("rm -rf /")

                assert result["success"] is False
                assert "blocked" in result["error"].lower()
                mock_run.assert_not_called()

    @patch("agent_tools.subprocess.run")
    def test_command_timeout(self, mock_run):
        """Test command timeout handling"""
        import subprocess

        from agent_tools import SecureCommandRunner

        mock_run.side_effect = subprocess.TimeoutExpired("cmd", 30)

        with patch("config.SECURITY") as mock_security:
            mock_security.__getitem__ = lambda s, k: (
                [Path("/tmp")] if k == "allowed_directories" else []
            )

            result = SecureCommandRunner.execute_command("sleep 60")

            assert result["success"] is False
            assert "timeout" in result["error"].lower()


class TestExecuteTool:
    """Tests for execute_tool function"""

    def test_execute_unknown_tool(self):
        """Test executing unknown tool returns error"""
        from agent_tools import execute_tool

        result = execute_tool("unknown_tool", {})

        assert result["success"] is False
        assert "unknown" in result["error"].lower()

    def test_execute_create_file(self, tmp_path):
        """Test execute_tool with create_file"""
        from agent_tools import execute_tool

        test_file = tmp_path / "workspace" / "test.txt"
        test_file.parent.mkdir(parents=True)

        with patch("config.BASE_DIR", tmp_path), patch("config.SECURITY") as mock_security:
            mock_security.__getitem__ = lambda s, k: (
                [tmp_path]
                if k == "allowed_directories"
                else [".txt"] if k == "allowed_extensions" else []
            )

            result = execute_tool(
                "create_file", {"path": str(test_file), "content": "Test content"}
            )

            assert result["success"] is True

    def test_execute_list_directory_default_path(self):
        """Test list_directory with default path"""
        from agent_tools import execute_tool

        result = execute_tool("list_directory", {})

        # Should use "." as default path
        assert isinstance(result, dict)

    def test_execute_tool_invalid_arguments(self):
        """Test execute_tool with invalid arguments"""
        from agent_tools import execute_tool

        result = execute_tool("create_file", {})  # Missing required args

        assert result["success"] is False
        assert "invalid" in result["error"].lower()


class TestToolsDefinitions:
    """Tests for TOOLS_DEFINITIONS"""

    def test_tools_definitions_is_list(self):
        """Test TOOLS_DEFINITIONS is a list"""
        from agent_tools import TOOLS_DEFINITIONS

        assert isinstance(TOOLS_DEFINITIONS, list)
        assert len(TOOLS_DEFINITIONS) > 0

    def test_each_tool_has_required_fields(self):
        """Test each tool definition has required fields"""
        from agent_tools import TOOLS_DEFINITIONS

        for tool in TOOLS_DEFINITIONS:
            assert tool["type"] == "function"
            assert "function" in tool
            assert "name" in tool["function"]
            assert "description" in tool["function"]
            assert "parameters" in tool["function"]

    def test_required_tools_present(self):
        """Test all required tools are defined"""
        from agent_tools import TOOLS_DEFINITIONS

        tool_names = [tool["function"]["name"] for tool in TOOLS_DEFINITIONS]

        required_tools = [
            "create_file",
            "read_file",
            "update_file",
            "delete_file",
            "list_directory",
            "execute_command",
        ]

        for tool in required_tools:
            assert tool in tool_names, f"Missing tool: {tool}"
