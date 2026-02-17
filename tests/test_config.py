"""
Tests for config.py module
"""

import os
from pathlib import Path
from unittest.mock import patch

import yaml


class TestLoadYamlConfig:
    """Tests for load_yaml_config function"""

    def test_load_existing_config(self, tmp_path):
        """Test loading existing YAML config file"""
        from config import load_yaml_config

        config_path = tmp_path / "test_config.yaml"
        config_data = {"security": {"max_file_size_mb": 20}}
        config_path.write_text(yaml.dump(config_data))

        result = load_yaml_config(config_path)
        assert result["security"]["max_file_size_mb"] == 20

    def test_load_nonexistent_config(self, tmp_path):
        """Test loading non-existent config returns empty dict"""
        from config import load_yaml_config

        config_path = tmp_path / "nonexistent.yaml"
        result = load_yaml_config(config_path)
        assert result == {}

    def test_load_config_without_path(self):
        """Test loading config from default location"""
        from config import load_yaml_config

        # Should try to load from config.yaml in parent directory
        result = load_yaml_config()
        # Result depends on whether config.yaml exists
        assert isinstance(result, dict)


class TestParseListFromEnv:
    """Tests for parse_list_from_env function"""

    def test_parse_comma_separated_values(self):
        """Test parsing comma-separated environment variable"""
        from config import parse_list_from_env

        result = parse_list_from_env(".txt,.json,.py")
        assert result == [".txt", ".json", ".py"]

    def test_parse_with_spaces(self):
        """Test parsing values with extra spaces"""
        from config import parse_list_from_env

        result = parse_list_from_env(" .txt , .json , .py ")
        assert result == [".txt", ".json", ".py"]

    def test_parse_empty_string(self):
        """Test parsing empty string returns empty list"""
        from config import parse_list_from_env

        result = parse_list_from_env("")
        assert result == []

    def test_parse_none_value(self):
        """Test parsing None returns empty list"""
        from config import parse_list_from_env

        result = parse_list_from_env(None)
        assert result == []


class TestGetEnvOrConfig:
    """Tests for get_env_or_config function"""

    def test_env_var_takes_precedence(self):
        """Test environment variable overrides config value"""
        from config import get_env_or_config

        with patch.dict(os.environ, {"TEST_VAR": "env_value"}):
            result = get_env_or_config("TEST_VAR", "config_value")
            assert result == "env_value"

    def test_config_value_fallback(self):
        """Test config value used when env var not set"""
        from config import get_env_or_config

        with patch.dict(os.environ, {}, clear=True):
            result = get_env_or_config("NONEXISTENT_VAR", "config_value")
            assert result == "config_value"

    def test_cast_to_int(self):
        """Test casting environment variable to int"""
        from config import get_env_or_config

        with patch.dict(os.environ, {"TEST_INT": "42"}):
            result = get_env_or_config("TEST_INT", 10, int)
            assert result == 42
            assert isinstance(result, int)

    def test_cast_to_bool_true(self):
        """Test casting various true values to bool"""
        from config import get_env_or_config

        for val in ["true", "True", "1", "yes", "on"]:
            with patch.dict(os.environ, {"TEST_BOOL": val}):
                result = get_env_or_config("TEST_BOOL", False, bool)
                assert result is True, f"Failed for value: {val}"

    def test_cast_to_bool_false(self):
        """Test casting various false values to bool"""
        from config import get_env_or_config

        for val in ["false", "False", "0", "no", "off"]:
            with patch.dict(os.environ, {"TEST_BOOL": val}):
                result = get_env_or_config("TEST_BOOL", True, bool)
                assert result is False, f"Failed for value: {val}"

    def test_cast_to_list(self):
        """Test casting environment variable to list"""
        from config import get_env_or_config

        with patch.dict(os.environ, {"TEST_LIST": "a,b,c"}):
            result = get_env_or_config("TEST_LIST", [], list)
            assert result == ["a", "b", "c"]

    def test_invalid_int_returns_config(self):
        """Test invalid int casting falls back to config"""
        from config import get_env_or_config

        with patch.dict(os.environ, {"TEST_INT": "not_a_number"}):
            result = get_env_or_config("TEST_INT", 10, int)
            assert result == 10


class TestConfigurationValues:
    """Tests for configuration module-level values"""

    def test_base_dir_is_path(self):
        """Test BASE_DIR is a Path object"""
        from config import BASE_DIR

        assert isinstance(BASE_DIR, Path)

    def test_security_config_structure(self):
        """Test SECURITY config has required keys"""
        from config import SECURITY

        required_keys = [
            "max_file_size_mb",
            "allowed_extensions",
            "blocked_extensions",
            "allowed_directories",
            "blocked_patterns",
        ]
        for key in required_keys:
            assert key in SECURITY, f"Missing key: {key}"

    def test_security_lists_are_valid(self):
        """Test security lists contain expected values"""
        from config import SECURITY

        assert isinstance(SECURITY["allowed_extensions"], list)
        assert isinstance(SECURITY["blocked_extensions"], list)
        assert ".txt" in SECURITY["allowed_extensions"]
        assert ".exe" in SECURITY["blocked_extensions"]

    def test_dangerous_commands_list(self):
        """Test DANGEROUS_COMMANDS is a list"""
        from config import DANGEROUS_COMMANDS

        assert isinstance(DANGEROUS_COMMANDS, list)
        assert "rm" in DANGEROUS_COMMANDS
        assert "del" in DANGEROUS_COMMANDS

    def test_safe_commands_list(self):
        """Test SAFE_COMMANDS is a list"""
        from config import SAFE_COMMANDS

        assert isinstance(SAFE_COMMANDS, list)
        assert "ls" in SAFE_COMMANDS
        assert "echo" in SAFE_COMMANDS

    def test_timeout_values_are_integers(self):
        """Test timeout constants are integers"""
        from config import API_RETRY_TIMEOUT, API_TIMEOUT, COMMAND_TIMEOUT

        assert isinstance(COMMAND_TIMEOUT, int)
        assert isinstance(API_TIMEOUT, int)
        assert isinstance(API_RETRY_TIMEOUT, int)
        assert COMMAND_TIMEOUT > 0
        assert API_TIMEOUT > 0
        assert API_RETRY_TIMEOUT > 0

    def test_conversation_settings(self):
        """Test conversation-related settings"""
        from config import MAX_CONVERSATION_MESSAGES, MAX_TOOL_ITERATIONS

        assert isinstance(MAX_CONVERSATION_MESSAGES, int)
        assert isinstance(MAX_TOOL_ITERATIONS, int)
        assert MAX_CONVERSATION_MESSAGES > 0
        assert MAX_TOOL_ITERATIONS > 0

    def test_logging_settings(self):
        """Test logging configuration"""
        from config import AUDIT_LOG, ENABLE_AUDIT_LOGGING, LOG_LEVEL

        assert LOG_LEVEL in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        assert isinstance(ENABLE_AUDIT_LOGGING, bool)
        assert isinstance(AUDIT_LOG, Path)

    def test_debug_is_boolean(self):
        """Test DEBUG is a boolean"""
        from config import DEBUG

        assert isinstance(DEBUG, bool)


class TestEnsureDirectories:
    """Tests for ensure_directories function"""

    def test_creates_workspace_directory(self, tmp_path):
        """Test workspace directory is created"""
        from config import ensure_directories

        with patch("config.BASE_DIR", tmp_path):
            ensure_directories()
            assert (tmp_path / "workspace").exists()

    def test_creates_logs_directory(self, tmp_path):
        """Test logs directory is created"""
        from config import ensure_directories

        with patch("config.BASE_DIR", tmp_path):
            with patch("config.AUDIT_LOG", tmp_path / "logs" / "audit.log"):
                ensure_directories()
                assert (tmp_path / "logs").exists()

    def test_creates_data_directory(self, tmp_path):
        """Test data directory is created"""
        from config import ensure_directories

        with patch("config.BASE_DIR", tmp_path):
            ensure_directories()
            assert (tmp_path / "data").exists()

    def test_existing_directories_not_error(self, tmp_path):
        """Test function works when directories already exist"""
        from config import ensure_directories

        (tmp_path / "workspace").mkdir()

        with patch("config.BASE_DIR", tmp_path):
            # Should not raise an error
            ensure_directories()


class TestEnvironmentVariableOverride:
    """Integration tests for environment variable overrides"""

    def test_max_file_size_from_env(self):
        """Test MAX_FILE_SIZE_MB can be overridden via env var"""
        # This tests the actual module behavior
        with patch.dict(os.environ, {"MAX_FILE_SIZE_MB": "50"}):
            # Must reimport to pick up new env var
            import importlib

            import config

            importlib.reload(config)

            assert config.SECURITY["max_file_size_mb"] == 50

    def test_log_level_from_env(self):
        """Test LOG_LEVEL can be overridden via env var"""
        with patch.dict(os.environ, {"LOG_LEVEL": "DEBUG"}):
            import importlib

            import config

            importlib.reload(config)

            assert config.LOG_LEVEL == "DEBUG"

    def test_debug_from_env(self):
        """Test DEBUG can be overridden via env var"""
        with patch.dict(os.environ, {"DEBUG": "true"}):
            import importlib

            import config

            importlib.reload(config)

            assert config.DEBUG is True
