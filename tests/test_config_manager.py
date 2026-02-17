"""
Tests for config_manager.py module
"""

import os
import json
import pytest
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock
from datetime import datetime


class TestProviderConfig:
    """Tests for ProviderConfig dataclass"""

    def test_default_values(self):
        """Test ProviderConfig has correct default values"""
        from config_manager import ProviderConfig

        config = ProviderConfig()

        assert config.api_key == ""
        assert config.model == ""
        assert config.last_used == ""

    def test_custom_values(self):
        """Test ProviderConfig accepts custom values"""
        from config_manager import ProviderConfig

        config = ProviderConfig(api_key="test_key", model="gpt-4", last_used="2026-02-17T12:00:00")

        assert config.api_key == "test_key"
        assert config.model == "gpt-4"
        assert config.last_used == "2026-02-17T12:00:00"


class TestConfigManagerInit:
    """Tests for ConfigManager initialization"""

    def test_default_config_structure(self):
        """Test ConfigManager has correct default config"""
        from config_manager import ConfigManager

        with patch.object(ConfigManager, "_load"):
            manager = ConfigManager()

            assert manager.config["last_provider"] == ""
            assert manager.config["providers"] == {}
            assert manager.config["version"] == "1.0"

    def test_load_called_on_init(self):
        """Test _load is called during initialization"""
        from config_manager import ConfigManager

        with patch.object(ConfigManager, "_load") as mock_load:
            manager = ConfigManager()
            mock_load.assert_called_once()


class TestConfigManagerLoad:
    """Tests for ConfigManager._load method"""

    def test_load_existing_config(self, tmp_path):
        """Test loading existing configuration file"""
        from config_manager import ConfigManager

        config_file = tmp_path / ".ai_agent_config.json"
        config_data = {
            "last_provider": "openai",
            "providers": {
                "openai": {
                    "api_key": "test_key",
                    "model": "gpt-4",
                    "last_used": "2026-02-17T12:00:00",
                }
            },
            "version": "1.0",
        }
        config_file.write_text(json.dumps(config_data))

        with patch.object(ConfigManager, "CONFIG_FILE", config_file):
            manager = ConfigManager()

            assert manager.config["last_provider"] == "openai"
            assert "openai" in manager.config["providers"]

    def test_load_nonexistent_config(self, tmp_path):
        """Test loading when config file doesn't exist"""
        from config_manager import ConfigManager

        config_file = tmp_path / "nonexistent.json"

        with patch.object(ConfigManager, "CONFIG_FILE", config_file):
            manager = ConfigManager()

            # Should use defaults
            assert manager.config["last_provider"] == ""
            assert manager.config["providers"] == {}

    def test_load_corrupted_config(self, tmp_path):
        """Test loading corrupted config file"""
        from config_manager import ConfigManager

        config_file = tmp_path / ".ai_agent_config.json"
        config_file.write_text("invalid json")

        with patch.object(ConfigManager, "CONFIG_FILE", config_file):
            manager = ConfigManager()

            # Should use defaults
            assert manager.config["last_provider"] == ""


class TestConfigManagerSave:
    """Tests for ConfigManager.save method"""

    def test_save_config_success(self, tmp_path):
        """Test saving configuration successfully"""
        from config_manager import ConfigManager

        config_file = tmp_path / ".ai_agent_config.json"

        with patch.object(ConfigManager, "CONFIG_FILE", config_file):
            with patch.object(ConfigManager, "_load"):
                manager = ConfigManager()
                manager.config["last_provider"] = "openai"
                manager.save()

                # Verify file was created
                assert config_file.exists()

                # Verify content
                saved_data = json.loads(config_file.read_text())
                assert saved_data["last_provider"] == "openai"

    def test_save_config_failure(self, tmp_path, capsys):
        """Test save handles errors gracefully"""
        from config_manager import ConfigManager

        config_file = tmp_path / ".ai_agent_config.json"

        with patch.object(ConfigManager, "CONFIG_FILE", config_file):
            with patch.object(ConfigManager, "_load"):
                manager = ConfigManager()

                # Make file read-only to cause error
                with patch("builtins.open", side_effect=PermissionError("Access denied")):
                    manager.save()

                    # Should print warning
                    captured = capsys.readouterr()
                    assert "Warning" in captured.out


class TestConfigManagerGetProviderConfig:
    """Tests for ConfigManager.get_provider_config method"""

    def test_get_existing_provider(self):
        """Test getting config for existing provider"""
        from config_manager import ConfigManager, ProviderConfig

        with patch.object(ConfigManager, "_load"):
            manager = ConfigManager()
            manager.config["providers"]["openai"] = {
                "api_key": "test_key",
                "model": "gpt-4",
                "last_used": "2026-02-17T12:00:00",
            }

            config = manager.get_provider_config("openai")

            assert isinstance(config, ProviderConfig)
            assert config.api_key == "test_key"
            assert config.model == "gpt-4"

    def test_get_nonexistent_provider(self):
        """Test getting config for non-existent provider returns defaults"""
        from config_manager import ConfigManager, ProviderConfig

        with patch.object(ConfigManager, "_load"):
            manager = ConfigManager()

            config = manager.get_provider_config("nonexistent")

            assert isinstance(config, ProviderConfig)
            assert config.api_key == ""
            assert config.model == ""


class TestConfigManagerSetProviderConfig:
    """Tests for ConfigManager.set_provider_config method"""

    @patch("config_manager.datetime")
    def test_set_provider_config(self, mock_datetime):
        """Test setting provider configuration"""
        from config_manager import ConfigManager

        mock_now = MagicMock()
        mock_now.isoformat.return_value = "2026-02-17T12:00:00"
        mock_datetime.now.return_value = mock_now

        with patch.object(ConfigManager, "_load"):
            with patch.object(ConfigManager, "save") as mock_save:
                manager = ConfigManager()
                manager.set_provider_config("openai", "test_key", "gpt-4")

                assert manager.config["last_provider"] == "openai"
                assert "openai" in manager.config["providers"]
                assert manager.config["providers"]["openai"]["api_key"] == "test_key"
                assert manager.config["providers"]["openai"]["model"] == "gpt-4"
                assert manager.config["providers"]["openai"]["last_used"] == "2026-02-17T12:00:00"
                mock_save.assert_called_once()

    def test_set_provider_creates_providers_dict(self):
        """Test set_provider_config creates providers dict if missing"""
        from config_manager import ConfigManager

        with patch.object(ConfigManager, "_load"):
            with patch.object(ConfigManager, "save"):
                manager = ConfigManager()
                del manager.config["providers"]  # Remove providers key

                manager.set_provider_config("openai", "key", "model")

                assert "providers" in manager.config


class TestConfigManagerGetLastProvider:
    """Tests for ConfigManager.get_last_provider method"""

    def test_get_last_provider_exists(self):
        """Test getting last provider when set"""
        from config_manager import ConfigManager

        with patch.object(ConfigManager, "_load"):
            manager = ConfigManager()
            manager.config["last_provider"] = "openai"

            result = manager.get_last_provider()

            assert result == "openai"

    def test_get_last_provider_none_when_empty(self):
        """Test getting last provider returns None when empty"""
        from config_manager import ConfigManager

        with patch.object(ConfigManager, "_load"):
            manager = ConfigManager()
            manager.config["last_provider"] = ""

            result = manager.get_last_provider()

            assert result is None

    def test_get_last_provider_none_when_missing(self):
        """Test getting last provider returns None when key missing"""
        from config_manager import ConfigManager

        with patch.object(ConfigManager, "_load"):
            manager = ConfigManager()
            del manager.config["last_provider"]

            result = manager.get_last_provider()

            assert result is None


class TestConfigManagerHasSavedConfig:
    """Tests for ConfigManager.has_saved_config method"""

    def test_has_saved_config_true(self):
        """Test checking saved config for existing provider"""
        from config_manager import ConfigManager

        with patch.object(ConfigManager, "_load"):
            manager = ConfigManager()
            manager.config["providers"]["openai"] = {"api_key": "test"}

            assert manager.has_saved_config("openai") is True

    def test_has_saved_config_false(self):
        """Test checking saved config for non-existing provider"""
        from config_manager import ConfigManager

        with patch.object(ConfigManager, "_load"):
            manager = ConfigManager()

            assert manager.has_saved_config("nonexistent") is False


class TestConfigManagerGetAllSavedProviders:
    """Tests for ConfigManager.get_all_saved_providers method"""

    def test_get_all_saved_providers(self):
        """Test getting all saved providers"""
        from config_manager import ConfigManager, ProviderConfig

        with patch.object(ConfigManager, "_load"):
            manager = ConfigManager()
            manager.config["providers"] = {
                "openai": {"api_key": "key1", "model": "gpt-4"},
                "anthropic": {"api_key": "key2", "model": "claude"},
            }

            providers = manager.get_all_saved_providers()

            assert len(providers) == 2
            assert "openai" in providers
            assert "anthropic" in providers
            assert isinstance(providers["openai"], ProviderConfig)

    def test_get_all_saved_providers_empty(self):
        """Test getting all saved providers when none exist"""
        from config_manager import ConfigManager

        with patch.object(ConfigManager, "_load"):
            manager = ConfigManager()

            providers = manager.get_all_saved_providers()

            assert providers == {}


class TestConfigManagerClearProvider:
    """Tests for ConfigManager.clear_provider method"""

    def test_clear_existing_provider(self):
        """Test clearing existing provider"""
        from config_manager import ConfigManager

        with patch.object(ConfigManager, "_load"):
            with patch.object(ConfigManager, "save") as mock_save:
                manager = ConfigManager()
                manager.config["providers"]["openai"] = {"api_key": "test"}
                manager.config["last_provider"] = "openai"

                manager.clear_provider("openai")

                assert "openai" not in manager.config["providers"]
                assert manager.config["last_provider"] == ""
                mock_save.assert_called_once()

    def test_clear_nonexistent_provider(self):
        """Test clearing non-existent provider doesn't error"""
        from config_manager import ConfigManager

        with patch.object(ConfigManager, "_load"):
            with patch.object(ConfigManager, "save") as mock_save:
                manager = ConfigManager()

                manager.clear_provider("nonexistent")

                # Should not raise error
                mock_save.assert_not_called()

    def test_clear_provider_updates_last_provider(self):
        """Test clearing provider updates last_provider if it was the last used"""
        from config_manager import ConfigManager

        with patch.object(ConfigManager, "_load"):
            with patch.object(ConfigManager, "save"):
                manager = ConfigManager()
                manager.config["providers"]["openai"] = {"api_key": "test"}
                manager.config["last_provider"] = "openai"

                manager.clear_provider("openai")

                assert manager.config["last_provider"] == ""


class TestGlobalConfigInstance:
    """Tests for global config_manager instance"""

    def test_global_instance_exists(self):
        """Test global config_manager instance is created"""
        from config_manager import config_manager, ConfigManager

        assert isinstance(config_manager, ConfigManager)
