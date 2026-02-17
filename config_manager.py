"""
Configuration Manager for AI Agent
Handles persistent storage of API keys and preferences
"""

import json
import os
from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass, asdict


@dataclass
class ProviderConfig:
    """Configuration for a single provider"""

    api_key: str = ""
    model: str = ""
    last_used: str = ""  # ISO timestamp


class ConfigManager:
    """Manages persistent configuration storage"""

    CONFIG_FILE = Path(".ai_agent_config.json")

    def __init__(self):
        self.config: Dict[str, any] = {
            "last_provider": "",
            "providers": {},
            "version": "1.0",
        }
        self._load()

    def _load(self):
        """Load configuration from file"""
        if self.CONFIG_FILE.exists():
            try:
                with open(self.CONFIG_FILE, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    self.config.update(loaded)
            except Exception:
                pass  # Use defaults if file is corrupted

    def save(self):
        """Save configuration to file"""
        try:
            with open(self.CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save config: {e}")

    def get_provider_config(self, provider: str) -> ProviderConfig:
        """Get configuration for a specific provider"""
        provider_data = self.config.get("providers", {}).get(provider, {})
        return ProviderConfig(**provider_data)

    def set_provider_config(self, provider: str, api_key: str, model: str):
        """Save configuration for a specific provider"""
        from datetime import datetime

        if "providers" not in self.config:
            self.config["providers"] = {}

        self.config["providers"][provider] = {
            "api_key": api_key,
            "model": model,
            "last_used": datetime.now().isoformat(),
        }
        self.config["last_provider"] = provider
        self.save()

    def get_last_provider(self) -> Optional[str]:
        """Get the last used provider"""
        return self.config.get("last_provider") or None

    def has_saved_config(self, provider: str) -> bool:
        """Check if we have saved config for a provider"""
        return provider in self.config.get("providers", {})

    def get_all_saved_providers(self) -> Dict[str, ProviderConfig]:
        """Get all providers with saved configuration"""
        providers = {}
        for name, data in self.config.get("providers", {}).items():
            providers[name] = ProviderConfig(**data)
        return providers

    def clear_provider(self, provider: str):
        """Clear configuration for a specific provider"""
        if provider in self.config.get("providers", {}):
            del self.config["providers"][provider]
            if self.config.get("last_provider") == provider:
                self.config["last_provider"] = ""
            self.save()


# Global config instance
config_manager = ConfigManager()
