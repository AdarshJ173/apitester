# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Comprehensive test suite with 106+ tests
  - Tests for config module (94% coverage)
  - Tests for agent_tools module (76% coverage)
  - Tests for config_manager module (97% coverage)
  - Tests for tool_parser module (91% coverage)
- External configuration support
  - YAML configuration file (config.yaml)
  - Environment variable support with .env files
  - Template .env.example file
- CI/CD pipeline with GitHub Actions
  - Automated testing on multiple OS and Python versions
  - Automated linting and formatting checks
  - Security scanning with Bandit
  - Automated PyPI releases
- Code quality tools
  - Ruff for linting
  - Black for formatting
  - MyPy for type checking
  - Pre-commit hooks
- MIT License
- Contributing guidelines
- Project metadata in pyproject.toml

### Changed

- Migrated from hardcoded configuration to external YAML/env-based config
- Updated dependencies to latest secure versions
  - requests: 2.31.0 → 2.32.0+
  - Added python-dotenv, pydantic, pyyaml
- Improved type hints throughout codebase
- Enhanced documentation

### Fixed

- Removed committed venv/ and __pycache__ from repository
- Updated deprecated typing imports (Dict → dict, List → list, etc.)
- Fixed import ordering
- Removed unused imports

### Security

- Updated requests library to address CVEs
- Added Bandit security scanning to CI pipeline
- Enhanced path validation in tests
- Added security section to documentation

## [1.0.0] - 2026-02-17

### Added

- Initial release of AI Agent CRUD
- Core AI agent functionality
  - Multi-provider support (OpenAI, Anthropic, Groq, OpenRouter, NVIDIA, Together, Mistral, Cohere)
  - File system operations (create, read, update, delete)
  - Directory listing
  - Safe command execution
  - Conversation memory
- Security features
  - Sandboxed workspace access
  - Path traversal protection
  - File type restrictions
  - Command injection prevention
  - Audit logging
  - Automatic backups
- Configuration management
  - Persistent API key storage
  - Provider switching
  - Model selection
- Tool parsing system
  - Text-based tool parsing for universal compatibility
  - Native function calling support
- Rich CLI interface
  - Interactive prompts
  - Progress spinners
  - Formatted output

### Notes

This is the initial production release. The codebase has been used in production environments but is now being prepared for open source distribution with proper testing, documentation, and CI/CD.

---

## Release Notes Template

When creating a new release, use this template:

```markdown
## [X.Y.Z] - YYYY-MM-DD

### Added
- New features

### Changed
- Changes to existing functionality

### Deprecated
- Soon-to-be removed features

### Removed
- Now removed features

### Fixed
- Bug fixes

### Security
- Security improvements
```
