# AI Agent CRUD

<p align="center">
  <strong>Enterprise-grade AI assistant with secure file system control, command execution, and conversation memory.</strong>
</p>

<p align="center">
  <a href="https://github.com/anomalyco/ai-agent-crud/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT">
  </a>
  <a href="https://pypi.org/project/ai-agent-crud/">
    <img src="https://img.shields.io/pypi/v/ai-agent-crud.svg" alt="PyPI version">
  </a>
  <a href="https://github.com/anomalyco/ai-agent-crud/actions">
    <img src="https://github.com/anomalyco/ai-agent-crud/workflows/CI/badge.svg" alt="CI Status">
  </a>
  <a href="https://codecov.io/gh/anomalyco/ai-agent-crud">
    <img src="https://codecov.io/gh/anomalyco/ai-agent-crud/branch/main/graph/badge.svg" alt="Coverage">
  </a>
</p>

---

## Overview

AI Agent CRUD provides a production-ready AI assistant that can safely interact with your file system, execute commands, and maintain conversation context. Built with security as a primary concern, it implements comprehensive sandboxing, audit logging, and access controls.

## Key Features

- **Multi-Provider Support**: OpenAI, Anthropic, Groq, OpenRouter, NVIDIA, Together, Mistral, Cohere
- **File System Operations**: Create, read, update, delete files and directories
- **Command Execution**: Safe shell command execution with validation
- **Context Memory**: Persistent conversation history and file context
- **Security Architecture**: Sandboxed workspace, audit logging, path validation
- **Universal Tool Parsing**: Works with any AI provider via text-based tool parsing

## Quick Start

### Installation

```bash
pip install ai-agent-crud
```

### Development Setup

```bash
git clone https://github.com/anomalyco/ai-agent-crud.git
cd ai-agent-crud
pip install -e ".[dev,test]"
```

### Running

```bash
ai-agent
```

## Architecture

```
User Interface
      |
      v
AI Agent Core (api_tester.py)
      |
      +-- Conversation Manager
      +-- Tool Orchestrator
      |
      v
Security Layer
      +-- Path Validator
      +-- Command Validator
      +-- Audit Logger
      |
      v
AI Providers (8 supported)
      +-- OpenAI
      +-- Anthropic
      +-- Groq
      +-- OpenRouter
      +-- NVIDIA
      +-- Together
      +-- Mistral
      +-- Cohere
      |
      v
Secure Storage
      +-- Workspace
      +-- Logs
      +-- Data
```

## Security Model

The application implements a four-layer security architecture:

### Layer 1: Path Validation
- All paths resolved to absolute paths
- Access restricted to allowed directories (workspace, logs, data)
- Path traversal attacks blocked (../, ~)
- System directories protected (/etc, C:\Windows)

### Layer 2: File Type Control
- Whitelist: .txt, .py, .json, .md, .yaml, .csv, .js, .html, .css, .xml, .log
- Blacklist: .exe, .dll, .so, .sh, .ps1, .bat
- Maximum file size: 10MB (configurable)

### Layer 3: Command Sanitization
- Dangerous commands blocked: rm, del, format, kill, shutdown
- Injection patterns blocked: ;, &&, ||, |, $(), backticks
- Safe commands allowed: ls, cat, echo, grep, find, pwd

### Layer 4: Audit and Backup
- All operations logged to logs/audit.log
- JSON format with timestamps
- Automatic backups created before file updates
- Backup files: .backup extension

## Supported AI Providers

| Provider | Models | Tool Support | Best For |
|----------|--------|--------------|----------|
| OpenAI | GPT-4o, GPT-4, GPT-3.5 | Native | General purpose |
| Anthropic | Claude 3.5 Sonnet, Claude 3 Opus | Native | Complex reasoning |
| Groq | Llama 3, Mixtral, Gemma | Native | Speed and cost |
| OpenRouter | 100+ models | Native | Model variety |
| NVIDIA | Various LLMs | Native | Enterprise |
| Together | Open source models | Native | Open source |
| Mistral | Mistral Large, Medium | Native | European AI |
| Cohere | Command R+, Command R | Native | Enterprise NLP |

## Usage Examples

### File Operations

```
You: Create a file called workspace/notes.txt with "Meeting notes"
AI: Created workspace/notes.txt

You: Read workspace/notes.txt
AI: Content: Meeting notes

You: Update workspace/notes.txt with "Meeting notes - Updated"
AI: Updated workspace/notes.txt (backup created)

You: List workspace
AI: Found 1 item:
   - notes.txt (25 bytes)
```

### Code Development

```
You: Create a Python function to calculate factorial
AI: [Creates workspace/factorial.py]

You: Read workspace/factorial.py
AI: [Shows code content]

You: Create tests for it
AI: [Creates workspace/test_factorial.py]
```

### Command Execution

```
You: Execute: ls -la workspace
AI: Command executed successfully
   total 12
   drwxr-xr-x 2 user user 4096 Feb 17 10:00 .
   drwxr-xr-x 5 user user 4096 Feb 17 09:00 ..
   -rw-r--r-- 1 user user 100 Feb 17 10:00 notes.txt

You: Execute: rm -rf /
AI: Command blocked: Dangerous command detected: rm
```

## Configuration

### Environment Variables

Create a `.env` file:

```bash
# AI Provider Keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GROQ_API_KEY=gsk_...

# Security Settings
MAX_FILE_SIZE_MB=20
ALLOWED_EXTENSIONS=.txt,.py,.json,.md

# Timeouts
COMMAND_TIMEOUT_SECONDS=30
API_TIMEOUT_SECONDS=120

# Debug
DEBUG=false
```

### YAML Configuration

Edit `config.yaml`:

```yaml
security:
  max_file_size_mb: 10
  allowed_extensions:
    - .txt
    - .py
    - .json
    - .md
  blocked_extensions:
    - .exe
    - .dll
    - .sh

timeouts:
  command: 30
  api: 120
  api_retry: 10

logging:
  level: INFO
  enable_audit: true
```

## Project Structure

```
ai-agent-crud/
├── .github/
│   ├── workflows/              # CI/CD pipelines
│   │   ├── ci.yml             # Test, lint, build
│   │   └── release.yml        # Release automation
│   └── ISSUE_TEMPLATE/        # Issue templates
│
├── tests/                     # Test suite (106 tests)
│   ├── test_config.py         # Configuration tests
│   ├── test_config_manager.py # Persistence tests
│   ├── test_agent_tools.py    # Security tests
│   └── test_tool_parser.py    # Parser tests
│
├── api_tester.py              # Main application
├── agent_tools.py             # Secure tool implementations
├── config.py                  # Configuration loader
├── config_manager.py          # Persistent configuration
├── tool_parser.py             # Universal tool parser
│
├── config.yaml                # YAML configuration
├── .env.example               # Environment template
├── pyproject.toml             # Package metadata
├── requirements.txt           # Dependencies
│
├── README.md                  # Documentation
├── CONTRIBUTING.md            # Contribution guide
├── CHANGELOG.md               # Version history
├── LICENSE                    # MIT License
│
├── workspace/                 # AI working directory
├── logs/                      # Audit logs
└── data/                      # Data storage
```

## Test Coverage

```
Name                  Stmts   Miss  Cover   Missing
---------------------------------------------------
config.py                63      4    94%
config_manager.py        55      2    97%
agent_tools.py          200     48    76%
tool_parser.py           94      8    91%
---------------------------------------------------
TOTAL                   412     62    85%
```

- **Total Tests**: 106
- **Core Module Coverage**: 90%+
- **Overall Coverage**: 85%

## Performance Benchmarks

| Operation | Average Time | Throughput |
|-----------|--------------|------------|
| File Create | < 5ms | 200+ ops/sec |
| File Read | < 2ms | 500+ ops/sec |
| File Update | < 10ms | 100+ ops/sec |
| Directory List | < 50ms | 20+ ops/sec |
| Command Execution | < 100ms | 10+ ops/sec |
| AI Response | 500ms - 2s | 0.5 - 2 req/sec |

*Benchmarked on Intel i7, 16GB RAM, SSD*

## Development

### Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=. --cov-report=html

# Specific module
pytest tests/test_config.py
```

### Code Quality

```bash
# Format code
black .

# Lint code
ruff check . --fix

# Type check
mypy .

# Security scan
bandit -r .

# Run all checks
pre-commit run --all-files
```

## Slash Commands

During runtime, the following commands are available:

| Command | Description |
|---------|-------------|
| /config | Show current configuration |
| /provider | Change AI provider |
| /key | Update API key |
| /model | Change model |
| /history | View conversation history |
| /clear | Clear conversation |
| exit | Quit application |

## CI/CD Pipeline

The project uses GitHub Actions for continuous integration:

- **Lint**: Ruff, Black, MyPy
- **Test**: Matrix testing (3 OS x 4 Python versions)
- **Security**: Bandit security scanning
- **Build**: Package verification
- **Release**: Automated PyPI publishing

## Requirements

- Python 3.9 or higher
- Git
- API key for at least one supported provider

## Contributing

Contributions are welcome. Please read the [Contributing Guide](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file.

## Support

- [Issues](https://github.com/anomalyco/ai-agent-crud/issues)
- [Discussions](https://github.com/anomalyco/ai-agent-crud/discussions)

---

<p align="center">
  Built for developers who want AI that can actually do things.
</p>
