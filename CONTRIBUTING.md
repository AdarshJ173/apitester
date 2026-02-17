# Contributing to AI Agent CRUD

First off, thank you for considering contributing to AI Agent CRUD! It's people like you that make this tool better for everyone.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
  - [Reporting Bugs](#reporting-bugs)
  - [Suggesting Features](#suggesting-features)
  - [Pull Requests](#pull-requests)
- [Style Guidelines](#style-guidelines)
  - [Python Code Style](#python-code-style)
  - [Commit Messages](#commit-messages)
  - [Documentation](#documentation)
- [Testing](#testing)
- [Security](#security)

## Code of Conduct

This project and everyone participating in it is governed by our commitment to:

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Respect different viewpoints and experiences

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
3. Create a new branch for your feature or fix
4. Make your changes
5. Run tests and linting
6. Submit a pull request

## Development Setup

### Prerequisites

- Python 3.9 or higher
- Git
- A code editor (VS Code, PyCharm, etc.)

### Setup Steps

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/ai-agent-crud.git
cd ai-agent-crud

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install in development mode
pip install -e ".[dev,test]"

# Install pre-commit hooks
pre-commit install
```

## How to Contribute

### Reporting Bugs

Before creating a bug report, please check the existing issues to see if the problem has already been reported.

When creating a bug report, please include:

- **Clear title and description**
- **Steps to reproduce** the issue
- **Expected behavior** vs actual behavior
- **Environment details**:
  - OS and version
  - Python version
  - Package version
- **Code samples** or screenshots if applicable
- **Logs** from `logs/audit.log` if relevant

Use the [Bug Report template](https://github.com/anomalyco/ai-agent-crud/issues/new?template=bug_report.md) when creating issues.

### Suggesting Features

Feature requests are welcome! Please provide:

- **Clear use case** - What problem does this solve?
- **Detailed description** - How should it work?
- **Possible alternatives** - Have you considered other approaches?

Use the [Feature Request template](https://github.com/anomalyco/ai-agent-crud/issues/new?template=feature_request.md) when suggesting features.

### Pull Requests

1. **Create a branch** from `main`:
   ```bash
   git checkout -b feature/my-feature
   # or
   git checkout -b fix/my-bugfix
   ```

2. **Make your changes** following our style guidelines

3. **Test your changes**:
   ```bash
   pytest
   ruff check .
   black --check .
   mypy .
   ```

4. **Commit your changes** with clear messages (see [Commit Messages](#commit-messages))

5. **Push to your fork**:
   ```bash
   git push origin feature/my-feature
   ```

6. **Open a Pull Request** using the PR template

## Style Guidelines

### Python Code Style

We follow PEP 8 with some modifications:

- **Line length**: 100 characters (not 80)
- **Quotes**: Double quotes for strings
- **Imports**: Sorted with isort (black profile)

We use automated tools to enforce style:

```bash
# Format code
black .

# Fix imports
isort .

# Lint code
ruff check . --fix
```

### Type Hints

Please add type hints to new code:

```python
def process_data(data: dict[str, Any]) -> list[str]:
    """Process data and return list of strings."""
    return [str(v) for v in data.values()]
```

### Docstrings

Use Google-style docstrings:

```python
def my_function(param1: str, param2: int) -> bool:
    """Short description of what the function does.

    Longer description if needed, explaining the purpose
    and any important details.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ValueError: When param2 is negative
    """
    if param2 < 0:
        raise ValueError("param2 must be non-negative")
    return len(param1) == param2
```

### Commit Messages

We follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

Format:
```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

Examples:
```
feat(tools): Add support for file globbing

fix(agent): Handle timeout errors gracefully

docs(readme): Update installation instructions
```

### Documentation

- Update README.md if you change user-facing features
- Add docstrings to new functions and classes
- Update CHANGELOG.md with your changes
- Include code examples where helpful

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_config.py

# Run specific test
pytest tests/test_config.py::TestLoadYamlConfig
```

### Writing Tests

- Add tests for new features
- Update tests for bug fixes
- Aim for >80% coverage on new code
- Use descriptive test names
- Use fixtures for setup/teardown

Example:
```python
def test_my_feature_handles_edge_case(tmp_path):
    """Test that my_feature correctly handles edge case X."""
    # Arrange
    test_file = tmp_path / "test.txt"
    test_file.write_text("content")
    
    # Act
    result = my_feature(test_file)
    
    # Assert
    assert result == expected_value
```

## Security

Security is critical for this project. Please:

- **Never commit API keys or secrets**
- **Report security issues privately** via email, not public issues
- **Follow security best practices** in your code
- **Use the security audit tools**:
  ```bash
  bandit -r .
  ```

### Security Checklist

- [ ] No hardcoded credentials
- [ ] Input validation on all user inputs
- [ ] Path traversal protection
- [ ] Command injection prevention
- [ ] Audit logging for sensitive operations

## Questions?

Feel free to:
- Open an issue for questions
- Join discussions in existing issues
- Reach out to maintainers

Thank you for contributing! 🎉
