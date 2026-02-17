# 🤖 AI Agent with Secure CRUD Operations

**Enterprise-grade AI assistant with file system control, command execution, and conversation memory.**

## 🎯 Features

### Core Capabilities
- ✅ **Full CRUD Operations**: Create, Read, Update, Delete files
- ✅ **Directory Management**: List and navigate workspace
- ✅ **Safe Command Execution**: Run shell commands with security validation
- ✅ **Context Memory**: AI remembers file contents across conversation
- ✅ **Multi-Provider Support**: OpenAI, Anthropic, Groq, OpenRouter
- ✅ **Function Calling**: Native tool use with latest AI models

### Security Features
- 🔒 **Sandboxed Workspace**: AI can only access designated folders
- 🔒 **Command Validation**: Blocks dangerous commands (rm, format, etc.)
- 🔒 **Path Traversal Protection**: Prevents directory escaping
- 🔒 **File Type Restrictions**: Whitelist/blacklist for extensions
- 🔒 **Size Limits**: Max 10MB per file
- 🔒 **Audit Logging**: All actions logged to `logs/audit.log`
- 🔒 **Automatic Backups**: Creates .backup files before updates

### Performance
- ⚡ **Fast Algorithms**: Optimized deque for conversation history
- ⚡ **Model Caching**: Reduces API calls
- ⚡ **Thread-Safe**: Concurrent operations supported
- ⚡ **Timeout Protection**: 30s command, 120s API timeouts
- ⚡ **Retry Logic**: Auto-retry for transient failures

---

## 🚀 Quick Start

### Step 1: Setup Environment

**Windows:**
```bash
python -m venv venv
venv\\Scripts\\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run
```bash
python api_tester.py
```

---

## 📖 Usage Guide

### First Run
1. **Select Provider**: Choose from OpenAI, Anthropic, Groq, OpenRouter
2. **Enter API Key**: Provide your API key for the selected provider
3. **Select Model**: Choose a tool-capable model (GPT-4, Claude 3.5, etc.)

### Example Commands

#### File Operations
```
You: Create a file called workspace/notes.txt with "Hello World"
AI: ✓ Created workspace/notes.txt

You: Read workspace/notes.txt
AI: Content: Hello World

You: Update workspace/notes.txt with "Updated content"
AI: ✓ Updated workspace/notes.txt (backup created)

You: List all files in workspace
AI: Found 3 items:
    - notes.txt (12 bytes)
    - data.json (256 bytes)
    ...
```

#### Advanced Usage
```
You: Create a Python script that calculates fibonacci numbers
AI: [Creates workspace/fibonacci.py]

You: Read that file and explain it
AI: [Reads file, content becomes part of context]
    The script implements...

You: Now create a test file for it
AI: [Uses context from previous read to create test]
```

### Slash Commands

| Command | Description |
|---------|-------------|
| `/config` | Show current configuration |
| `/provider` | Change AI provider |
| `/key` | Update API key |
| `/model` | Change model |
| `/history` | View conversation history |
| `exit` | Quit application |

---

## 🗂️ Project Structure

```
.
├── api_tester.py          # Main application
├── agent_tools.py         # Secure tool implementations
├── config.py              # Security configuration
├── requirements.txt       # Python dependencies
├── workspace/             # AI working directory (auto-created)
├── data/                  # Data files directory (auto-created)
└── logs/
    └── audit.log          # Action audit trail (auto-created)
```

---

## 🔒 Security Model

### Allowed Directories
AI can only access:
- `workspace/` - Main working directory
- `data/` - Data storage
- `logs/` - Log files

### Blocked Operations
- ❌ System directories (`/etc`, `/sys`, `C:\\Windows`)
- ❌ Dangerous commands (`rm -rf`, `format`, `shutdown`)
- ❌ Binary executables (`.exe`, `.dll`, `.so`)
- ❌ Path traversal (`../`, `~`)
- ❌ Command injection (`;`, `&&`, `|`)

### Allowed File Types
`.txt`, `.json`, `.yaml`, `.md`, `.csv`, `.py`, `.js`, `.html`, `.css`, `.xml`, `.log`

---

## 🛠️ Advanced Configuration

### Customize Security Settings

Edit `config.py`:

```python
SECURITY = {
    "max_file_size_mb": 10,  # Increase size limit
    "allowed_extensions": [".txt", ".json", ...],  # Add extensions
    "allowed_directories": [BASE_DIR / "custom_folder"],  # Add folders
}
```

### Use Different AI Providers

#### OpenAI (GPT-4)
```bash
# Recommended models: gpt-4o, gpt-4-turbo
API Key format: sk-...
```

#### Anthropic (Claude)
```bash
# Recommended: claude-3-5-sonnet-20241022
API Key format: sk-ant-...
```

#### Groq
```bash
# Recommended: llama3-groq-70b-8192-tool-use-preview
API Key format: gsk_...
```

---

## 📊 Audit Logging

All operations are logged to `logs/audit.log`:

```json
{
  "timestamp": "2026-02-11T10:45:23.123456",
  "action": "create_file",
  "details": {"path": "workspace/test.txt", "size": 42},
  "status": "success"
}
```

---

## 🐛 Troubleshooting

### "Access denied: Path must be within workspace"
**Solution**: Use relative paths starting with `workspace/`
```
✗ /tmp/file.txt
✓ workspace/file.txt
```

### "Command blocked: Dangerous command detected"
**Solution**: Only safe commands allowed. Dangerous operations require manual execution.

### "File type not in allowed list"
**Solution**: Check `config.py` and add extension to `allowed_extensions`

### API Errors (401, 403)
**Solution**: 
- Verify API key with `/key`
- Check provider account status
- Ensure model access permissions

---

## 🎓 Best Practices

1. **Start Simple**: Test with basic file operations first
2. **Check Context**: Use `/history` to see conversation memory
3. **Review Audit Log**: Monitor `logs/audit.log` for actions
4. **Backup Important Files**: AI creates backups, but manual backups recommended
5. **Use Specific Paths**: Always include `workspace/` prefix

---

## 🔧 Technical Details

### Conversation Memory
- Uses `collections.deque` with max 20 messages (O(1) operations)
- File contents automatically injected into context when read
- System prompt provides AI with operational guidelines

### Tool Execution Flow
```
User Request → AI Analyzes → Selects Tools → Security Check 
→ Execute → Inject Results → AI Responds
```

### Performance Optimizations
- Model caching: Reduces repeated API calls
- Lazy directory creation: Only creates folders when needed
- Efficient path resolution: Single pass validation
- Thread-safe spinner: Non-blocking UI updates

---

## 📝 License

MIT License - Use freely for personal and commercial projects

---

## 🤝 Contributing

Found a bug? Have a feature request?
- Open an issue with detailed description
- Include audit log excerpt if relevant
- Specify provider and model used

---

## ⚠️ Disclaimer

This tool provides AI with file system access. While security measures are in place:
- Always run in isolated environments for sensitive work
- Review audit logs regularly
- Never use with untrusted API keys
- Understand your AI provider's data policies

---

## 🎉 Happy Building!

You now have a powerful AI assistant with secure file system control. Start with simple tasks and gradually build more complex workflows.

**Example Project Ideas:**
- Automated documentation generator
- Code file organizer
- Log file analyzer
- Data transformation pipeline
- Configuration file manager

---

**Made with ❤️ for developers who want AI that can actually DO things**
