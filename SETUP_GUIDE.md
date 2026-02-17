# 🚀 Quick Setup Guide

## What You Got

A **professional, enterprise-grade AI agent** that can:
- 📝 Create, read, update, delete files
- 📂 List and manage directories  
- 💻 Execute safe shell commands
- 🧠 Remember conversation context (reads files into memory)
- 🔒 Highly secure with audit logging

## Files Included

1. **api_tester.py** - Main application (AI agent with tool use)
2. **agent_tools.py** - Secure CRUD implementations
3. **config.py** - Security settings (customizable)
4. **requirements.txt** - Dependencies
5. **README.md** - Full documentation

## 3-Step Setup

### 1️⃣ Install Dependencies
```bash
# Make sure venv is activated
pip install -r requirements.txt
```

### 2️⃣ Run the Program
```bash
python api_tester.py
```

### 3️⃣ Follow Prompts
- Select AI provider (OpenAI, Anthropic, Groq, etc.)
- Enter API key
- Select model (tool-capable models recommended)

## First Test

Try this:
```
You: Create a file called workspace/hello.txt with content "AI is working!"
AI: [Creates file]

You: Read workspace/hello.txt
AI: [Shows content - now in context memory]

You: List all files in workspace
AI: [Shows directory listing]
```

## Security Features ✅

- **Sandboxed**: AI can only access `workspace/`, `data/`, `logs/`
- **Command blocking**: Dangerous commands auto-blocked
- **Path protection**: No `../` or system directory access
- **File type filtering**: Only safe extensions allowed
- **Audit logging**: Everything tracked in `logs/audit.log`
- **Auto-backups**: Creates .backup before updating files

## Supported Providers

| Provider | Tool Support | Example Models |
|----------|--------------|----------------|
| OpenAI | ✅ Yes | gpt-4o, gpt-4-turbo |
| Anthropic | ✅ Yes | claude-3-5-sonnet |
| Groq | ✅ Yes | llama3-groq-70b-tool-use |
| OpenRouter | ✅ Yes | Various |

## Performance

- ⚡ **Fast**: Optimized deque (O(1) operations)
- ⚡ **Cached**: Models cached to reduce API calls
- ⚡ **Safe**: 30s command timeout, 120s API timeout
- ⚡ **Reliable**: Auto-retry with exponential backoff

## Commands During Chat

- `/config` - View current setup
- `/provider` - Change AI provider
- `/history` - See conversation memory
- `exit` - Quit

## Example Use Cases

✅ "Create a Python script that sorts a list"
✅ "Read config.json and explain what it does"  
✅ "List all .txt files in workspace"
✅ "Update README.md to add a new section"
✅ "Execute command: ls -la workspace"

## Need Help?

Check **README.md** for:
- Detailed documentation
- Troubleshooting guide
- Security configuration
- Advanced examples

---

**You're all set! 🎉**

Run `python api_tester.py` and start building.
