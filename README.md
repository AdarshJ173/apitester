# Universal API Tester 🚀

Test API keys from NVIDIA, OpenRouter, Groq, OpenAI, and Anthropic with a beautiful terminal interface.

## Setup Instructions

### Step 1: Create Virtual Environment

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Run the Program

```bash
python api_tester.py
```

## Usage

1. **Enter your API key** when prompted
2. The program will auto-detect the service (NVIDIA, OpenRouter, Groq, etc.)
3. **Type your prompt** and press Enter to get a response
4. **Type /** to see all available models and switch between them
5. **Type 'exit'** to quit

## Supported Services

- ✅ NVIDIA (nvapi-*)
- ✅ OpenRouter (sk-or-*)
- ✅ Groq (gsk_*)
- ✅ OpenAI (sk-*)
- ✅ Anthropic (sk-ant-*)

## Features

✨ Auto-detects API service from key format
✨ Lists all available models
✨ Beautiful terminal UI with colors
✨ Clear error messages with reasons
✨ Easy model switching with /
✨ Minimal and clean interface

## Troubleshooting

**If you get "command not found" for python:**
- Try `python3` instead of `python`
- Try `py` on Windows

**If installation fails:**
- Make sure you're in the virtual environment (you should see `(venv)` in your terminal)
- Try upgrading pip: `pip install --upgrade pip`
