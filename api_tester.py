"""
Universal API Tester with AI Agent Capabilities
Fixed: Tool message formatting for OpenAI/Groq
"""

import time
import threading
import json
import os
from typing import Dict, List, Optional
from collections import deque

import requests
import questionary
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel
from rich.table import Table
from rich import box
from rich.markdown import Markdown

from config import ensure_directories
from agent_tools import execute_tool, TOOLS_DEFINITIONS
from config_manager import config_manager
from tool_parser import ToolParser, UniversalToolExecutor

console = Console()
ensure_directories()


class Spinner:
    """Thread-safe spinner"""

    def __init__(self, message: str = "Processing"):
        self.message = message
        self._running = False
        self._thread = None
        self._lock = threading.Lock()

    def _spin(self):
        frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        i = 0
        while True:
            with self._lock:
                if not self._running:
                    break
            console.print(
                f"[dim]{frames[i % len(frames)]} {self.message}...[/dim]", end="\r"
            )
            time.sleep(0.08)
            i += 1
        console.print(" " * 100, end="\r")

    def start(self):
        with self._lock:
            if self._running:
                return
            self._running = True
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def stop(self):
        with self._lock:
            self._running = False
        if self._thread:
            self._thread.join(timeout=0.5)
        console.print(" " * 100, end="\r")


class ConversationManager:
    """Conversation history with context memory"""

    def __init__(self, max_messages: int = 20):
        self.messages: deque = deque(maxlen=max_messages)
        self._pending_context: List[
            Dict
        ] = []  # Context messages queued during tool cycles
        self.use_text_tools = False
        self.system_prompt_native = """You are a helpful AI assistant with access to file system tools.

You can create, read, update, and delete files, list directories, and execute safe commands.

IMPORTANT: Always use relative paths (e.g., 'workspace/test.txt' or 'test.py'). The current directory is the project root.

When you read files, their content becomes part of your context - reference it in responses."""

    @property
    def system_prompt(self) -> str:
        """Get the appropriate system prompt based on tool mode"""
        if self.use_text_tools:
            return ToolParser.create_tool_system_prompt()
        return self.system_prompt_native

    def set_tool_mode(self, use_text: bool):
        """Set whether to use text-based tool parsing"""
        self.use_text_tools = use_text

    def add_message(self, role: str, content: str, **kwargs):
        """Add message with optional extra fields"""
        msg = {"role": role, "content": content}
        msg.update(kwargs)
        self.messages.append(msg)

    def inject_file_context(self, file_path: str, content: str):
        """Queue file context to be added after the tool cycle completes.

        CRITICAL: We must NOT insert system messages between an assistant
        message with tool_calls and the corresponding tool result messages.
        The OpenAI/Groq API requires tool results to immediately follow
        the assistant tool_calls message.
        """
        context_msg = {
            "role": "system",
            "content": f"[File Context: {file_path}]\n{content}",
        }
        self._pending_context.append(context_msg)

    def flush_pending_context(self):
        """Add any queued file context messages to the conversation.
        Call this AFTER the entire tool cycle is complete."""
        for msg in self._pending_context:
            self.messages.append(msg)
        self._pending_context.clear()

    def get_messages(self) -> List[Dict]:
        """Build properly formatted message list for the API.

        Ensures:
        - System prompt is always first
        - tool messages always have tool_call_id
        - assistant messages with tool_calls have proper structure
        - No system messages break the tool call chain
        """
        result = [{"role": "system", "content": self.system_prompt}]
        for msg in self.messages:
            clean_msg = dict(msg)  # Copy to avoid mutating stored messages
            result.append(clean_msg)
        return result

    def clear(self):
        self.messages.clear()
        self._pending_context.clear()


class AIAgent:
    def __init__(self):
        self.api_key: Optional[str] = None
        self.service: Optional[str] = None
        self.current_model: Optional[str] = None
        self.conversation = ConversationManager()
        self._model_cache: Dict[str, List[str]] = {}

        self.services: Dict[str, Dict] = {
            "openai": {
                "label": "OpenAI",
                "base_url": "https://api.openai.com/v1",
                "models_endpoint": "/models",
                "chat_endpoint": "/chat/completions",
                "auth_header": "Authorization",
                "auth_prefix": "Bearer",
                "supports_tools": True,
                "style": "openai",
                "tool_mode": "text",
            },
            "anthropic": {
                "label": "Anthropic (Claude)",
                "base_url": "https://api.anthropic.com/v1",
                "models_endpoint": None,
                "chat_endpoint": "/messages",
                "auth_header": "x-api-key",
                "auth_prefix": None,
                "supports_tools": True,
                "style": "anthropic",
                "models": [
                    "claude-3-5-sonnet-20241022",
                    "claude-3-5-haiku-20241022",
                    "claude-3-opus-20240229",
                ],
                "tool_mode": "text",
            },
            "groq": {
                "label": "Groq",
                "base_url": "https://api.groq.com/openai/v1",
                "models_endpoint": "/models",
                "chat_endpoint": "/chat/completions",
                "auth_header": "Authorization",
                "auth_prefix": "Bearer",
                "supports_tools": True,
                "style": "openai",
                "tool_mode": "text",
            },
            "openrouter": {
                "label": "OpenRouter",
                "base_url": "https://openrouter.ai/api/v1",
                "models_endpoint": "/models",
                "chat_endpoint": "/chat/completions",
                "auth_header": "Authorization",
                "auth_prefix": "Bearer",
                "supports_tools": True,
                "style": "openai",
                "tool_mode": "text",
            },
            "nvidia": {
                "label": "NVIDIA",
                "base_url": "https://integrate.api.nvidia.com/v1",
                "models_endpoint": "/models",
                "chat_endpoint": "/chat/completions",
                "auth_header": "Authorization",
                "auth_prefix": "Bearer",
                "supports_tools": True,
                "style": "openai",
                "tool_mode": "text",  # Uses text-based tool parsing
            },
            "together": {
                "label": "Together AI",
                "base_url": "https://api.together.xyz/v1",
                "models_endpoint": "/models",
                "chat_endpoint": "/chat/completions",
                "auth_header": "Authorization",
                "auth_prefix": "Bearer",
                "supports_tools": True,
                "style": "openai",
                "tool_mode": "text",  # Uses text-based tool parsing
            },
            "mistral": {
                "label": "Mistral AI",
                "base_url": "https://api.mistral.ai/v1",
                "models_endpoint": "/models",
                "chat_endpoint": "/chat/completions",
                "auth_header": "Authorization",
                "auth_prefix": "Bearer",
                "supports_tools": True,
                "style": "openai",
                "tool_mode": "text",  # Uses text-based tool parsing
            },
            "cohere": {
                "label": "Cohere",
                "base_url": "https://api.cohere.ai/v1",
                "models_endpoint": None,
                "chat_endpoint": "/chat",
                "auth_header": "Authorization",
                "auth_prefix": "Bearer",
                "supports_tools": True,
                "style": "cohere",
                "models": ["command-r-plus", "command-r", "command-light"],
                "tool_mode": "text",  # Uses text-based tool parsing
            },
        }

    def get_headers(self, service: str) -> Dict[str, str]:
        config = self.services[service]
        auth_header = config["auth_header"]
        prefix = config.get("auth_prefix")
        auth_value = f"{prefix} {self.api_key}" if prefix else self.api_key
        headers = {auth_header: auth_value, "content-type": "application/json"}
        if service == "anthropic":
            headers["anthropic-version"] = "2023-06-01"
        return headers

    def select_provider(self) -> Optional[str]:
        choices = []
        for key, cfg in self.services.items():
            tools_indicator = (
                "🔧 Tools" if cfg.get("supports_tools") else "💬 Chat Only"
            )
            choices.append(
                questionary.Choice(
                    title=f"{cfg['label']:<20} [{tools_indicator}]", value=key
                )
            )

        try:
            return questionary.select(
                "Select AI provider",
                choices=choices,
                qmark="🤖",
                pointer="➤",
            ).ask()
        except (KeyboardInterrupt, EOFError):
            return None

    def ask_api_key(self, provider: str) -> Optional[str]:
        config = self.services[provider]

        # Check environment variable first
        env_var = f"{provider.upper()}_API_KEY"
        if os.environ.get(env_var):
            return os.environ.get(env_var)

        try:
            key = Prompt.ask(
                f"[bold]Enter API key for {config['label']}[/bold]"
            ).strip()
            return key if key else None
        except (KeyboardInterrupt, EOFError):
            return None

    def fetch_models(self, service: str, use_cache: bool = True) -> List[str]:
        if use_cache and service in self._model_cache:
            return self._model_cache[service]

        config = self.services[service]

        if config.get("models"):
            models = config["models"]
            self._model_cache[service] = models
            return models

        if not config.get("models_endpoint"):
            return []

        url = config["base_url"] + config["models_endpoint"]
        headers = self.get_headers(service)

        try:
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                models: List[str] = []

                if isinstance(data, dict) and "data" in data:
                    for m in data["data"]:
                        if isinstance(m, dict) and m.get("id"):
                            models.append(m["id"])
                elif isinstance(data, list):
                    for m in data:
                        if isinstance(m, dict) and m.get("id"):
                            models.append(m["id"])

                if service == "groq":
                    # Filter for chat models, excluding guard models
                    chat_models = [
                        m
                        for m in models
                        if any(
                            x in m.lower()
                            for x in ["llama", "mixtral", "gemma", "deepseek"]
                        )
                        and "guard" not in m.lower()
                    ]

                    if chat_models:
                        # Prioritize higher capability models
                        def model_priority(name):
                            name_lower = name.lower()
                            if "llama-3.3-70b" in name_lower:
                                return 0
                            if "llama-3.1-70b" in name_lower:
                                return 1
                            if "mixtral-8x7b" in name_lower:
                                return 2
                            if "llama-3.1-8b" in name_lower:
                                return 3
                            return 10

                        chat_models.sort(key=model_priority)
                        models = chat_models

                if models:
                    self._model_cache[service] = models
                    return models

            elif resp.status_code == 401:
                console.print(f"[red]❌ Authentication failed (401)[/red]")
                return []

        except Exception as e:
            console.print(f"[yellow]⚠ Error: {e}[/yellow]")
            return []

        return []

    def select_model(self, models: List[str]) -> Optional[str]:
        if not models:
            console.print("[yellow]No models available[/yellow]")
            return None

        try:
            display_models = models[:50]
            if len(models) > 50:
                console.print(f"[dim]Showing first 50 of {len(models)} models[/dim]")

            answer = questionary.select(
                "Select model",
                choices=display_models,
                qmark="📋",
                pointer="➤",
            ).ask()
            return answer
        except (KeyboardInterrupt, EOFError):
            return None

    def send_simple_message(self, user_message: str) -> Optional[str]:
        """Send message and optionally parse text-based tools"""
        if not self.service:
            return "❌ No provider selected"
        config = self.services[self.service]
        url = config["base_url"] + config["chat_endpoint"]
        headers = self.get_headers(self.service)

        self.conversation.add_message("user", user_message)

        spinner = Spinner("AI is thinking")
        spinner.start()
        start = time.perf_counter()

        try:
            if self.service == "anthropic":
                payload = {
                    "model": self.current_model,
                    "max_tokens": 4096,
                    "messages": [
                        m
                        for m in self.conversation.get_messages()
                        if m["role"] != "system"
                    ],
                    "system": self.conversation.system_prompt,
                }
            elif self.service == "cohere":
                payload = {
                    "model": self.current_model,
                    "message": user_message,
                }
            else:
                # For text-based tool providers, only send user/assistant messages (no system)
                if config.get("tool_mode") == "text":
                    payload = {
                        "model": self.current_model,
                        "messages": [
                            {
                                "role": "system",
                                "content": self.conversation.system_prompt,
                            },
                            {"role": "user", "content": user_message},
                        ],
                    }
                else:
                    payload = {
                        "model": self.current_model,
                        "messages": self.conversation.get_messages(),
                    }

            resp = requests.post(url, headers=headers, json=payload, timeout=120)
            elapsed = time.perf_counter() - start
            spinner.stop()

            if resp.status_code == 200:
                data = resp.json()

                if self.service == "anthropic":
                    content = data["content"][0]["text"]
                elif self.service == "cohere":
                    content = data.get("text") or data.get("message") or str(data)
                else:
                    content = data["choices"][0]["message"]["content"]

                # Check if this provider uses text-based tool parsing
                if config.get("tool_mode") == "text":
                    # Parse and execute tools from text
                    cleaned_content, results = UniversalToolExecutor(
                        execute_tool
                    ).process_response(content)

                    # Show tool execution status in console
                    if results:
                        console.print()
                        for result in results:
                            tool_name = result["tool"]
                            tool_result = result["result"]
                            if tool_result.get("success"):
                                console.print(f"[green]✓[/green] {tool_name}")
                            else:
                                console.print(
                                    f"[red]✗[/red] {tool_name}: {tool_result.get('error', 'Unknown error')}"
                                )

                    # Add results to conversation
                    for result in results:
                        self.conversation.add_message(
                            "system",
                            f"[Tool Result: {result['tool']}]\n{json.dumps(result['result'])}",
                        )

                    self.conversation.add_message("assistant", cleaned_content)
                    return f"{cleaned_content}\n\n[dim]⏱ {elapsed:.2f}s ({int(elapsed * 1000)}ms)[/dim]"
                else:
                    self.conversation.add_message("assistant", content)
                    return f"{content}\n\n[dim]⏱ {elapsed:.2f}s ({int(elapsed * 1000)}ms)[/dim]"
            else:
                try:
                    err = resp.json().get("error", {}).get("message", resp.text)
                except:
                    err = resp.text[:300]
                return f"❌ Error ({resp.status_code}): {err}"

        except requests.exceptions.Timeout:
            spinner.stop()
            return "❌ Request timed out (120s)"
        except Exception as e:
            spinner.stop()
            return f"❌ Error: {e}"

    def _sanitize_messages_for_api(self, messages: List[Dict]) -> List[Dict]:
        """Sanitize messages to ensure proper format for OpenAI-style APIs.

        Rules enforced:
        1. Every 'tool' role message MUST have a 'tool_call_id' field
        2. Tool messages must follow an assistant message that has 'tool_calls'
        3. No system messages between assistant tool_calls and tool results
        4. Remove orphaned tool messages that lack proper context
        """
        sanitized = []
        i = 0
        while i < len(messages):
            msg = messages[i]
            role = msg.get("role")

            if role == "assistant" and msg.get("tool_calls"):
                # Add the assistant message with tool_calls
                sanitized.append(msg)
                i += 1

                # Collect all subsequent tool messages first, defer system messages
                deferred_system = []
                while i < len(messages):
                    next_msg = messages[i]
                    if next_msg.get("role") == "tool":
                        if next_msg.get("tool_call_id"):
                            sanitized.append(next_msg)
                        # Skip tool messages without tool_call_id (broken)
                        i += 1
                    elif next_msg.get("role") == "system":
                        # Defer system messages until after ALL tool messages
                        deferred_system.append(next_msg)
                        i += 1
                    else:
                        break
                # Now add any deferred system messages AFTER all tool results
                sanitized.extend(deferred_system)
            elif role == "tool":
                # Orphaned tool message (no preceding assistant with tool_calls)
                # Skip it to prevent API errors
                i += 1
            else:
                sanitized.append(msg)
                i += 1

        return sanitized

    def call_ai_with_tools(self, user_message: str) -> Optional[str]:
        """Call AI with tool support - handles tool message formatting correctly."""
        if not self.service:
            return "❌ No provider selected"
        config = self.services[self.service]
        url = config["base_url"] + config["chat_endpoint"]
        headers = self.get_headers(self.service)

        self.conversation.add_message("user", user_message)

        spinner = Spinner("AI is thinking")
        spinner.start()
        start = time.perf_counter()

        max_iterations = 5
        iteration = 0

        try:
            while iteration < max_iterations:
                iteration += 1

                if self.service == "anthropic":
                    # Anthropic: filter system messages out, pass as top-level param
                    api_messages = [
                        m
                        for m in self.conversation.get_messages()
                        if m["role"] != "system"
                    ]
                    payload = {
                        "model": self.current_model,
                        "max_tokens": 4096,
                        "messages": api_messages,
                        "system": self.conversation.system_prompt,
                        "tools": TOOLS_DEFINITIONS,
                    }
                else:
                    # OpenAI-style: sanitize messages to fix tool chain ordering
                    api_messages = self._sanitize_messages_for_api(
                        self.conversation.get_messages()
                    )
                    payload = {
                        "model": self.current_model,
                        "messages": api_messages,
                        "tools": TOOLS_DEFINITIONS,
                        "tool_choice": "auto",
                    }

                resp = requests.post(url, headers=headers, json=payload, timeout=120)

                if resp.status_code != 200:
                    spinner.stop()
                    try:
                        err_data = resp.json()
                        err = err_data.get("error", {}).get("message", resp.text)
                    except Exception:
                        err = resp.text[:300]

                    # If the error is about tool calling not being supported by the model,
                    # fall back to simple message mode
                    err_lower = str(err).lower()
                    if resp.status_code == 400 and (
                        "tool" in err_lower
                        and (
                            "not supported" in err_lower or "not available" in err_lower
                        )
                    ):
                        # Remove the user message we just added (it will be re-added by send_simple_message)
                        if (
                            self.conversation.messages
                            and self.conversation.messages[-1].get("role") == "user"
                        ):
                            self.conversation.messages.pop()
                        return self.send_simple_message(user_message)

                    return f"❌ Error ({resp.status_code}): {err}"

                data = resp.json()

                if self.service == "anthropic":
                    content_blocks = data.get("content", [])
                    tool_uses = [
                        b for b in content_blocks if b.get("type") == "tool_use"
                    ]

                    if tool_uses:
                        spinner.stop()
                        for tool_use in tool_uses:
                            tool_name = tool_use["name"]
                            tool_args = tool_use["input"]

                            console.print(
                                f"[dim]🔧 {tool_name}({json.dumps(tool_args)})[/dim]"
                            )
                            result = execute_tool(tool_name, tool_args)

                            if result.get("success"):
                                console.print(f"[green]✓ {tool_name}[/green]")
                                if tool_name == "read_file" and "content" in result:
                                    # self.conversation.inject_file_context(result["path"], result["content"])
                                    pass
                            else:
                                console.print(
                                    f"[red]✗ {tool_name}: {result.get('error')}[/red]"
                                )

                            self.conversation.add_message(
                                "user",
                                json.dumps(
                                    {
                                        "type": "tool_result",
                                        "tool_use_id": tool_use["id"],
                                        "content": json.dumps(result),
                                    }
                                ),
                            )

                        # Flush any pending file context AFTER all tool results are added
                        self.conversation.flush_pending_context()
                        spinner.start()
                        continue

                    text_blocks = [
                        b["text"] for b in content_blocks if b.get("type") == "text"
                    ]
                    if text_blocks:
                        response_text = "\n".join(text_blocks)
                        self.conversation.flush_pending_context()
                        self.conversation.add_message("assistant", response_text)
                        spinner.stop()
                        elapsed = time.perf_counter() - start
                        return f"{response_text}\n\n[dim]⏱ {elapsed:.2f}s ({int(elapsed * 1000)}ms)[/dim]"

                else:  # OpenAI-style (Groq, OpenAI, OpenRouter)
                    choice = data["choices"][0]
                    message = choice["message"]

                    if message.get("tool_calls"):
                        spinner.stop()

                        # First, add the assistant message with tool_calls
                        # Include content only if present (some APIs send null)
                        assistant_content = message.get("content") or ""
                        self.conversation.add_message(
                            "assistant",
                            assistant_content,
                            tool_calls=message["tool_calls"],
                        )

                        # Execute ALL tools and add ALL tool responses BEFORE doing anything else
                        for tool_call in message["tool_calls"]:
                            func = tool_call.get("function", {})
                            tool_name = func.get("name", "unknown")
                            tool_call_id = tool_call.get("id", "")

                            try:
                                tool_args = json.loads(func.get("arguments", "{}"))
                            except json.JSONDecodeError:
                                tool_args = {}

                            console.print(
                                f"[dim]🔧 {tool_name}({json.dumps(tool_args)})[/dim]"
                            )
                            result = execute_tool(tool_name, tool_args)

                            if result.get("success"):
                                console.print(f"[green]✓ {tool_name}[/green]")
                                # For read_file, we need to make sure the model actually Sees the content.
                                # Instead of system messages (which can break tool chains), we'll rely on the
                                # tool output itself being part of the conversation history.
                                # The 'content' field in the tool output JSON is what the model sees.
                                pass
                            else:
                                console.print(
                                    f"[red]✗ {tool_name}: {result.get('error')}[/red]"
                                )

                            # Add tool result message with REQUIRED tool_call_id
                            # Add tool result message with REQUIRED tool_call_id
                            content_str = json.dumps(result)
                            self.conversation.add_message(
                                "tool",
                                content_str,
                                tool_call_id=tool_call_id,
                                name=tool_name,
                            )

                        # NOW flush pending file context (after ALL tool results are in place)
                        self.conversation.flush_pending_context()

                        spinner.start()
                        continue

                    # No tool calls — regular text response
                    response_text = message.get("content", "")
                    if response_text:
                        self.conversation.flush_pending_context()
                        self.conversation.add_message("assistant", response_text)
                        spinner.stop()
                        elapsed = time.perf_counter() - start
                        return f"{response_text}\n\n[dim]⏱ {elapsed:.2f}s ({int(elapsed * 1000)}ms)[/dim]"

                spinner.stop()
                return str(data)

            spinner.stop()
            return "⚠ Max iterations reached"

        except requests.exceptions.Timeout:
            spinner.stop()
            return "❌ Request timed out (120s)"
        except json.JSONDecodeError as e:
            spinner.stop()
            return f"❌ Invalid JSON in tool arguments: {e}"
        except requests.exceptions.ConnectionError as e:
            spinner.stop()
            if "10013" in str(e):
                return f"❌ Network access denied (WinError 10013). Check your firewall, antivirus, or VPN settings."
            return f"❌ Connection error: {e}"
        except Exception as e:
            spinner.stop()
            return f"❌ Error: {e}"

    def show_config(self):
        table = Table(title="🔧 Configuration", box=box.ROUNDED)
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")

        if self.service:
            cfg = self.services[self.service]
            table.add_row("Provider", cfg["label"])
            table.add_row(
                "Tools Support", "✅ Yes" if cfg.get("supports_tools") else "❌ No"
            )
        if self.current_model:
            table.add_row("Model", self.current_model)
        if self.api_key:
            masked = self.api_key[:8] + "..." + self.api_key[-4:]
            table.add_row("API Key", masked)

        table.add_row("Context Messages", str(len(self.conversation.messages)))

        console.print(table)

    def setup(self) -> bool:
        console.clear()
        console.print(
            Panel.fit(
                "[bold cyan]🤖 AI Agent with Secure CRUD Operations[/bold cyan]\n"
                "[dim]File operations • Command execution • Context memory[/dim]\n"
                "[yellow]Commands: /config, /provider, /key, /model, /clear, exit[/yellow]",
                border_style="cyan",
                box=box.DOUBLE,
            )
        )

        # Check for saved configurations
        last_provider = config_manager.get_last_provider()
        saved_providers = config_manager.get_all_saved_providers()

        if saved_providers:
            # Show recommendation menu
            choices = []

            # Add last used as first option with star
            if last_provider and last_provider in saved_providers:
                last_cfg = saved_providers[last_provider]
                provider_label = self.services[last_provider]["label"]
                masked_key = (
                    last_cfg.api_key[:8] + "..." + last_cfg.api_key[-4:]
                    if len(last_cfg.api_key) > 12
                    else "***"
                )
                choices.append(
                    questionary.Choice(
                        title=f"⭐ Last Used: {provider_label} ({last_cfg.model}) [Key: {masked_key}]",
                        value={
                            "type": "saved",
                            "provider": last_provider,
                            "config": last_cfg,
                        },
                    )
                )

            # Add other saved providers
            for provider_name, provider_cfg in saved_providers.items():
                if provider_name != last_provider:
                    provider_label = self.services[provider_name]["label"]
                    masked_key = (
                        provider_cfg.api_key[:8] + "..." + provider_cfg.api_key[-4:]
                        if len(provider_cfg.api_key) > 12
                        else "***"
                    )
                    choices.append(
                        questionary.Choice(
                            title=f"💾 {provider_label} ({provider_cfg.model}) [Key: {masked_key}]",
                            value={
                                "type": "saved",
                                "provider": provider_name,
                                "config": provider_cfg,
                            },
                        )
                    )

            # Add option for new configuration
            choices.append(questionary.Separator())
            choices.append(
                questionary.Choice(
                    title="➕ Configure New Provider", value={"type": "new"}
                )
            )

            try:
                selection = questionary.select(
                    "Select configuration (use saved or create new)",
                    choices=choices,
                    qmark="🤖",
                    pointer="➤",
                ).ask()

                if not selection:
                    return False

                if selection["type"] == "saved":
                    # Use saved configuration
                    provider = selection["provider"]
                    provider_cfg = selection["config"]

                    self.service = provider
                    self.api_key = provider_cfg.api_key
                    self.current_model = provider_cfg.model

                    cfg = self.services[provider]
                    tools_status = (
                        "with tool support"
                        if cfg.get("supports_tools")
                        else "(chat only, no tools)"
                    )
                    console.print(
                        f"[green]✓[/green] Provider: [bold]{cfg['label']}[/bold] [dim]{tools_status}[/dim]"
                    )
                    console.print(
                        "[green]✓[/green] API key loaded from saved configuration"
                    )
                    console.print(
                        f"[green]✓[/green] Model: [cyan]{self.current_model}[/cyan]\n"
                    )

                    # Update last used timestamp
                    if self.api_key and self.current_model:
                        config_manager.set_provider_config(
                            provider, self.api_key, self.current_model
                        )
                    return True

            except (KeyboardInterrupt, EOFError):
                return False

        # No saved config or user chose new - proceed with normal setup
        provider = self.select_provider()
        if not provider:
            return False

        self.service = provider
        cfg = self.services[provider]
        tools_status = (
            "with tool support"
            if cfg.get("supports_tools")
            else "(chat only, no tools)"
        )
        console.print(
            f"[green]✓[/green] Provider: [bold]{cfg['label']}[/bold] [dim]{tools_status}[/dim]"
        )

        api_key = self.ask_api_key(provider)
        if not api_key:
            return False

        self.api_key = api_key
        console.print("[green]✓[/green] API key configured")

        console.print("[dim]Fetching models...[/dim]")
        models = self.fetch_models(self.service, use_cache=False)

        if not models:
            console.print("[yellow]⚠ Could not fetch models[/yellow]")
            manual = Prompt.ask(
                "[bold]Enter model name manually[/bold]", default=""
            ).strip()
            if manual:
                self.current_model = manual
                console.print(f"[green]✓[/green] Model: [cyan]{manual}[/cyan]\n")
                # Save configuration
                config_manager.set_provider_config(provider, api_key, manual)
                return True
            return False

        self.current_model = models[0]
        console.print(
            f"[green]✓[/green] Model: [cyan]{self.current_model}[/cyan] [dim]({len(models)} available)[/dim]\n"
        )

        # Save configuration for future use
        config_manager.set_provider_config(provider, api_key, self.current_model)
        return True

    def command_provider(self):
        provider = self.select_provider()
        if not provider:
            return

        api_key = self.ask_api_key(provider)
        if not api_key:
            console.print("[yellow]⚠ Cancelled[/yellow]")
            return

        self.service = provider
        self.api_key = api_key
        self._model_cache.clear()
        self.conversation.clear()

        cfg = self.services[provider]
        console.print(f"[green]✓[/green] Provider: [bold]{cfg['label']}[/bold]")

        console.print("[dim]Fetching models...[/dim]")
        models = self.fetch_models(self.service, use_cache=False)

        if not models:
            console.print("[yellow]⚠ Could not fetch models[/yellow]")
            self.current_model = None
            return

        self.current_model = models[0]
        console.print(f"[green]✓[/green] Model: [cyan]{self.current_model}[/cyan]\n")

    def command_model(self):
        if not self.service:
            console.print("[red]❌ No provider selected[/red]")
            return

        console.print("[dim]Loading models...[/dim]")
        models = self.fetch_models(self.service, use_cache=True)

        if not models:
            console.print("[yellow]⚠ No models available[/yellow]")
            manual = Prompt.ask(
                "[bold]Enter model name manually[/bold]", default=""
            ).strip()
            if manual:
                self.current_model = manual
                console.print(f"[green]✓[/green] Model: [cyan]{manual}[/cyan]\n")
            return

        selected = self.select_model(models)
        if selected:
            self.current_model = selected
            console.print(f"[green]✓[/green] Model: [cyan]{selected}[/cyan]\n")

    def run(self):
        if not self.setup():
            return

        if not self.service:
            console.print("[red]❌ No provider configured[/red]")
            return

        cfg = self.services[self.service]

        # All providers now use text-based tools
        self.conversation.set_tool_mode(True)

        console.print(
            Panel(
                "[bold green]🎉 AI Agent Ready![/bold green]\n\n"
                "The AI can:\n"
                "  • Create, read, update, delete files\n"
                "  • List directories\n"
                "  • Execute safe commands\n"
                "  • Remember context from files\n\n"
                "[dim]Try: 'Create a file called workspace/hello.txt with greeting'[/dim]",
                border_style="green",
                box=box.ROUNDED,
            )
        )

        while True:
            try:
                user_input = Prompt.ask("\n[bold cyan]You[/bold cyan]").strip()

                if not user_input:
                    continue

                if user_input.lower() in ("exit", "quit", "q"):
                    console.print("[cyan]👋 Goodbye![/cyan]")
                    break

                if user_input == "/config":
                    self.show_config()
                    continue

                if user_input == "/clear":
                    self.conversation.clear()
                    console.print("[green]✓ Conversation cleared[/green]")
                    continue

                if user_input == "/provider":
                    self.command_provider()
                    continue

                if user_input in ("/model", "/"):
                    self.command_model()
                    continue

                if user_input == "/key":
                    if not self.service:
                        console.print("[red]❌ No provider selected[/red]")
                        continue
                    api_key = self.ask_api_key(self.service)
                    if api_key:
                        self.api_key = api_key
                        console.print("[green]✓ API key updated[/green]")
                    continue

                if not self.current_model:
                    console.print("[red]❌ No model selected. Use /model[/red]")
                    continue

                # All providers use text-based tool parsing
                response = self.send_simple_message(user_input)

                if response:
                    # Render markdown for proper formatting (bold, italic, etc.)
                    if "\n\n[dim]⏱" in response:
                        content, timing = response.rsplit("\n\n", 1)
                        content = content.strip()
                        # Use markdown if content looks like it has formatting
                        if content and (
                            "**" in content
                            or "_" in content
                            or "`" in content
                            or "#" in content
                        ):
                            display_content = Markdown(content)
                        else:
                            display_content = content
                        console.print(
                            Panel(
                                display_content,
                                title="[bold]🤖 AI Agent[/bold]",
                                subtitle=timing,
                                border_style="blue",
                                box=box.ROUNDED,
                            )
                        )
                    else:
                        response = response.strip()
                        # Use markdown if content looks like it has formatting
                        if response and (
                            "**" in response
                            or "_" in response
                            or "`" in response
                            or "#" in response
                        ):
                            display_content = Markdown(response)
                        else:
                            display_content = response
                        console.print(
                            Panel(
                                display_content,
                                title="[bold]🤖 AI Agent[/bold]",
                                border_style="blue",
                                box=box.ROUNDED,
                            )
                        )

            except (KeyboardInterrupt, EOFError):
                console.print("\n[cyan]👋 Goodbye![/cyan]")
                break
            except Exception as e:
                console.print(f"[red]❌ Error: {e}[/red]")


if __name__ == "__main__":
    agent = AIAgent()
    agent.run()
