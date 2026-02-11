import time
import threading
from typing import Dict, List, Optional

import requests
import questionary
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.panel import Panel
from rich.table import Table

console = Console()


class Spinner:
    """Thread-safe spinner animation."""

    def __init__(self, message: str = "Waiting"):
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
            frame = frames[i % len(frames)]
            console.print(
                f"[dim]{frame} {self.message}...[/dim]",
                end="\r",
                highlight=False,
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


class APITester:
    def __init__(self):
        self.api_key: Optional[str] = None
        self.service: Optional[str] = None
        self.current_model: Optional[str] = None
        self._cached_models: Dict[str, List[str]] = {}

        self.services: Dict[str, Dict] = {
            "nvidia": {
                "label": "NVIDIA",
                "base_url": "https://integrate.api.nvidia.com/v1",
                "models_endpoint": "/models",
                "chat_endpoint": "/chat/completions",
                "auth_header": "Authorization",
                "auth_prefix": "Bearer",
                "style": "openai",
                "key_example": "nvapi-...",
            },
            "openrouter": {
                "label": "OpenRouter",
                "base_url": "https://openrouter.ai/api/v1",
                "models_endpoint": "/models",
                "chat_endpoint": "/chat/completions",
                "auth_header": "Authorization",
                "auth_prefix": "Bearer",
                "style": "openai",
                "key_example": "sk-or-...",
            },
            "groq": {
                "label": "Groq",
                "base_url": "https://api.groq.com/openai/v1",
                "models_endpoint": "/models",
                "chat_endpoint": "/chat/completions",
                "auth_header": "Authorization",
                "auth_prefix": "Bearer",
                "style": "openai",
                "key_example": "gsk_...",
            },
            "openai": {
                "label": "OpenAI",
                "base_url": "https://api.openai.com/v1",
                "models_endpoint": "/models",
                "chat_endpoint": "/chat/completions",
                "auth_header": "Authorization",
                "auth_prefix": "Bearer",
                "style": "openai",
                "key_example": "sk-...",
            },
            "anthropic": {
                "label": "Anthropic",
                "base_url": "https://api.anthropic.com/v1",
                "models_endpoint": None,
                "chat_endpoint": "/messages",
                "auth_header": "x-api-key",
                "auth_prefix": None,
                "style": "anthropic",
                "key_example": "sk-ant-...",
                "models": [
                    "claude-3-5-sonnet-20241022",
                    "claude-3-5-haiku-20241022",
                    "claude-3-opus-20240229",
                ],
            },
            "together": {
                "label": "Together AI",
                "base_url": "https://api.together.xyz/v1",
                "models_endpoint": "/models",
                "chat_endpoint": "/chat/completions",
                "auth_header": "Authorization",
                "auth_prefix": "Bearer",
                "style": "openai",
                "key_example": "tg-... or ...",
            },
            "mistral": {
                "label": "Mistral AI",
                "base_url": "https://api.mistral.ai/v1",
                "models_endpoint": "/models",
                "chat_endpoint": "/chat/completions",
                "auth_header": "Authorization",
                "auth_prefix": "Bearer",
                "style": "openai",
                "key_example": "...",
            },
            "cohere": {
                "label": "Cohere",
                "base_url": "https://api.cohere.ai/v1",
                "models_endpoint": None,
                "chat_endpoint": "/chat",
                "auth_header": "Authorization",
                "auth_prefix": "Bearer",
                "style": "cohere",
                "key_example": "...",
                "models": [
                    "command-r-plus",
                    "command-r",
                    "command-light",
                ],
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
        choices = [
            questionary.Choice(title=f"{cfg['label']}", value=key)
            for key, cfg in self.services.items()
        ]
        try:
            return questionary.select(
                "Select provider",
                choices=choices,
                qmark="🌐",
                pointer="➤",
            ).ask()
        except (KeyboardInterrupt, EOFError):
            return None

    def ask_api_key(self, provider: str) -> Optional[str]:
        config = self.services[provider]
        hint = config.get("key_example", "")
        prompt_text = f"[bold]Enter API key for {config['label']}[/bold]"
        if hint:
            prompt_text += f" [dim]({hint})[/dim]"

        try:
            key = Prompt.ask(prompt_text).strip()
            return key if key else None
        except (KeyboardInterrupt, EOFError):
            return None

    def fetch_models(self, service: str, use_cache: bool = True) -> List[str]:
        """Fetch models with caching and retry logic."""
        if use_cache and service in self._cached_models:
            return self._cached_models[service]

        config = self.services[service]

        # Static models
        if config.get("models"):
            models = config["models"]
            self._cached_models[service] = models
            return models

        if not config.get("models_endpoint"):
            return []

        url = config["base_url"] + config["models_endpoint"]
        headers = self.get_headers(service)

        # Retry logic
        for attempt in range(3):
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

                    if models:
                        self._cached_models[service] = models
                        return models
                    else:
                        console.print("[yellow]⚠ No models found in response[/yellow]")
                        return []

                elif resp.status_code == 401:
                    console.print(
                        f"[red]❌ Authentication failed (401). Invalid API key for {config['label']}.[/red]"
                    )
                    return []
                elif resp.status_code == 403:
                    console.print(
                        f"[red]❌ Access forbidden (403). Check API key permissions.[/red]"
                    )
                    return []
                else:
                    console.print(
                        f"[yellow]⚠ HTTP {resp.status_code}: {resp.text[:200]}[/yellow]"
                    )
                    if attempt < 2:
                        time.sleep(1)
                        continue
                    return []

            except requests.exceptions.Timeout:
                if attempt < 2:
                    console.print(
                        f"[yellow]⚠ Timeout, retrying... ({attempt + 1}/3)[/yellow]"
                    )
                    time.sleep(1)
                else:
                    console.print(
                        "[red]❌ Failed to fetch models after 3 attempts (timeout)[/red]"
                    )
                    return []
            except Exception as e:
                console.print(f"[red]❌ Error fetching models: {e}[/red]")
                return []

        return []

    def select_model(self, models: List[str]) -> Optional[str]:
        if not models:
            console.print("[yellow]No models available[/yellow]")
            return None

        try:
            return questionary.select(
                "Select model",
                choices=models,
                qmark="📋",
                pointer="➤",
            ).ask()
        except (KeyboardInterrupt, EOFError):
            return None

    def build_payload(self, prompt: str, service: str, model: str) -> Dict:
        style = self.services[service]["style"]

        if style == "anthropic":
            return {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 2048,
            }
        if style == "cohere":
            return {"model": model, "message": prompt}

        return {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }

    def extract_content(self, data: Dict, service: str) -> str:
        style = self.services[service]["style"]

        try:
            if style == "anthropic":
                return data["content"][0]["text"]
            if style == "cohere":
                return data.get("text") or data.get("message") or str(data)
            return data["choices"][0]["message"]["content"]
        except Exception:
            return str(data)

    def send_message(self, prompt: str, service: str, model: str) -> None:
        config = self.services[service]
        url = config["base_url"] + config["chat_endpoint"]
        headers = self.get_headers(service)
        payload = self.build_payload(prompt, service, model)

        spinner = Spinner("Generating reply")
        spinner.start()
        start = time.perf_counter()

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=120)
            elapsed = time.perf_counter() - start
            spinner.stop()

            latency_ms = int(elapsed * 1000)
            latency_s = f"{elapsed:.3f}"

            if resp.status_code == 200:
                data = resp.json()
                content = self.extract_content(data, service)
                console.print(
                    Panel(
                        content,
                        title="[bold green]✓ Response[/bold green]",
                        subtitle=f"[dim]{latency_s}s ({latency_ms}ms)[/dim]",
                        border_style="green",
                    )
                )
            else:
                try:
                    err_data = resp.json()
                    msg = err_data.get("error", {}).get("message", resp.text)
                except Exception:
                    msg = resp.text[:500]

                console.print(
                    Panel(
                        f"[red]HTTP {resp.status_code}[/red]\n{msg}\n\n"
                        f"[dim]Latency: {latency_s}s[/dim]",
                        title="[bold red]❌ Error[/bold red]",
                        border_style="red",
                    )
                )
                
                # Suggestions
                if resp.status_code == 401:
                    console.print("[yellow]💡 Tip: Check your API key with /key[/yellow]")
                elif resp.status_code == 404:
                    console.print("[yellow]💡 Tip: Try a different model with /model[/yellow]")

        except requests.exceptions.Timeout:
            spinner.stop()
            console.print(
                "[red]❌ Request timed out (120s). Model may be unavailable.[/red]\n"
                "[yellow]💡 Try: /model to select a different model[/yellow]"
            )
        except requests.exceptions.ConnectionError:
            spinner.stop()
            console.print("[red]❌ Connection error. Check your internet.[/red]")
        except Exception as e:
            spinner.stop()
            console.print(f"[red]❌ Error: {e}[/red]")

    def show_config(self):
        """Display current configuration."""
        table = Table(title="Current Configuration", show_header=True)
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")

        if self.service:
            table.add_row("Provider", self.services[self.service]["label"])
        else:
            table.add_row("Provider", "[dim]Not set[/dim]")

        if self.api_key:
            masked = self.api_key[:8] + "..." + self.api_key[-4:] if len(self.api_key) > 12 else "***"
            table.add_row("API Key", masked)
        else:
            table.add_row("API Key", "[dim]Not set[/dim]")

        if self.current_model:
            table.add_row("Model", self.current_model)
        else:
            table.add_row("Model", "[dim]Not set[/dim]")

        console.print(table)

    def setup(self) -> bool:
        console.clear()
        console.print(
            Panel.fit(
                "[bold cyan]🚀 Universal API Tester[/bold cyan]\n"
                "[dim]Commands: /provider, /key, /model, /, /config, exit[/dim]",
                border_style="cyan",
            )
        )

        # Select provider
        provider = self.select_provider()
        if not provider:
            console.print("[red]❌ Setup cancelled[/red]")
            return False

        self.service = provider
        console.print(
            f"[green]✓ Provider:[/green] [bold]{self.services[provider]['label']}[/bold]"
        )

        # Get API key
        api_key = self.ask_api_key(provider)
        if not api_key:
            console.print("[red]❌ API key required[/red]")
            return False

        self.api_key = api_key
        console.print("[green]✓ API key saved[/green]")

        # Fetch models
        console.print("[dim]Fetching models...[/dim]")
        models = self.fetch_models(self.service, use_cache=False)

        if not models:
            console.print(
                "[yellow]⚠ Could not fetch models. You can still try manually entering a model.[/yellow]"
            )
            manual = Prompt.ask(
                "[bold]Enter model name manually (or press Enter to retry setup)[/bold]"
            ).strip()
            if manual:
                self.current_model = manual
                console.print(f"[green]✓ Using model:[/green] [cyan]{manual}[/cyan]\n")
                return True
            return False

        self.current_model = models[0]
        console.print(
            f"[green]✓ Found {len(models)} models[/green]\n"
            f"[dim]Default:[/dim] [cyan]{self.current_model}[/cyan]\n"
        )
        return True

    def command_provider(self):
        """Switch provider and get new API key."""
        provider = self.select_provider()
        if not provider:
            return

        # Ask for new API key for this provider
        api_key = self.ask_api_key(provider)
        if not api_key:
            console.print("[yellow]⚠ Provider change cancelled - no API key provided[/yellow]")
            return

        self.service = provider
        self.api_key = api_key
        self._cached_models.clear()  # Clear cache

        console.print(
            f"[green]✓ Provider:[/green] [bold]{self.services[provider]['label']}[/bold]"
        )

        console.print("[dim]Fetching models...[/dim]")
        models = self.fetch_models(self.service, use_cache=False)

        if not models:
            console.print("[yellow]⚠ Could not fetch models with this key[/yellow]")
            self.current_model = None
            return

        self.current_model = models[0]
        console.print(
            f"[green]✓ Loaded {len(models)} models[/green]\n"
            f"[dim]Default:[/dim] [cyan]{self.current_model}[/cyan]\n"
        )

    def command_key(self):
        """Update API key for current provider."""
        if not self.service:
            console.print("[red]❌ No provider selected[/red]")
            return

        api_key = self.ask_api_key(self.service)
        if not api_key:
            console.print("[yellow]⚠ API key not changed[/yellow]")
            return

        self.api_key = api_key
        self._cached_models.pop(self.service, None)  # Clear cache for this service

        console.print("[green]✓ API key updated[/green]")

        # Re-fetch models
        console.print("[dim]Fetching models...[/dim]")
        models = self.fetch_models(self.service, use_cache=False)
        if models:
            self.current_model = models[0]
            console.print(
                f"[green]✓ Loaded {len(models)} models[/green]\n"
                f"[dim]Default:[/dim] [cyan]{self.current_model}[/cyan]\n"
            )
        else:
            console.print("[yellow]⚠ Could not fetch models with new key[/yellow]")

    def command_model(self):
        if not self.service:
            console.print("[red]❌ No provider selected[/red]")
            return

        models = self.fetch_models(self.service, use_cache=True)
        if not models:
            console.print(
                "[yellow]⚠ No models available. You can enter a model name manually.[/yellow]"
            )
            manual = Prompt.ask("[bold]Model name[/bold]").strip()
            if manual:
                self.current_model = manual
                console.print(f"[green]✓ Model:[/green] [cyan]{manual}[/cyan]\n")
            return

        selected = self.select_model(models)
        if selected:
            self.current_model = selected
            console.print(f"[green]✓ Model:[/green] [cyan]{selected}[/cyan]\n")

    def run(self):
        if not self.setup():
            return

        console.print(
            "[bold green]🎉 Ready![/bold green]\n"
            "[dim]Type your message or use commands: /provider, /key, /model, /, /config, exit[/dim]\n"
        )

        while True:
            try:
                user_input = Prompt.ask("[bold]You[/bold]").strip()

                if not user_input:
                    continue

                if user_input.lower() in ("exit", "quit", "q"):
                    console.print("[cyan]👋 Goodbye![/cyan]")
                    break

                if user_input == "/provider":
                    self.command_provider()
                    continue

                if user_input == "/key":
                    self.command_key()
                    continue

                if user_input in ("/model", "/"):
                    self.command_model()
                    continue

                if user_input == "/config":
                    self.show_config()
                    print()
                    continue

                if not self.current_model:
                    console.print("[red]❌ No model selected. Use /model[/red]")
                    continue

                self.send_message(user_input, self.service, self.current_model)
                print()

            except (KeyboardInterrupt, EOFError):
                console.print("\n[cyan]👋 Goodbye![/cyan]")
                break
            except Exception as e:
                console.print(f"[red]❌ Unexpected error: {e}[/red]")


if __name__ == "__main__":
    tester = APITester()
    tester.run()
