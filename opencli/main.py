#!/usr/bin/env python3

import os
import sys
import json
import subprocess
import difflib
import requests
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.spinner import Spinner
from rich.live import Live
from rich.text import Text
from rich.align import Align
from rich.table import Table
from rich import box
import time
import select
import tty
import termios

console = Console()

# =====================================================
# CONFIG SYSTEM
# =====================================================

CONFIG_PATH = Path.home() / ".opencli"
CONFIG_FILE = CONFIG_PATH / "config.json"
HISTORY_FILE = CONFIG_PATH / "history.json"

# Approx token context windows for models
CONTEXT_WINDOWS = {
    "claude-3-5-sonnet-20240620": 200000,
    "claude-5-sonnet-20260203": 200000,
    "gemini-3-pro": 2000000,
    "gemini-2.0-flash": 1000000,
    "gpt-5.3-codex": 128000,
    "gpt-4o": 128000,
    "moonshotai/kimi-k2.5": 128000,
    "anthropic/claude-3.5-sonnet": 200000,
    "anthropic/claude-3-opus": 200000,
    "z-ai/glm-4.5-air:free": 128000
}

def get_context_limit(model_name):
    """Detects context limit based on model name keywords."""
    model_lower = model_name.lower()
    if model_name in CONTEXT_WINDOWS:
        return CONTEXT_WINDOWS[model_name]
    
    if "gemini" in model_lower:
        if "pro" in model_lower: return 2000000
        return 1000000
    if "claude" in model_lower:
        return 200000
    if "gpt-4" in model_lower or "gpt-5" in model_lower:
        return 128000
    
    return 128000 # Default fallback

def count_tokens(messages):
    """Rough character-based token approximation (1 token ~= 4 chars)."""
    text = ""
    for m in messages:
        text += m.get("content", "")
    return len(text) // 4

def compact_context(messages, model_name):
    """Prunes history if context usage exceeds threshold."""
    limit = get_context_limit(model_name)
    # Default to 75% if not set or invalid
    try:
        threshold_pct = int(config.get("compaction_threshold", 75))
    except:
        threshold_pct = 75
        
    threshold = (limit * threshold_pct) // 100
    
    current_tokens = count_tokens(messages)
    if current_tokens < threshold:
        return messages, current_tokens, limit
    
    # Keep system messages
    system_msgs = [m for m in messages if m["role"] == "system"]
    other_msgs = [m for m in messages if m["role"] != "system"]
    
    # We prune from the beginning of 'other_msgs', keeping pairs (user + assistant)
    # until we are below the threshold. We try to keep at least the last 2 pairs if possible.
    while count_tokens(system_msgs + other_msgs) > threshold and len(other_msgs) > 4:
        # Remove the oldest pair
        other_msgs = other_msgs[2:]
        
    compacted = system_msgs + other_msgs
    tokens_after = count_tokens(compacted)
    return compacted, tokens_after, limit

def load_history():
    if not HISTORY_FILE.exists():
        return {"sessions": []}
    try:
        with open(HISTORY_FILE, "r") as f:
            data = json.load(f)
            # Filter out sessions older than 24 hours
            now = time.time()
            data["sessions"] = [s for s in data["sessions"] if now - s.get("timestamp", 0) < 86400]
            return data
    except:
        return {"sessions": []}

def save_history(messages, session_id=None):
    CONFIG_PATH.mkdir(exist_ok=True)
    history = load_history()
    cwd = os.getcwd()
    
    # Generate title from first user message
    title = "New Chat"
    for m in messages:
        if m["role"] == "user":
            content = m["content"].strip().splitlines()[0]
            title = (content[:40] + "...") if len(content) > 40 else content
            break

    session_data = {
        "id": session_id or str(time.time()),
        "timestamp": time.time(),
        "title": title,
        "cwd": cwd,
        "messages": messages
    }

    # Update existing or add new
    found = False
    for i, s in enumerate(history["sessions"]):
        if s["id"] == session_data["id"]:
            history["sessions"][i] = session_data
            found = True
            break
    
    if not found:
        history["sessions"].insert(0, session_data)
    
    # Keep only last 10 (increased for project awareness)
    history["sessions"] = history["sessions"][:10]
    
    try:
        with open(HISTORY_FILE, "w") as f:
            json.dump(history, f)
    except:
        pass
    return session_data["id"]

def get_char(fd):
    """Reliably read a single character or escape sequence."""
    r, _, _ = select.select([fd], [], [], 0.01) # Check if data is already waiting
    ch = sys.stdin.read(1)
    if ch == '\x1b':
        # Arrow keys are ESC + [ + A/B/C/D. 
        # We wait up to 0.15s for the rest of the sequence for high reliability.
        r, _, _ = select.select([fd], [], [], 0.15)
        if r:
            ch2 = sys.stdin.read(1)
            if ch2 == '[':
                r, _, _ = select.select([fd], [], [], 0.1)
                if r:
                    ch3 = sys.stdin.read(1)
                    return f"ESC[{ch3}"
        return "ESC"
    return ch

def select_session_tui():
    history = load_history()
    all_sessions = history.get("sessions", [])
    cwd = os.getcwd()
    
    local = [s for s in all_sessions if s.get("cwd") == cwd]
    other = [s for s in all_sessions if s.get("cwd") != cwd]
    sessions = local + other

    if not sessions: return None
    
    idx = 0
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    
    try:
        # Enter persistent TUI mode
        tty.setraw(fd)
        console.show_cursor = False
        with Live(refresh_per_second=20, screen=False, auto_refresh=True) as live:
            while True:
                # Render
                table = Table(title="[bold cyan]Resume Recent Chat[/bold cyan]", box=box.ROUNDED, show_header=False)
                for i, s in enumerate(sessions):
                    is_local = s.get("cwd") == cwd
                    style = "bold green" if i == idx else ("cyan" if is_local else "white")
                    prefix = " > " if i == idx else "   "
                    local_tag = "[dim](here)[/dim] " if is_local else ""
                    dt = time.strftime("%H:%M", time.localtime(s["timestamp"]))
                    table.add_row(f"[{style}]{prefix}{local_tag}{s['title']} [dim]({dt})[/dim][/{style}]")
                
                live.update(Align.center(Panel(table, subtitle="[dim]↑/↓: Navigate • Enter: Resume • Esc/Q: Cancel[/dim]", border_style="dim")))
                
                # Input
                key = get_char(fd)
                
                if key == "ESC" or key.lower() == "q":
                    return None
                elif key == "ESC[A": # UP
                    idx = (idx - 1) % len(sessions)
                elif key == "ESC[B": # DOWN
                    idx = (idx + 1) % len(sessions)
                elif key in ["\r", "\n"]:
                    return sessions[idx]
    finally:
        console.show_cursor = True
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        # Final buffer drain
        while select.select([fd], [], [], 0.01)[0]:
            sys.stdin.read(1)

def load_config():
    defaults = {
        "nvidia_key": "",
        "openrouter_key": "",
        "anthropic_key": "",
        "openai_key": "",
        "google_key": "",
        "ollama_url": "http://localhost:11434/v1/chat/completions",
        "mode": "safe",
        "provider": "OpenRouter",
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "theme": "dark",
        "model": "z-ai/glm-4.5-air:free",
        "max_tokens": 4096,
        "compaction_threshold": 75,
        "provider_models": {
            "OpenRouter": "z-ai/glm-4.5-air:free",
            "Anthropic": "claude-3-5-sonnet-20240620",
            "Google": "gemini-3-pro",
            "OpenAI": "gpt-5.3-codex",
            "NVIDIA": "moonshotai/kimi-k2.5",
            "Ollama": "llama3"
        }
    }
    if not CONFIG_FILE.exists():
        return defaults
    try:
        with open(CONFIG_FILE, "r") as f:
            data = json.load(f)
            # Merge defaults
            for k, v in defaults.items():
                if k not in data:
                    data[k] = v
                elif k == "provider_models" and isinstance(v, dict):
                    # Ensure all default providers are present
                    for pk, pv in v.items():
                        if pk not in data[k]:
                            data[k][pk] = pv
            return data
    except:
        return defaults

def save_config(config):
    CONFIG_PATH.mkdir(exist_ok=True)
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)

config = load_config()

# =====================================================
# BANNER
# =====================================================

def get_theme_color(color_name):
    theme = config.get("theme", "dark")
    colors = {
        "dark": {
            "primary": "cyan",
            "secondary": "green",
            "warning": "yellow",
            "error": "red",
            "text": "white",
            "dim": "dim"
        },
        "light": {
            "primary": "blue",
            "secondary": "dark_green",
            "warning": "dark_orange",
            "error": "red",
            "text": "black",
            "dim": "dim"
        }
    }
    return colors.get(theme, colors["dark"]).get(color_name, "white")

def banner(mode, model):
    console.clear()
    logo = r"""
   _  ____  _____ _   _  ____ _     ___ 
  / \|  _ \|  ___| \ | |/ ___| |   |_ _|
 | | | |_) |  _| |  \| | |   | |    | | 
 | | |  __/| |___| |\  | |___| |___ | | 
  \_/|_|   |_____|_| \_|\____|_____|___|
"""
    
    primary = get_theme_color("primary")
    secondary = get_theme_color("secondary")
    text_color = get_theme_color("text")
    
    logo_text = Text(logo, style=primary)
    cwd = os.getcwd().replace(os.path.expanduser("~"), "~")
    
    splash_content = Text.assemble(
        logo_text,
        Text.from_markup(f"\n[dim]- by curren -[/dim]\n"),
        Text.from_markup(f"\n[bold {text_color}]Folder:[/bold {text_color}] [blue]{cwd}[/blue]"),
        Text.from_markup(f"\n[bold {text_color}]Model: [/bold {text_color}] [{secondary}]{model}[/{secondary}]"),
        Text.from_markup(f"\n[bold {text_color}]Mode:  [/bold {text_color}] [bold {'yellow' if mode == 'safe' else 'red'}]{mode.upper()}[/bold {'yellow' if mode == 'safe' else 'red'}]")
    )
    
    console.print(Align.center(Panel(splash_content, box=box.ROUNDED, border_style="dim", padding=(1, 4))))

# =====================================================
# SETTINGS
# =====================================================

def interactive_settings_menu():
    options = [
        {"name": "Provider", "key": "provider", "type": "toggle", "values": ["OpenRouter", "Anthropic", "Google", "OpenAI", "NVIDIA", "Ollama"]},
        {"name": "Model", "key": "model", "type": "input"},
        {"name": "Ollama URL", "key": "ollama_url", "type": "input"},
        {"name": "Theme", "key": "theme", "type": "toggle", "values": ["dark", "light"]},
        {"name": "Execution Mode", "key": "mode", "type": "toggle", "values": ["safe", "unsafe"]},
        {"name": "Max Tokens", "key": "max_tokens", "type": "input"},
        {"name": "Compaction Threshold (%)", "key": "compaction_threshold", "type": "input"},
        {"name": "Anthropic API Key", "key": "anthropic_key", "type": "input"},
        {"name": "Google API Key", "key": "google_key", "type": "input"},
        {"name": "OpenAI API Key", "key": "openai_key", "type": "input"},
        {"name": "NVIDIA API Key", "key": "nvidia_key", "type": "input"},
        {"name": "OpenRouter API Key", "key": "openrouter_key", "type": "input"},
    ]
    
    selected_idx = 0
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    
    try:
        tty.setraw(fd)
        console.show_cursor = False
        with Live(refresh_per_second=20, screen=False, auto_refresh=True) as live:
            while True:
                table = Table(title="[bold cyan]OpenCLI Settings[/bold cyan]", box=box.ROUNDED)
                table.add_column("Setting", style="cyan", width=25)
                table.add_column("Value", style="dim", width=30, no_wrap=True)
                
                for i, opt in enumerate(options):
                    prefix = "› " if i == selected_idx else "  "
                    style = "bold yellow" if i == selected_idx else "white"
                    val = config.get(opt["key"], "")
                    if opt["type"] == "toggle":
                        val_str = str(val).upper()
                        val_display = f"[bold green]{val_str}[/bold green]" if val in ["safe", "NVIDIA", "Anthropic", "dark", "OpenRouter", "OpenAI", "Google", "Ollama"] else f"[bold red]{val_str}[/bold red]"
                    else:
                        v = str(val)
                        masked = v[:4] + "..." + v[-4:] if len(v) > 12 else v
                        val_display = f"[green]SET[/green] [dim]{masked}[/dim]" if val else "[red]EMPTY[/red]"
                    table.add_row(f"{prefix}{opt['name']}", val_display, style=style)

                live.update(Align.center(Panel(table, subtitle="[dim]↑/↓: Navigate • Enter: Change • Esc/Q: Exit[/dim]", border_style="dim")))
                
                key = get_char(fd)
                if key == "ESC" or key.lower() == "q": break
                elif key == "ESC[A": selected_idx = (selected_idx - 1) % len(options)
                elif key == "ESC[B": selected_idx = (selected_idx + 1) % len(options)
                elif key in ["\r", "\n"]:
                    opt = options[selected_idx]
                    termios.tcsetattr(fd, termios.TCSADRAIN, old)
                    live.stop()
                    
                    if opt["type"] == "toggle":
                        vals = opt["values"]
                        cur = config.get(opt["key"])
                        try:
                            new_idx = (vals.index(cur) + 1) % len(vals)
                        except (ValueError, KeyError):
                            new_idx = 0
                        new_val = vals[new_idx]
                        config[opt["key"]] = new_val
                        
                        if opt["key"] == "provider":
                            pm = config.get("provider_models", {})
                            last_model = pm.get(new_val)
                            if new_val == "OpenRouter":
                                config["url"], config["model"] = "https://openrouter.ai/api/v1/chat/completions", last_model or "anthropic/claude-3.5-sonnet"
                            elif new_val == "Anthropic":
                                config["url"], config["model"] = "https://api.anthropic.com/v1/messages", last_model or "claude-3-5-sonnet-20240620"
                            elif new_val == "Google":
                                config["url"], config["model"] = "https://generativelanguage.googleapis.com/v1beta/models/", last_model or "gemini-3-pro"
                            elif new_val == "OpenAI":
                                config["url"], config["model"] = "https://api.openai.com/v1/chat/completions", last_model or "gpt-5.3-codex"
                            elif new_val == "NVIDIA":
                                config["url"], config["model"] = "https://integrate.api.nvidia.com/v1/chat/completions", last_model or "moonshotai/kimi-k2.5"
                            elif new_val == "Ollama":
                                config["url"], config["model"] = config.get("ollama_url", "http://localhost:11434/v1/chat/completions"), last_model or "llama3"
                            pm[new_val] = config["model"]
                            config["provider_models"] = pm
                        save_config(config)
                    else:
                        console.clear()
                        new = input(f"Edit {opt['name']} (Empty to cancel): ").strip()
                        if new:
                            val = "" if new.lower() == "clear" else new
                            config[opt["key"]] = val
                            if opt["key"] == "model":
                                pm = config.get("provider_models", {})
                                pm[config.get("provider", "OpenRouter")] = val
                                config["provider_models"] = pm
                            save_config(config)
                    
                    live.start()
                    tty.setraw(fd)
    finally:
        console.show_cursor = True
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        while select.select([fd], [], [], 0.01)[0]:
            sys.stdin.read(1)
    
    console.clear()

# =====================================================
# TOOLS
# =====================================================

def list_files(path="."):
    import os
    ignore_dirs = {"node_modules", ".git", "__pycache__", ".next", ".venv", "venv", "dist", "build"}
    try:
        files = []
        for root, dirs, filenames in os.walk(path):
            dirs[:] = [d for d in dirs if d not in ignore_dirs]
            for f in filenames:
                file_path = os.path.join(root, f)
                files.append(file_path)
                if len(files) >= 100:
                    files.append("... (truncated: too many files)")
                    return "\n".join(files)
        return "\n".join(files) if files else "No files found."
    except Exception as e:
        return f"Error: {str(e)}"

def read_file(path):
    if not os.path.exists(path): return "File not found."
    with open(path, "r") as f: return f.read()

def write_file(path, content):
    with open(path, "w") as f: f.write(content)
    return f"Wrote {path}"

def run_shell(command):
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=300)
        output = f"{result.stdout}\n{result.stderr}".strip()
        return output if output else "(Command executed with no output)"
    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 300 seconds."
    except Exception as e:
        return f"Error: {str(e)}"

def replace_text(path, old_text, new_text):
    if not os.path.exists(path): return f"Error: File '{path}' not found."
    try:
        with open(path, "r") as f: content = f.read()
        if old_text not in content: return f"Error: Could not find exact match."
        new_content = content.replace(old_text, new_text, 1)
        with open(path, "w") as f: f.write(new_content)
        return f"Successfully replaced text in {path}."
    except Exception as e:
        return f"Error: {str(e)}"

def search_code(query, path="."):
    try:
        result = subprocess.run(["grep", "-r", "-n", "--exclude-dir={.git,node_modules,__pycache__}", query, path], capture_output=True, text=True, timeout=30)
        output = result.stdout.strip()
        if not output: return f"No matches found for '{query}'."
        lines = output.splitlines()
        if len(lines) > 50: return "\n".join(lines[:50]) + f"\n\n... (truncated)"
        return output
    except Exception as e:
        return f"Error: {str(e)}"

TOOLS = {
    "list_files": list_files,
    "read_file": read_file,
    "write_file": write_file,
    "replace_text": replace_text,
    "search_code": search_code,
    "run_shell": run_shell
}

import re

# =====================================================
# SYSTEM PROMPT
# =====================================================

def get_system_prompt():
    prompt_path = Path(__file__).parent / "OPENCLI.md"
    if prompt_path.exists():
        with open(prompt_path, "r") as f: return f.read()
    return "You are OpenCLI, an autonomous coding agent."

SYSTEM_PROMPT = get_system_prompt()

def extract_json(text):
    try:
        match = re.search(r'(\{.*\})', text, re.DOTALL)
        if match:
            json_str = match.group(1)
            obj = json.loads(json_str)
            if "tool" in obj:
                clean_text = text.replace(json_str, "").strip()
                clean_text = re.sub(r'```json\s*```', '', clean_text).strip()
                clean_text = re.sub(r'```\s*$', '', clean_text).strip()
                return clean_text, obj
    except: pass
    text = re.sub(r'```json\s*```', '', text).strip()
    return text, None

def check_stop():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        rlist, _, _ = select.select([sys.stdin], [], [], 0)
        if rlist:
            key = sys.stdin.read(1)
            if key == '\x1b' or key == 'q': return True
    except: pass
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return False

# =====================================================
# AGENT LOOP
# =====================================================

def check_ollama(url):
    try:
        base_url = url.split("/v1")[0]
        response = requests.get(base_url, timeout=2)
        return response.status_code == 200
    except: return False

def call_model(provider_name, provider_model, key, messages):
    headers = {"Content-Type": "application/json"}
    if provider_name == "OpenRouter": url = "https://openrouter.ai/api/v1/chat/completions"
    elif provider_name == "Anthropic": url = "https://api.anthropic.com/v1/messages"
    elif provider_name == "Google": url = "https://generativelanguage.googleapis.com/v1beta/models/"
    elif provider_name == "OpenAI": url = "https://api.openai.com/v1/chat/completions"
    elif provider_name == "NVIDIA": url = "https://integrate.api.nvidia.com/v1/chat/completions"
    elif provider_name == "Ollama":
        url = config.get("ollama_url", "http://localhost:11434/v1/chat/completions")
        if not check_ollama(url): return "[red]Error: Ollama server not found.[/red]"
    else: url = config.get("url")
    
    max_tokens = int(config.get("max_tokens", 4096))
    if provider_name == "OpenRouter":
        headers["Authorization"] = f"Bearer {key}"
        headers["HTTP-Referer"] = "https://github.com/curren/OpenCLI"
        headers["X-Title"] = "OpenCLI"
        payload = {"model": provider_model, "messages": messages, "stream": True, "max_tokens": max_tokens}
    elif provider_name == "NVIDIA":
        headers["Authorization"] = f"Bearer {key}"
        payload = {"model": provider_model, "messages": messages, "temperature": 1.0, "stream": True, "chat_template_kwargs": {"thinking": True}, "max_tokens": max_tokens}
    elif provider_name == "Anthropic":
        headers["x-api-key"] = key
        headers["anthropic-version"] = "2023-06-01"
        system_msg = ""
        anth_msgs = []
        for m in messages:
            if m["role"] == "system": system_msg += m["content"] + "\n"
            else: anth_msgs.append(m)
        payload = {"model": provider_model, "messages": anth_msgs, "system": system_msg.strip(), "max_tokens": max_tokens, "stream": True}
    elif provider_name == "Google":
        url = f"{url}{provider_model}:streamGenerateContent?key={key}"
        contents = []
        for m in messages:
            role = "user" if m["role"] == "user" else "model"
            contents.append({"role": role, "parts": [{"text": m["content"]}]})
        payload = {"contents": contents, "generationConfig": {"maxOutputTokens": max_tokens, "temperature": 1.0}}
    elif provider_name == "OpenAI":
        headers["Authorization"] = f"Bearer {key}"
        payload = {"model": provider_model, "messages": messages, "stream": True, "max_tokens": max_tokens}
    elif provider_name == "Ollama":
        payload = {"model": provider_model, "messages": messages, "stream": True, "max_tokens": max_tokens}

    full_text = ""
    thinking_text = ""
    try:
        with Live(Spinner("dots", text="Thinking..."), refresh_per_second=12):
            response = requests.post(url, headers=headers, json=payload, stream=True, timeout=60)
            if response.status_code != 200:
                try:
                    err = response.json().get("error", {}).get("message", "Unknown error")
                    return f"[red]API Error ({response.status_code}): {err}[/red]"
                except: return f"[red]API Error ({response.status_code}): {response.text[:100]}[/red]"

            for line in response.iter_lines():
                if check_stop(): break
                if line:
                    chunk = line.decode("utf-8").strip()
                    if provider_name == "Anthropic":
                        if chunk.startswith("data: "):
                            try:
                                data = json.loads(chunk[6:])
                                if data["type"] == "content_block_delta":
                                    full_text += data["delta"].get("text", "")
                            except: pass
                    elif provider_name == "Google":
                        try:
                            data = json.loads(chunk)
                            if "candidates" in data:
                                full_text += data["candidates"][0].get("content", {}).get("parts", [{}])[0].get("text", "")
                        except: pass
                    else:
                        if chunk.startswith("data: "): chunk = chunk[6:]
                        if chunk == "[DONE]": break
                        try:
                            data = json.loads(chunk)
                            if "choices" in data:
                                delta = data["choices"][0].get("delta", {})
                                full_text += delta.get("content", "")
                                thinking_text += delta.get("reasoning_content", "")
                        except: pass

        if thinking_text or full_text:
            if thinking_text and full_text: return f"[dim]💭 Thinking: {thinking_text}[/dim]\n\n{full_text}"
            return thinking_text or full_text
        return "[dim](Empty response)[/dim]"
    except Exception as e: return f"[red]Error: {str(e)}[/red]"

def run_onboarding():
    console.clear()
    welcome_text = Text.assemble(
        (r"""
   _  ____  _____ _   _  ____ _     ___ 
  / \|  _ \|  ___| \ | |/ ___| |   |_ _|
  | | | |_) |  _| |  \| | |   | |    | | 
  | | |  __/| |___| |\  | |___| |___ | | 
  \_/|_|   |_____|_| \_|\____|_____|___|
""", "cyan"),
        ("\n\nWelcome to OpenCLI!\n", "bold white")
    )
    console.print(Align.center(Panel(welcome_text, box=box.ROUNDED, border_style="cyan")))
    theme_choice = Prompt.ask("\n[bold white]1. Theme?[/bold white]", choices=["dark", "light"], default="dark")
    config["theme"] = theme_choice
    provider = Prompt.ask("\n2. Provider", choices=["OpenRouter", "Anthropic", "Google", "OpenAI", "NVIDIA", "Ollama"], default="OpenRouter")
    config["provider"] = provider
    if provider == "OpenRouter":
        config["url"], config["model"] = "https://openrouter.ai/api/v1/chat/completions", "anthropic/claude-3.5-sonnet"
        key_k = "openrouter_key"
    elif provider == "Anthropic":
        config["url"], config["model"] = "https://api.anthropic.com/v1/messages", "claude-5-sonnet-20260203"
        key_k = "anthropic_key"
    elif provider == "Google":
        config["url"], config["model"] = "https://generativelanguage.googleapis.com/v1beta/models/", "gemini-3-pro"
        key_k = "google_key"
    elif provider == "OpenAI":
        config["url"], config["model"] = "https://api.openai.com/v1/chat/completions", "gpt-5.3-codex"
        key_k = "openai_key"
    elif provider == "NVIDIA":
        config["url"], config["model"] = "https://integrate.api.nvidia.com/v1/chat/completions", "moonshotai/kimi-k2.5"
        key_k = "nvidia_key"
    else:
        config["url"], config["model"] = "http://localhost:11434/v1/chat/completions", "llama3"
        key_k = None
    if key_k: config[key_k] = Prompt.ask(f"3. {provider} Key", password=True)
    save_config(config)

def main():
    global config
    if "--version" in sys.argv:
        print("OpenCLI v0.1.0")
        return
    if "--settings" in sys.argv:
        interactive_settings_menu()
        config = load_config()
        banner(config.get("mode", "safe"), config.get("model"))
    if not CONFIG_FILE.exists(): run_onboarding()
    
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    session_id = None
    if "--resume" in sys.argv:
        sel = select_session_tui()
        if sel:
            messages, session_id = sel["messages"], sel["id"]
            console.print(f"[dim]📜 Resumed: {sel['title']}[/dim]")

    cwd = os.getcwd()
    files_list = list_files(".")
    messages.append({"role": "system", "content": f"PROJECT: {cwd}\nFILES:\n{files_list}"})
    
    mode = config.get("mode", "safe")
    banner(mode, config.get("model"))

    while True:
        try:
            config = load_config()
            p_name = config.get("provider", "OpenRouter")
            p_model = config.get("model")
            key_m = {"NVIDIA":"nvidia_key","OpenRouter":"openrouter_key","Anthropic":"anthropic_key","OpenAI":"openai_key","Google":"google_key"}
            key = config.get(key_m.get(p_name))
            if not key and p_name != "Ollama":
                if Prompt.ask("No API key. Setting?", default="y") == "y": interactive_settings_menu(); continue
                return

            text_c = get_theme_color("text")
            console.print(f"\n[bold {text_c}]You[/bold {text_c}] [dim](type / for settings)[/dim]")
            user_input = console.input("› ")

            if user_input.strip() == "/":
                interactive_settings_menu()
                config = load_config()
                mode = config.get("mode", "safe")
                banner(mode, config.get("model"))
                continue
            if user_input.strip() == "/clear":
                messages = [{"role": "system", "content": SYSTEM_PROMPT}]
                session_id = None
                console.clear()
                config = load_config()
                banner(config.get("mode", "safe"), config.get("model"))
                continue
            if user_input.lower() in ["exit", "quit"]: break

            messages.append({"role": "user", "content": user_input})
            while True:
                messages, tokens, limit = compact_context(messages, p_model)
                console.print(Align.right(Text(f"Context: {(tokens*100)//limit}%", style="dim")))
                reply = call_model(p_name, p_model, key, messages)
                messages.append({"role": "assistant", "content": reply})
                reasoning, tool_call = extract_json(reply)
                if reasoning.strip(): console.print(Panel(reasoning, style="cyan", title="OpenCLI 💭"))
                if tool_call:
                    t_name, args = tool_call["tool"], tool_call.get("args", {})
                    if t_name not in TOOLS:
                        messages.append({"role": "user", "content": f"Error: Tool {t_name} not found."})
                        continue
                    if mode == "safe" and Prompt.ask(f"Approve {t_name}?", default="n") != "y":
                        messages.append({"role": "user", "content": "Denied."}); continue
                    try:
                        res = TOOLS[t_name](**args)
                        console.print(Panel(str(res)[:500] + ("..." if len(str(res))>500 else ""), title=f"Tool: {t_name}"))
                        messages.append({"role": "user", "content": f"RESULT: {res}"})
                    except TypeError as e:
                        err_msg = f"Error: Tool '{t_name}' call failed due to incorrect arguments: {str(e)}. Please ensure all required arguments (like 'content' for 'write_file') are provided."
                        console.print(Panel(err_msg, title="Tool Error", style="bold red"))
                        messages.append({"role": "user", "content": err_msg})
                    except Exception as e:
                        err_msg = f"Error executing tool '{t_name}': {str(e)}"
                        console.print(Panel(err_msg, title="Tool Error", style="bold red"))
                        messages.append({"role": "user", "content": err_msg})
                    session_id = save_history(messages, session_id)
                    continue
                session_id = save_history(messages, session_id)
                break
        except KeyboardInterrupt:
            if Prompt.ask("Quit?", choices=["y","n"], default="n") == "y": break
            banner(mode, config.get("model"))

if __name__ == "__main__":
    main()