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
from rich.markdown import Markdown
from rich import box
import time
import select
import tty
import termios
import base64
import subprocess as sp
from PIL import Image
from io import BytesIO

import signal
import atexit

# Global session tracking for graceful shutdown
CURRENT_MESSAGES = None
CURRENT_SESSION_ID = None

def persist_session_on_exit():
    global CURRENT_MESSAGES, CURRENT_SESSION_ID
    if CURRENT_MESSAGES is not None:
        try:
            save_history(CURRENT_MESSAGES, CURRENT_SESSION_ID)
        except:
            pass

def handle_termination(signum, frame):
    persist_session_on_exit()
    sys.exit(0)

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
    """Read full escape sequences safely (fix arrow keys exiting)."""
    ch = sys.stdin.read(1)

    if ch == '\x1b':  # ESC
        seq = ch
        # Try to read the rest of the escape sequence
        for _ in range(2):
            r, _, _ = select.select([fd], [], [], 0.05)
            if r:
                seq += sys.stdin.read(1)
            else:
                break

        if seq.startswith('\x1b[') and len(seq) == 3:
            return f"ESC[{seq[2]}"
        return "ESC"

    return ch


def select_session_menu():
    console.clear()
    history = load_history()
    sessions = history.get("sessions", [])
    cwd = os.getcwd()

    if not sessions:
        console.print(Panel("No recent sessions found.", style="yellow"))
        console.input("Press Enter to return...")
        return None

    console.print(Panel("[bold cyan]Resume Recent Chat[/bold cyan]", border_style="cyan"))

    while True:
        console.print()
        for i, s in enumerate(sessions, 1):
            is_local = s.get("cwd") == cwd
            tag = "[dim](here)[/dim] " if is_local else ""
            dt = time.strftime("%H:%M", time.localtime(s["timestamp"]))
            console.print(f"[cyan]{i}.[/cyan] {tag}{s['title']} [dim]({dt})[/dim]")

        console.print("\n[dim]Enter number to resume • q to cancel[/dim]")
        choice = console.input("› ").strip()

        if choice.lower() == "q":
            return None
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(sessions):
                return sessions[idx]

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

    cwd = os.getcwd().replace(os.path.expanduser("~"), "~")

    logo = r"""
   _  ____  _____ _   _  ____ _     ___ 
  / \|  _ \|  ___| \ | |/ ___| |   |_ _|
 | | | |_) |  _| |  \| | |   | |    | | 
 | | |  __/| |___| |\  | |___| |___ | | 
  \_/|_|   |_____|_| \_|\____|_____|___|
"""

    ghost = """
⠀⠀⠀⠀⠀⠀⠀⢀⠤⠐⠀⠀⠒⠂⠄⡀⠀⠀
⠀⠀⠀⠀⠀⠀⡠⠁⠀⠀⠀⠀⠀⠀⠀⢸⣦⠀
⠀⠀⠀⠀⠀⠰⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠻⢃
⠀⠀⠀⠀⢠⠃⠀⠀⠀⠀⢠⣄⠀⠀⠀⢀⣀⠘
⠀⠀⠀⠀⠌⠀⠀⠀⠀⢀⠘⢍⠀⣄⠤⡨⠯⠀
⠀⠀⠀⡐⠀⠀⠀⠀⠀⠈⠈⠃⠀⠀⠀⠑⠒⠸
⠀⠀⢠⣱⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⠇
⠀⡠⠻⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡌⠀
⠔⠁⣼⣐⡄⠀⠀⠀⢠⠀⠀⠀⠀⡄⠀⡜⠀⠀
⠑⠚⠉⠀⡄⠀⢀⠴⡏⠀⠀⢠⢞⣧⠌⠀⠀⠀
⠀⠀⠀⠀⠑⠚⠓⠉⠗⠧⠒⠉⠉⠀⠀⠀⠀⠀
"""
    ghost = "\n".join(line[3:] if len(line) >= 4 else line for line in ghost.splitlines())

    left_block = Text.assemble(
        (logo, "green"),
        ("\n- by curren -\n", "dim"),
        (f"\nFolder: {cwd}", "green"),
        (f"\nModel:  {model}", "green"),
        (f"\nMode:   {mode.upper()}\n", "bold yellow" if mode == "safe" else "bold red"),
        ("\nTip: SAFE mode asks before tools • UNSAFE auto-runs.\n", "dim")
    )

    from rich.columns import Columns

    combined_layout = Columns(
        [
            left_block,
            Text(ghost, style="green")
        ],
        expand=True,
        equal=False
    )

    console.print(
        Panel(
            combined_layout,
            box=box.ROUNDED,
            border_style="green",
            padding=(1, 2)
        )
    )

# =====================================================
# SETTINGS
# =====================================================

def interactive_settings_menu():
    console.clear()
    console.print(Panel("[bold cyan]OpenCLI Settings[/bold cyan]", border_style="cyan"))

    options = [
        ("Provider", "provider"),
        ("Model", "model"),
        ("Ollama URL", "ollama_url"),
        ("Theme", "theme"),
        ("Execution Mode", "mode"),
        ("Max Tokens", "max_tokens"),
        ("Compaction Threshold (%)", "compaction_threshold"),
        ("Anthropic API Key", "anthropic_key"),
        ("Google API Key", "google_key"),
        ("OpenAI API Key", "openai_key"),
        ("NVIDIA API Key", "nvidia_key"),
        ("OpenRouter API Key", "openrouter_key"),
    ]

    while True:
        console.print()
        for i, (name, key) in enumerate(options, 1):
            value = str(config.get(key, ""))
            if "key" in key and value:
                value = value[:4] + "..." + value[-4:]
            console.print(f"[cyan]{i}.[/cyan] {name}: [green]{value if value else 'EMPTY'}[/green]")

        console.print("\n[dim]Enter number to edit • q to exit[/dim]")
        choice = console.input("› ").strip()

        if choice.lower() == "q":
            break

        if not choice.isdigit():
            continue

        idx = int(choice) - 1
        if idx < 0 or idx >= len(options):
            continue

        name, key = options[idx]

        # Provider selection (dropdown style)
        if key == "provider":
            console.print("\nSelect Provider:")
            providers = ["OpenRouter", "Anthropic", "Google", "OpenAI", "NVIDIA", "Ollama"]
            for i, p in enumerate(providers, 1):
                console.print(f"[cyan]{i}.[/cyan] {p}")
            choice = console.input("› ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(providers):
                new_provider = providers[int(choice) - 1]
                config["provider"] = new_provider
                default_models = config.get("provider_models", {})
                config["model"] = default_models.get(new_provider, config.get("model"))
                save_config(config)
                console.print("[green]Provider updated.[/green]")
            continue

        # Theme editing (dropdown style)
        if key == "theme":
            console.print("\nTheme Options:")
            console.print("[cyan]1.[/cyan] dark")
            console.print("[cyan]2.[/cyan] light")
            t_choice = console.input("› ").strip()
            if t_choice == "1":
                config["theme"] = "dark"
            elif t_choice == "2":
                config["theme"] = "light"
            save_config(config)
            console.print("[green]Theme updated.[/green]")
            continue

        # Execution Mode editing (clarity)
        if key == "mode":
            console.print("\nExecution Mode:")
            console.print("[cyan]1.[/cyan] SAFE  (approval required before tools)")
            console.print("[cyan]2.[/cyan] UNSAFE (auto-executes tools — dangerous)")
            m_choice = console.input("› ").strip()
            if m_choice == "1":
                config["mode"] = "safe"
            elif m_choice == "2":
                config["mode"] = "unsafe"
            save_config(config)
            console.print("[green]Mode updated.[/green]")
            continue

        # Model selection (auto-fetch if supported)
        if key == "model":
            provider = config.get("provider")

            def fetch_models():
                try:
                    if provider == "OpenRouter":
                        headers = {"Authorization": f"Bearer {config.get('openrouter_key')}"}
                        r = requests.get("https://openrouter.ai/api/v1/models", headers=headers, timeout=10)
                        data = r.json().get("data", [])
                        return [m["id"] for m in data if "free" in m["id"]][:30]

                    if provider == "Google":
                        key = config.get("google_key")
                        url = f"https://generativelanguage.googleapis.com/v1beta/models?key={key}"
                        r = requests.get(url, timeout=10)
                        models = r.json().get("models", [])
                        return [
                            m["name"].replace("models/", "")
                            for m in models
                            if "generateContent" in m.get("supportedGenerationMethods", [])
                        ][:20]

                    if provider == "OpenAI":
                        headers = {"Authorization": f"Bearer {config.get('openai_key')}"}
                        r = requests.get("https://api.openai.com/v1/models", headers=headers, timeout=10)
                        models = r.json().get("data", [])
                        return [m["id"] for m in models if "gpt" in m["id"]][:20]

                    if provider == "NVIDIA":
                        return ["moonshotai/kimi-k2.5"]

                    if provider == "Anthropic":
                        # Curated up-to-date Anthropic model list (Haiku → Sonnet → Opus)
                        return [
                            # Haiku (fast / cost-efficient)
                            "claude-haiku-4-5-20251001",

                            # Sonnet (balanced reasoning / coding)
                            "claude-sonnet-4-5-20250929",

                            # Opus (highest capability)
                            "claude-opus-4-6"
                        ]

                    if provider == "Ollama":
                        return ["llama3", "mistral", "codellama"]

                except Exception:
                    return []

            console.print("\nFetching available models...\n")
            models = fetch_models()

            if models:
                for i, m in enumerate(models, 1):
                    console.print(f"[cyan]{i}.[/cyan] {m}")
                console.print("[cyan]M.[/cyan] Manual entry")
                choice = console.input("› ").strip()

                if choice.lower() == "m":
                    manual = console.input("Enter model name: ").strip()
                    if manual:
                        config["model"] = manual
                elif choice.isdigit() and 1 <= int(choice) <= len(models):
                    config["model"] = models[int(choice) - 1]
                save_config(config)
                console.print("[green]Model updated.[/green]")
            else:
                console.print("[yellow]Could not fetch models. Manual entry required.[/yellow]")
                manual = console.input("Enter model name: ").strip()
                if manual:
                    config["model"] = manual
                    save_config(config)
                    console.print("[green]Model updated.[/green]")
            continue

        new_val = console.input(f"New value for {name} (leave blank to cancel): ").strip()
        if new_val:
            config[key] = new_val
            save_config(config)
            console.print("[green]Updated.[/green]")

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

def read_file(path, offset=None, limit=None):
    """
    Read a file. Supports optional offset (line number) and limit (number of lines)
    for partial reads. Fully backward compatible with old calls.
    """
    if not os.path.exists(path):
        return "File not found."

    try:
        with open(path, "r") as f:
            lines = f.readlines()

        # Normalize offset
        if offset is not None:
            try:
                offset = int(offset)
            except:
                offset = 0
        else:
            offset = 0

        # Normalize limit
        if limit is not None:
            try:
                limit = int(limit)
            except:
                limit = None

        # Slice safely
        if limit is not None:
            sliced = lines[offset:offset + limit]
        else:
            sliced = lines[offset:]

        return "".join(sliced)

    except Exception as e:
        return f"Error reading file: {str(e)}"

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

# Helper to truncate tool output for context safety
def truncate_tool_output(output, limit=2000):
    """
    Truncate tool output before storing in conversation history
    to prevent token explosion.
    """
    text = str(output)
    if len(text) <= limit:
        return text
    return text[:limit] + "\n\n... (truncated for context safety)"

import re

# =====================================================
# SYSTEM PROMPT
# =====================================================

def get_system_prompt():
    prompt_path = Path(__file__).parent / "OPENCLI.md"
    if prompt_path.exists():
        with open(prompt_path, "r") as f: return f.read()
    return "You are OpenCLI, an autonomous coding agent."

SYSTEM_PROMPT = get_system_prompt() + """

=== TOOL USAGE RULES (STRICT) ===
When you decide to use a tool:
- Output ONLY a single JSON object.
- Do NOT wrap it in backticks.
- Do NOT prefix it with Thought:, Action:, or Explanation.
- Do NOT include any text before or after the JSON.
- The format MUST be exactly:

{"tool": "tool_name", "args": { ... }}

After a tool result is returned, you may respond normally.

Never simulate tool execution.
Never describe a tool call — only emit valid JSON.

=== CODE MODIFICATION RULES (REPLACE_TEXT FIRST POLICY) ===
When modifying code:
- ALWAYS prefer `replace_text` over `write_file`.
- Treat `write_file` as a LAST RESORT.
- If a file already exists, you MUST attempt `replace_text` first.
- Only use `write_file` when:
  1. Creating a completely new file, OR
  2. The user explicitly says: "rewrite the entire file".

Before calling a modification tool, you MUST:
1. Clearly state the file path.
2. State whether this is an insertion or replacement.
3. Show the exact snippet being replaced.
4. Show the exact new snippet being inserted.
5. Keep changes minimal and surgical.

Never rewrite entire files for small changes.
Never regenerate large unchanged sections.
Minimize token usage and preserve existing structure.

If unsure whether a full rewrite is necessary, default to `replace_text`.
"""

def extract_json(text):
    """
    Extract first valid tool JSON object from model output.
    Safely handles malformed trailing braces and embedded text.
    """
    cleaned = text.strip()

    # Scan for first balanced JSON object
    start = None
    brace_count = 0

    for i, ch in enumerate(cleaned):
        if ch == "{":
            if start is None:
                start = i
            brace_count += 1
        elif ch == "}":
            brace_count -= 1
            if brace_count == 0 and start is not None:
                candidate = cleaned[start:i+1]
                try:
                    obj = json.loads(candidate)
                    if isinstance(obj, dict) and "tool" in obj:
                        visible_text = cleaned[:start].strip()
                        return visible_text, obj
                except Exception:
                    pass
                start = None

    # No valid tool JSON found
    return cleaned, None

# --- ADD A NORMALIZER FUNCTION ---
def normalize_openai_tool_call(data):
    """
    Normalize OpenAI-style tool_calls or function_call into
    internal {"tool": name, "args": {...}} format.
    """
    try:
        if not isinstance(data, dict):
            return None

        # Handle tool_calls (new OpenAI format)
        if "choices" in data and data["choices"]:
            msg = data["choices"][0].get("message", {})
            if "tool_calls" in msg and msg["tool_calls"]:
                call = msg["tool_calls"][0]
                name = call.get("function", {}).get("name")
                args_raw = call.get("function", {}).get("arguments", "{}")
                try:
                    args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                except:
                    args = {}
                return {"tool": name, "args": args}

            # Handle legacy function_call
            if "function_call" in msg:
                fc = msg["function_call"]
                name = fc.get("name")
                args_raw = fc.get("arguments", "{}")
                try:
                    args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                except:
                    args = {}
                return {"tool": name, "args": args}

        return None
    except:
        return None

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
    # --- SANITIZE MESSAGES (fix Anthropic trailing whitespace error) ---
    sanitized_messages = []
    for m in messages:
        sanitized = m.copy()
        if isinstance(sanitized.get("content"), str):
            sanitized["content"] = sanitized["content"].rstrip()
        sanitized_messages.append(sanitized)

    messages = sanitized_messages
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
        headers["Accept"] = "text/event-stream"
        payload = {
            "model": provider_model,
            "messages": messages,
            "temperature": 0.7,
            "stream": True,
            "max_tokens": max_tokens
        }
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
                # Anthropic fallback logic for outdated model strings
                if provider_name == "Anthropic" and response.status_code == 404:
                    fallback_map = {
                        "claude-haiku-4-5-20251001": "claude-haiku-4-5",
                        "claude-sonnet-4-5-20250929": "claude-sonnet-4-5",
                        "claude-opus-4-6": "claude-opus-4-6"
                    }

                    fallback_model = fallback_map.get(provider_model)
                    if fallback_model and fallback_model != provider_model:
                        payload["model"] = fallback_model
                        retry = requests.post(url, headers=headers, json=payload, stream=True, timeout=60)
                        if retry.status_code == 200:
                            response = retry
                        else:
                            return f"[red]API Error ({retry.status_code}): {retry.text}[/red]"
                    else:
                        return f"[red]API Error ({response.status_code}): {response.text}[/red]"
                else:
                    try:
                        return f"[red]API Error ({response.status_code}): {response.text}[/red]"
                    except Exception:
                        return f"[red]API Error ({response.status_code}): Unknown response[/red]"

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
            final_text = full_text.strip()

            # 1️⃣ Normalize OpenAI-style tool_calls / function_call
            try:
                parsed = json.loads(final_text)
                normalized = normalize_openai_tool_call(parsed)
                if normalized:
                    return json.dumps(normalized)
            except:
                pass

            # 2️⃣ Normalize NVIDIA-style tool format:
            # <|tool_call_begin|> functions.read_file:5 ...
            if "<|tool_call_begin|>" in final_text:
                try:
                    name_match = re.search(r'functions\.(\w+)', final_text)
                    arg_match = re.search(r'\{.*\}', final_text, re.DOTALL)
                    if name_match and arg_match:
                        tool_name = name_match.group(1)
                        args = json.loads(arg_match.group(0))
                        return json.dumps({"tool": tool_name, "args": args})
                except:
                    pass

            # 3️⃣ Fallback to normal text output
            if thinking_text and full_text:
                return f"[dim]💭 Thinking: {thinking_text}[/dim]\n\n{full_text}"
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
    resumed = False

    global CURRENT_MESSAGES, CURRENT_SESSION_ID
    CURRENT_MESSAGES = messages
    CURRENT_SESSION_ID = session_id

    # Register graceful shutdown handlers
    signal.signal(signal.SIGTERM, handle_termination)
    signal.signal(signal.SIGHUP, handle_termination)
    atexit.register(persist_session_on_exit)

    if "--resume" in sys.argv:
        sel = select_session_menu()
        if sel:
            messages = sel["messages"]
            session_id = sel["id"]
            CURRENT_MESSAGES = messages
            CURRENT_SESSION_ID = session_id
            resumed = True
            console.print(f"[dim]📜 Resumed: {sel['title']}[/dim]")
            time.sleep(1)

    # If resumed, display previous conversation
    if resumed:
        banner(config.get("mode", "safe"), config.get("model"))
        total_msgs = len(messages)
        tool_count = len([m for m in messages if m.get("role") == "tool"])
        console.print(Panel(
            f"[bold]Session Summary[/bold]\n"
            f"Messages: {total_msgs}\n"
            f"Tool Calls: {tool_count}\n"
            f"Working Dir: {os.getcwd()}",
            border_style="cyan"
        ))

        # Show only last 6 conversational turns (no tool dumps)
        conversational = [m for m in messages if m["role"] in ("user", "assistant")]
        for m in conversational[-6:]:
            if m["role"] == "assistant":
                console.print(Panel(m["content"][:800], style="cyan", title="OpenCLI 💭"))
            elif m["role"] == "user":
                console.print(f"[bold {get_theme_color('text')}]You[/bold {get_theme_color('text')}] › {m['content']}")

    # Lightweight project context (no full directory dump)
    if not resumed:
        cwd = os.getcwd()
        messages.append({
            "role": "system",
            "content": (
                f"WORKING_DIRECTORY: {cwd}\n"
                "You may use tools like list_files, read_file, and search_code "
                "to explore the project as needed.\n"
                "Do NOT assume full project knowledge without using tools."
            )
        })
    
    mode = config.get("mode", "safe")
    if not resumed:
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
            console.print(
                f"\n[bold {text_c}]You[/bold {text_c}] "
                "[dim](/ → settings • /resume → load chat • /clear → reset • exit → quit)[/dim]"
            )
            user_input = console.input("› ")

            if user_input.strip() == "/":
                interactive_settings_menu()
                config = load_config()
                mode = config.get("mode", "safe")
                banner(mode, config.get("model"))
                continue

            if user_input.strip() == "/provider":
                interactive_settings_menu()
                config = load_config()
                banner(config.get("mode", "safe"), config.get("model"))
                continue

            if user_input.strip() == "/resume":
                sel = select_session_menu()
                if sel:
                    messages = sel["messages"]
                    session_id = sel["id"]
                    CURRENT_MESSAGES = messages
                    CURRENT_SESSION_ID = session_id
                    banner(config.get("mode", "safe"), config.get("model"))

                    total_msgs = len(messages)
                    tool_count = len([m for m in messages if m.get("role") == "tool"])
                    console.print(Panel(
                        f"[bold]Session Summary[/bold]\n"
                        f"Messages: {total_msgs}\n"
                        f"Tool Calls: {tool_count}\n"
                        f"Working Dir: {os.getcwd()}",
                        border_style="cyan"
                    ))

                    conversational = [m for m in messages if m["role"] in ("user", "assistant")]
                    for m in conversational[-6:]:
                        if m["role"] == "assistant":
                            console.print(Panel(m["content"][:800], style="cyan", title="OpenCLI 💭"))
                        elif m["role"] == "user":
                            console.print(f"[bold {get_theme_color('text')}]You[/bold {get_theme_color('text')}] › {m['content']}")
                continue
            if user_input.strip() == "/clear":
                messages = [{"role": "system", "content": SYSTEM_PROMPT}]
                session_id = None
                CURRENT_MESSAGES = messages
                CURRENT_SESSION_ID = session_id
                console.clear()
                config = load_config()
                banner(config.get("mode", "safe"), config.get("model"))
                continue
            if user_input.lower() in ["exit", "quit"]:
                session_id = save_history(messages, session_id)
                CURRENT_MESSAGES = messages
                CURRENT_SESSION_ID = session_id
                break

            messages.append({"role": "user", "content": user_input})
            session_id = save_history(messages, session_id)
            CURRENT_MESSAGES = messages
            CURRENT_SESSION_ID = session_id
            # Initial compaction per user turn
            messages, tokens, limit = compact_context(messages, p_model)
            console.print(Align.right(Text(f"Context: {(tokens*100)//limit}%", style="dim")))
            while True:
                reply = call_model(p_name, p_model, key, messages)

                # ====== NEW LOGIC: Assistant message → tool call (tool JSON not stored in history) ======
                reasoning, tool_call = extract_json(reply)

                # --- ENSURE extract_json HANDLES RAW JSON TOOL RETURNS CLEANLY ---
                # If model returned raw JSON (OpenAI-normalized), parse it directly
                if not tool_call:
                    try:
                        parsed = json.loads(reply.strip())
                        if isinstance(parsed, dict) and "tool" in parsed:
                            tool_call = parsed
                            reasoning = ""
                    except:
                        pass


                # If the model included normal text, show it and store it
                if reasoning.strip():
                    md = Markdown(reasoning)
                    console.print(
                        Panel(md, border_style="cyan", title="OpenCLI 💭")
                    )
                    messages.append({"role": "assistant", "content": reasoning})
                    session_id = save_history(messages, session_id)
                    CURRENT_MESSAGES = messages
                    CURRENT_SESSION_ID = session_id

                # If a tool call was requested, handle it separately
                if tool_call:
                    t_name, args = tool_call["tool"], tool_call.get("args", {})

                    if t_name not in TOOLS:
                        err_msg = f"Error: Tool {t_name} not found."
                        console.print(Panel(err_msg, style="bold red"))
                        messages.append({"role": "assistant", "content": err_msg})
                        session_id = save_history(messages, session_id)
                        CURRENT_MESSAGES = messages
                        CURRENT_SESSION_ID = session_id
                        break

                    approval = Prompt.ask(f"Approve {t_name}? [y/N]", choices=["y","n"], default="n", show_default=False)

                    if mode == "safe" and approval != "y":
                        console.print(
                            Panel(
                                "Tool execution denied.\n\nWhat would you like the agent to do instead?",
                                title="Denied",
                                style="yellow"
                            )
                        )
                        session_id = save_history(messages, session_id)
                        CURRENT_MESSAGES = messages
                        CURRENT_SESSION_ID = session_id
                        break

                    try:
                        result = TOOLS[t_name](**args)
                        # Show compact one-line tool usage (no full dump)
                        arg_preview = ", ".join(f"{k}={v}" for k, v in args.items())
                        console.print(
                            f"[green]Tool:[/green] {t_name}"
                            + (f" [dim]({arg_preview})[/dim]" if arg_preview else "")
                        )

                        truncated = truncate_tool_output(result)

                        # Append structured tool result (Anthropic does NOT support role="tool")
                        if p_name == "Anthropic":
                            messages.append({
                                "role": "assistant",
                                "content": f"[Tool Result: {t_name}]\n{truncated}"
                            })
                        else:
                            messages.append({
                                "role": "tool",
                                "name": t_name,
                                "content": truncated
                            })

                        session_id = save_history(messages, session_id)
                        CURRENT_MESSAGES = messages
                        CURRENT_SESSION_ID = session_id

                        # Re-compact after tool output before next model call
                        messages, tokens, limit = compact_context(messages, p_model)
                        console.print(Align.right(Text(f"Context: {(tokens*100)//limit}%", style="dim")))

                        continue

                    except Exception as e:
                        err_msg = f"Error executing tool '{t_name}': {str(e)}"
                        console.print(Panel(err_msg, title="Tool Error", style="bold red"))

                        # Feed tool failure back to model so it can retry or adjust
                        error_text = f"Tool failed with error: {str(e)}"
                        truncated_error = truncate_tool_output(error_text, limit=1000)

                        if p_name == "Anthropic":
                            messages.append({
                                "role": "assistant",
                                "content": f"[Tool Error: {t_name}]\n{truncated_error}"
                            })
                        else:
                            messages.append({
                                "role": "tool",
                                "name": t_name,
                                "content": truncated_error
                            })

                        session_id = save_history(messages, session_id)
                        CURRENT_MESSAGES = messages
                        CURRENT_SESSION_ID = session_id

                        # Re-compact after tool error before retry
                        messages, tokens, limit = compact_context(messages, p_model)
                        console.print(Align.right(Text(f"Context: {(tokens*100)//limit}%", style="dim")))

                        continue
                # If no tool call, finish the loop (after showing/storing reasoning if any)
                break
        except KeyboardInterrupt:
            quit_choice = Prompt.ask("Quit? [y/N]", choices=["y","n"], default="n", show_default=False)
            if quit_choice == "y":
                session_id = save_history(messages, session_id)
                CURRENT_MESSAGES = messages
                CURRENT_SESSION_ID = session_id
                break
            banner(mode, config.get("model"))

if __name__ == "__main__":
    main()