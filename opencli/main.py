#!/usr/bin/env python3

import os
import sys
import json
import subprocess
import difflib
import threading
import requests
import shutil
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text
from rich.align import Align
from rich.markdown import Markdown
from rich import box
import time
import select
import tty
import termios

import signal
import atexit
import collections
from filelock import FileLock
import uuid
import time as time_module

# ============================================================
# INTERRUPT / ESC SYSTEM
# _esc_event is set whenever ESC is pressed during model processing.
# It is cleared before each new agent turn.
# ============================================================
_esc_event = threading.Event()

def _start_esc_watcher():
    """
    Start a background thread that watches raw stdin for ESC (0x1b) while
    the model is processing.  The thread exits as soon as _esc_event is set
    (either because ESC was pressed, or because processing finished and the
    caller cleared + stopped watching).
    """
    _stop = threading.Event()

    def _watch():
        fd = sys.stdin.fileno()
        try:
            old = termios.tcgetattr(fd)
        except termios.error:
            return  # non-tty (piped input), skip
        try:
            tty.setraw(fd)
            while not _stop.is_set() and not _esc_event.is_set():
                r, _, _ = select.select([sys.stdin], [], [], 0.05)
                if r:
                    ch = os.read(fd, 1)
                    if ch == b'\x1b':
                        _esc_event.set()
                        break
        except Exception:
            pass
        finally:
            try:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)
            except Exception:
                pass

    t = threading.Thread(target=_watch, daemon=True)
    t.start()
    return _stop, t

# ============================================================
# CONSTANTS
# ============================================================
MAX_AGENT_STEPS    = 100     # max tool-call iterations per user message
MAX_FILE_LINES     = 500     # max lines shown in read_file preview
MAX_SEARCH_RESULTS = 50      # max results from search_code
SESSION_HISTORY_DAYS = 7     # how long to keep sessions
MAX_SESSIONS       = 20      # max sessions shown in /resume
TOOL_BATCH_DELAY   = 2.0     # seconds between batched tool calls

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style
from prompt_toolkit.patch_stdout import patch_stdout

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

def handle_resize(signum, frame):
  """Handle window resize - clear screen and reset"""
  sys.stdout.write("\033[2J\033[H\033[3J")
  sys.stdout.flush()

# Register resize handler
signal.signal(signal.SIGWINCH, handle_resize)

def clear_screen():
  """Clear screen and reset cursor position"""
  sys.stdout.write("\033[2J\033[H")
  sys.stdout.flush()
  console.clear()

# Initialize console
console = Console()




def get_mode_color(mode):
    """Get color for mode indicator"""
    colors = {
        "safe": "green",
        "unsafe": "red",
        "plan": "yellow"
    }
    return colors.get(mode, "green")


CONFIG_PATH = Path.home() / ".opencli"
CONFIG_FILE = CONFIG_PATH / "config.json"
HISTORY_FILE = CONFIG_PATH / "history.json"
SESSIONS_DIR = CONFIG_PATH / "sessions"

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
  "z-ai/glm-4.5-air:free": 128000,
  "qwen/qwen3-235b-a22b:free": 32768,
  "qwen/qwen3-30b-a3b:free": 32768,
  "qwen/qwen3-8b:free": 32768,
  "deepseek/deepseek-r1-0528:free": 163840,
  "meta/llama-3.3-70b-instruct": 128000,
  "meta/llama-3.1-8b-instruct": 128000,
  "mistralai/mistral-7b-instruct-v0.3": 32768,
}

def get_context_limit(model_name):
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

def format_time(seconds):
  """Convert seconds to readable format (1.2s, 45ms, etc)."""
  if seconds >= 1:
    return f"{seconds:.1f}s"
  else:
    return f"{int(seconds * 1000)}ms"

def format_tokens(count):
  """Format token count with commas."""
  return f"{count:,}" if count > 999 else str(count)

def count_tokens(messages):
  """
  Better token approximation:
  - English text: ~1 token per 4 chars
  - Code: ~1 token per 3 chars (more dense)
  - JSON: ~1 token per 3 chars
  """
  total = 0
  for m in messages:
    content = m.get("content") or ""
    if not isinstance(content, str):
      content = json.dumps(content)
    if any(x in content for x in ['{', '}', 'def ', 'class ', 'function', 'const ']):
      total += len(content) // 3
    else:
      total += len(content) // 4
  return total

def compact_context(messages, model_name):
  """Prunes history if context usage exceeds threshold."""
  limit = get_context_limit(model_name)
  try:
    threshold_pct = int(config.get("compaction_threshold", 75))
  except:
    threshold_pct = 75
    
  threshold = (limit * threshold_pct) // 100
  
  current_tokens = count_tokens(messages)
  if current_tokens < threshold:
    return messages, current_tokens, limit
  
  system_msgs = [m for m in messages if m["role"] == "system"]
  other_msgs = [m for m in messages if m["role"] != "system"]
  
  while count_tokens(system_msgs + other_msgs) > threshold and len(other_msgs) > 4:
    other_msgs = other_msgs[2:]
    
  compacted = system_msgs + other_msgs
  tokens_after = count_tokens(compacted)
  return compacted, tokens_after, limit

def save_history(messages, session_id=None):
  CONFIG_PATH.mkdir(exist_ok=True)
  SESSIONS_DIR.mkdir(exist_ok=True)

  if not session_id:
    session_id = str(uuid.uuid4())

  session_file = SESSIONS_DIR / f"{session_id}.json"
  lock = FileLock(str(session_file) + ".lock")

  cwd = os.getcwd()

  title = "New Chat"
  for m in messages:
    if m.get("role") == "user":
      content = m.get("content", "").strip().splitlines()[0]
      title = (content[:40] + "...") if len(content) > 40 else content
      break

  session_data = {
    "id": session_id,
    "timestamp": time.time(),
    "title": title,
    "cwd": cwd,
    "messages": messages
  }

  try:
    with lock:
      with open(session_file, "w") as f:
        json.dump(session_data, f)
  except:
    pass

  return session_id


def load_history():
  CONFIG_PATH.mkdir(exist_ok=True)
  SESSIONS_DIR.mkdir(exist_ok=True)

  sessions = []

  if HISTORY_FILE.exists():
    try:
      with open(HISTORY_FILE, "r") as f:
        legacy = json.load(f)
        sessions.extend(legacy.get("sessions", []))
    except:
      pass

  for file in SESSIONS_DIR.glob("*.json"):
    try:
      lock = FileLock(str(file) + ".lock")
      with lock:
        with open(file, "r") as f:
          data = json.load(f)
          sessions.append(data)
    except:
      continue

  now = time.time()
  sessions = [s for s in sessions if now - s.get("timestamp", 0) < SESSION_HISTORY_DAYS * 86400]

  sessions.sort(key=lambda x: x.get("timestamp", 0), reverse=True)

  return {"sessions": sessions[:MAX_SESSIONS]}

def get_char(fd):
  ch = sys.stdin.read(1)
  if ch == '\x1b':
    seq = ch
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
    console.print(Panel("No recent sessions found.", style="dim"))
    console.input("Press Enter to return...")
    return None

  console.print(Panel("[bold green]Resume Recent Chat[/bold green]", border_style="green"))

  while True:
    console.print()
    for i, s in enumerate(sessions, 1):
      is_local = s.get("cwd") == cwd
      tag = "[dim](here)[/dim] " if is_local else ""
      dt = time.strftime("%H:%M", time.localtime(s["timestamp"]))
      console.print(f"[green]{i}.[/green] {tag}{s['title']} [dim]({dt})[/dim]")

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
    "exa_key": "",
    "brave_key": "",
    "ollama_url": "http://localhost:11434/v1/chat/completions",
    "mode": "safe", # safe, unsafe, or plan
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
  # Set secure file permissions (owner read/write only)
  try:
    os.chmod(CONFIG_FILE, 0o600)
    os.chmod(CONFIG_PATH, 0o700)
  except:
    pass  # Non-Unix systems may not support this

config = load_config()

# ============================================================
# PROMPT TOOLKIT SESSION (input box + bottom toolbar)
# ============================================================

# Shared mutable state for the toolbar (updated before each prompt call)
_ui_state = {"mode": "safe", "model": "", "ctx_pct": 0}


def _make_toolbar():
    """Render the bottom toolbar: mode badge + hints + model + context %."""
    m     = _ui_state["mode"]
    model = _ui_state["model"]
    ctx   = _ui_state["ctx_pct"]
    mode_color = {"safe": "ansibrightgreen", "unsafe": "ansired", "plan": "ansigreen"}
    c = mode_color.get(m, "ansibrightgreen")
    return HTML(
        f' <b><style fg="{c}"> {m.upper()} </style></b>'
        f'  <style fg="ansiwhite">Shift+Tab · cycle</style>'
        f'  <style fg="ansibrightblack">│  {model}  {ctx}%</style>'
    )


def _print_status_bar(mode, model, ctx_pct):
    """Print a status bar matching the toolbar style, for use during AI processing."""
    colors = {"safe": "bold green", "unsafe": "bold red", "plan": "green"}
    c = colors.get(mode, "bold green")
    console.print(
        f"[{c}] {mode.upper()} [/{c}]"
        f"[dim]  Shift+Tab · cycle  │  {model}  {ctx_pct}%[/dim]"
    )


def create_prompt_session(msg_queue=None):
    """
    Build a PromptSession with:
      - Shift+Tab  → cycle mode
      - Enter      → submit (queue the message)
      - Alt+Enter  → insert newline (multiline input)
      - Escape     → clear the current buffer (and any queued messages)
    """
    kb = KeyBindings()

    @kb.add('s-tab')
    def _(event):
        _ui_state["mode"] = cycle_mode(_ui_state["mode"])
        config["mode"] = _ui_state["mode"]
        save_config(config)
        event.app.invalidate()

    @kb.add('enter')
    def _(event):
        """Submit current buffer."""
        event.current_buffer.validate_and_handle()

    @kb.add('escape')
    def _(event):
        """Clear buffer and queued messages."""
        event.current_buffer.reset()
        if msg_queue is not None:
            msg_queue.clear()
        event.app.invalidate()

    @kb.add('c-j')   # Ctrl+J = newline (fallback for terminals)
    @kb.add('escape', 'enter')  # Alt+Enter
    def _(event):
        """Insert a newline without submitting."""
        event.current_buffer.insert_text('\n')

    prompt_style = Style.from_dict({
        "prompt":              "bold ansigreen",
        "bottom-toolbar":      "bg:#1a1a1a fg:#666666",
        "bottom-toolbar.text": "bg:#1a1a1a",
    })

    return PromptSession(
        key_bindings=kb,
        style=prompt_style,
        bottom_toolbar=_make_toolbar,
        multiline=True,
        wrap_lines=True,
        mouse_support=False,
        complete_while_typing=False,
    )



def render_message(thinking="", output="", is_tool=False):
  """Render an assistant message with optional thinking prefix."""
  if is_tool:
    if output:
      console.print(f"[green]●[/green] {output}")
    return
  if not output and not thinking:
    return
  if output:
    console.print(Markdown(output))

def cycle_mode(current_mode):
  """Cycle through modes: SAFE -> UNSAFE -> PLAN -> SAFE"""
  modes = ["safe", "unsafe", "plan"]
  current_index = modes.index(current_mode) if current_mode in modes else 0
  next_index = (current_index + 1) % len(modes)
  return modes[next_index]

def get_mode_indicator(mode):
  """Get styled mode indicator for display"""
  if mode == "safe":
    return "[green]SAFE[/green]"
  elif mode == "unsafe":
    return "[red]UNSAFE[/red]"
  else:  # plan
    return "[dark_green]PLAN[/dark_green]"

def banner(mode, model):
  console.clear()

  cwd = os.getcwd().replace(os.path.expanduser("~"), "~")

  logo = r"""
  _ ____ _____ _  _ ____ _   ___ 
 / \| _ \| ___| \ | |/ ___| |  |_ _|
 | | | |_) | _| | \| | |  | |  | | 
 | | | __/| |___| |\ | |___| |___ | | 
 \_/|_|  |_____|_| \_|\____|_____|___|
"""

  ghost = """
 ⠀⠤⠐
 ⠀⠤⠐
 ⠀⠤⠐
"""
  ghost = "\n".join(line[3:] if len(line) >= 4 else line for line in ghost.splitlines())

  left_block = Text.assemble(
    (logo, "green"),
    ("\n- by curren -\n", "dim"),
    (f"\nFolder: {cwd}", "green"),
    (f"\nModel: {model}", "green"),
    (f"\nMode:  {mode.upper()}\n", "bold green"),
    ("\nTip: SAFE asks before tools • UNSAFE auto-runs • PLAN reads & analyzes\n", "dim")
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
  console.print("[dim](/ → settings • /resume → load chat • /clear → reset • exit → quit)[/dim]\n")


def interactive_settings_menu():
  console.print()
  console.print(Panel("[bold green]OpenCLI Settings[/bold green]", border_style="green"))

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
    ("EXA API Key (Web Search)", "exa_key"),
    ("Brave API Key (Web Search)", "brave_key"),
  ]

  while True:
    console.print()
    for i, (name, key) in enumerate(options, 1):
      value = str(config.get(key, ""))
      if "key" in key and value:
        value = value[:4] + "..." + value[-4:]
      console.print(f"[green]{i}.[/green] {name}: [green]{value if value else 'EMPTY'}[/green]")

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

    if key == "provider":
      console.print("\nSelect Provider:")
      providers = ["OpenRouter", "Anthropic", "Google", "OpenAI", "NVIDIA", "Ollama"]
      for i, p in enumerate(providers, 1):
        console.print(f"[green]{i}.[/green] {p}")
      choice = console.input("› ").strip()
      if choice.isdigit() and 1 <= int(choice) <= len(providers):
        new_provider = providers[int(choice) - 1]
        config["provider"] = new_provider
        default_models = config.get("provider_models", {})
        config["model"] = default_models.get(new_provider, config.get("model"))
        save_config(config)
        console.print("[green]Provider updated.[/green]")
      continue

    if key == "theme":
      console.print("\nTheme Options:")
      console.print("[green]1.[/green] dark")
      console.print("[green]2.[/green] light")
      t_choice = console.input("› ").strip()
      if t_choice == "1":
        config["theme"] = "dark"
      elif t_choice == "2":
        config["theme"] = "light"
      save_config(config)
      console.print("[green]Theme updated.[/green]")
      continue

    if key == "mode":
      console.print("\nExecution Mode:")
      console.print("[green]1.[/green] SAFE  (requires approval for each tool)")
      console.print("[green]2.[/green] UNSAFE (auto-executes tools without asking)")
      console.print("[green]3.[/green] PLAN  (reads only - analyzes and plans, never executes)")
      m_choice = console.input("› ").strip()
      if m_choice == "1":
        config["mode"] = "safe"
      elif m_choice == "2":
        config["mode"] = "unsafe"
      elif m_choice == "3":
        config["mode"] = "plan"
      save_config(config)
      console.print("[green]Mode updated.[/green]")
      continue

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
            return [
              # Moonshot
              "moonshotai/kimi-k2.5",
              # GLM
              "z-ai/glm5",
              # Qwen 3
              "qwen/qwen3-235b-a22b",
              "qwen/qwen3-next-80b-a3b",
              "qwen/qwen2.5-7b-instruct",
              "qwen/qwen2.5-coder-7b-instruct",
              # DeepSeek
              "deepseek-ai/deepseek-r1",
              "deepseek-ai/deepseek-r1-distill-llama-8b",
              # Llama Instruct
              "meta/llama-3.3-70b-instruct",
              "meta/llama-3.1-8b-instruct",
              "meta/llama-3.1-70b-instruct",
              # Mistral Instruct
              "mistralai/mistral-7b-instruct-v0.3",
              "mistralai/mixtral-8x7b-instruct",
              # NVIDIA Nemotron
              "nvidia/nemotron-3-nano-30b-a3b",
              "nvidia/nemotron-mini-4b-instruct",
              # Phi
              "microsoft/phi-3-mini-4k-instruct",
              # Gemma
              "google/gemma-2-9b-it",
            ]

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
          console.print(f"[green]{i}.[/green] {m}")
        console.print("[green]M.[/green] Manual entry")
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
        console.print("[green]Could not fetch models. Manual entry required.[/green]")
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

# File cache for storing full file contents to avoid re-reading
# Key: absolute path, Value: full file content string
file_cache = {}

def cache_clear():
  """Clear the file cache. Useful when files are modified externally."""
  file_cache.clear()
  return "File cache cleared."

def read_file(path, offset=None, limit=None, query=None, search=None):
  """
  Read a file. Supports optional parameters for intelligent reading:
  - offset: Line number to start reading from (0-indexed)
  - limit: Maximum number of lines to read
  - query/search: Search for specific text within the file and return matching lines
    with context. This is more efficient than reading the entire file.
  
  The file is cached for subsequent operations.
  """
  if not os.path.exists(path):
    return "File not found."

  try:
    abs_path = os.path.abspath(path)
    
    # If searching, don't load entire file - use grep for efficient search
    search_query = query or search
    if search_query:
      try:
        result = subprocess.run(
          ["grep", "-n", "-i", "-C", "2", search_query, abs_path],
          capture_output=True, text=True, timeout=10
        )
        matches = result.stdout.strip()
        if matches:
          lines_found = len(matches.splitlines())
          # If too many matches, summarize
          if lines_found > 50:
            return f"Found {lines_found} matches for '{search_query}'. First 50:\n\n{matches.splitlines()[:50]}\n\n... (truncated)\n\nUse read_file with offset/limit to view specific sections."
          return f"Found {lines_found} matches for '{search_query}':\n\n{matches}"
        else:
          return f"No matches found for '{search_query}' in {path}."
      except Exception as e:
        # Fall back to cached search if grep fails
        pass
    
    # Read full file and cache it (only if not searching)
    if abs_path not in file_cache:
      with open(abs_path, "r") as f:
        file_cache[abs_path] = f.read()

    # Use cached content
    lines = file_cache[abs_path].splitlines(keepends=True)

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
  abs_path = os.path.abspath(path)
  os.makedirs(os.path.dirname(abs_path), exist_ok=True)
  with open(abs_path, "w") as f:
    f.write(content)
  file_cache[abs_path] = content
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

def replace_text(path, **kwargs):
  # Support multiple parameter names: old_text/new_text, old/new, oldString/newString
  old_text = kwargs.get('old_text') or kwargs.get('old') or kwargs.get('oldString')
  new_text = kwargs.get('new_text') or kwargs.get('new') or kwargs.get('newString')

  if not old_text or not new_text:
    return "Error: replace_text requires old_text/new_text (or old/new/oldString/newString) parameters."

  abs_path = os.path.abspath(path)
  if not os.path.exists(path): return f"Error: File '{path}' not found."
  try:
    with open(abs_path, "r") as f: content = f.read()
    if old_text not in content: return f"Error: Could not find exact match."
    new_content = content.replace(old_text, new_text, 1)
    with open(abs_path, "w") as f: f.write(new_content)
    # Update cache
    file_cache[abs_path] = new_content
    return f"Successfully replaced text in {path}."
  except Exception as e:
    return f"Error: {str(e)}"

def search_code(query, path="."):
  """
  Search for query in cached files. Uses file_cache for previously read files,
  falling back to grep for uncached files.
  """
  matches = []

  # First, search in cached files
  for abs_path, content in file_cache.items():
    if path != "." and not abs_path.startswith(os.path.abspath(path)):
      continue
    lines = content.splitlines()
    for i, line in enumerate(lines, 1):
      if query.lower() in line.lower():
        matches.append(f"{abs_path}:{i}:{line}")

  # Fall back to grep for uncached files
  try:
    result = subprocess.run(
      ["grep", "-r", "-n", "--exclude-dir={.git,node_modules,__pycache__}", query, path],
      capture_output=True, text=True, timeout=30
    )
    for line in result.stdout.strip().splitlines():
      # Skip if already in matches (from cache)
      if line not in [m.split(":", 2)[-1] if ":" in m else "" for m in matches]:
        matches.append(line)
  except:
    pass

  if not matches:
    return f"No matches found for '{query}'."

  if len(matches) > 50:
    return "\n".join(matches[:50]) + f"\n\n... (truncated to 50 matches)"

  return "\n".join(matches)

def web_search(query, num_results=8, provider="auto"):
  """
  Search the web for current information. Use this when you need:
  - Recent news or events
  - Technical documentation not in the codebase
  - API references or library information
  - Answers to questions that require up-to-date information
  
  Args:
    query: Search query string
    num_results: Number of results to return (default 8)
    provider: "exa", "brave", or "auto" (tries exa first, then brave)
  
  Returns top search results with snippets.
  """
  exa_key = config.get("exa_key", "")
  brave_key = config.get("brave_key", "")
  
  # Auto-detect provider based on available keys
  if provider == "auto":
    if exa_key and brave_key:
      provider = "exa"  # prefer exa
    elif exa_key:
      provider = "exa"
    elif brave_key:
      provider = "brave"
    else:
      return "Error: No web search API key configured. Run --settings to add EXA_API_KEY or BRAVE_API_KEY."
  elif provider == "exa" and not exa_key:
    return "Error: No EXA_API_KEY configured. Run --settings to add your API key."
  elif provider == "brave" and not brave_key:
    return "Error: No BRAVE_API_KEY configured. Run --settings to add your API key."
  
  try:
    if provider == "exa":
      headers = {
        "Accept": "application/json",
        "Authorization": f"{exa_key}"
      }
      params = {
        "query": query,
        "num_results": num_results,
        "type": "auto"
      }
      resp = requests.get(
        "https://api.exa.ai/search",
        headers=headers,
        params=params,
        timeout=30
      )
      if resp.status_code != 200:
        return f"Exa search error: {resp.status_code} - {resp.text[:200]}"
      
      results = resp.json().get("results", [])
      if not results:
        return f"No results found for: {query}"
      
      output = []
      for i, r in enumerate(results, 1):
        title = r.get("title", "Untitled")
        url = r.get("url", "")
        snippet = r.get("snippet", "")[:300]
        output.append(f"{i}. {title}\n   {url}\n   {snippet}\n")
      
      return "\n".join(output)
    
    elif provider == "brave":
      headers = {
        "Accept": "application/json",
        "X-Subscription-Token": brave_key
      }
      params = {
        "q": query,
        "count": num_results
      }
      resp = requests.get(
        "https://api.search.brave.com/res/v1/web/search",
        headers=headers,
        params=params,
        timeout=30
      )
      if resp.status_code != 200:
        return f"Brave search error: {resp.status_code} - {resp.text[:200]}"
      
      data = resp.json()
      results = data.get("web", {}).get("results", [])
      if not results:
        return f"No results found for: {query}"
      
      output = []
      for i, r in enumerate(results, 1):
        title = r.get("title", "Untitled")
        url = r.get("url", "")
        snippet = r.get("description", "")[:300]
        output.append(f"{i}. {title}\n   {url}\n   {snippet}\n")
      
      return "\n".join(output)
  
  except Exception as e:
    return f"Search failed: {str(e)}"

TOOLS = {
  "list_files": list_files,
  "read_file": read_file,
  "write_file": write_file,
  "replace_text": replace_text,
  "search_code": search_code,
  "web_search": web_search,
  "run_shell": run_shell,
  "cache_clear": cache_clear
}

TOOL_METADATA = {
  "list_files": {
    "description": "List files in a directory",
    "destructive": False
  },
  "read_file": {
    "description": "Read file contents",
    "destructive": False
  },
  "write_file": {
    "description": "Create or overwrite a file",
    "destructive": False  # resolved dynamically in get_tool_approval
  },
  "replace_text": {
    "description": "Edit text in an existing file",
    "destructive": True
  },
  "search_code": {
    "description": "Search code with grep",
    "destructive": False
  },
  "run_shell": {
    "description": "Run a shell command",
    "destructive": False  # resolved dynamically in get_tool_approval
  },
  "web_search": {
    "description": "Search the web for current information, documentation, or answers",
    "destructive": False
  },
  "cache_clear": {
    "description": "Clear the file cache",
    "destructive": False
  }
}

# Shell command patterns that are actually destructive
_DESTRUCTIVE_SHELL_PATTERNS = [
  "rm ", "rm\t", "rmdir", "del ", "unlink",
  "mv ", "mv\t",        # move can overwrite
  "dd ", "dd\t",        # disk write
  "> ",                  # redirect/overwrite
  "truncate",
  "shred", "wipe",
  "chmod", "chown",     # permission changes
  "sudo ",
  "DROP ", "DROP\t",    # SQL
  "DELETE FROM",
  "mkfs",               # format
  "fdisk", "parted",
]

def is_destructive_shell(command):
  """Return True if a shell command is potentially destructive."""
  cmd = command.strip()
  cmd_upper = cmd.upper()
  return any(pat.upper() in cmd_upper for pat in _DESTRUCTIVE_SHELL_PATTERNS)

TOOL_SCHEMAS = [
  {
    "type": "function",
    "function": {
      "name": "list_files",
      "description": "List files in a directory",
      "parameters": {
        "type": "object",
        "properties": {
          "path": {"type": "string", "description": "Directory path to inspect"}
        },
        "additionalProperties": False
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "search_code",
      "description": "Search code by text query",
      "parameters": {
        "type": "object",
        "properties": {
          "query": {"type": "string", "description": "Search text"},
          "path": {"type": "string", "description": "Path to search from"}
        },
        "required": ["query"],
        "additionalProperties": False
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "read_file",
      "description": "Read file contents. Use query/search to find specific text without loading entire file - much more token efficient.",
      "parameters": {
        "type": "object",
        "properties": {
          "path": {"type": "string", "description": "File path"},
          "offset": {"type": "integer", "description": "Line offset (0-indexed)"},
          "limit": {"type": "integer", "description": "Max lines to read"},
          "query": {"type": "string", "description": "Search for text in file (case-insensitive, returns matching lines with context). Efficient!"},
          "search": {"type": "string", "description": "Alias for query - search for text in file"}
        },
        "required": ["path"],
        "additionalProperties": False
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "write_file",
      "description": "Create or overwrite a file",
      "parameters": {
        "type": "object",
        "properties": {
          "path": {"type": "string", "description": "File path"},
          "content": {"type": "string", "description": "Full file content"}
        },
        "required": ["path", "content"],
        "additionalProperties": False
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "replace_text",
      "description": "Replace text in an existing file",
      "parameters": {
        "type": "object",
        "properties": {
          "path": {"type": "string", "description": "File path"},
          "old_text": {"type": "string", "description": "Text to replace"},
          "new_text": {"type": "string", "description": "Replacement text"},
          "old": {"type": "string", "description": "Alias for old_text"},
          "new": {"type": "string", "description": "Alias for new_text"}
        },
        "required": ["path"],
        "additionalProperties": False
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "run_shell",
      "description": "Execute a shell command",
      "parameters": {
        "type": "object",
        "properties": {
          "command": {"type": "string", "description": "Shell command to execute"}
        },
        "required": ["command"],
        "additionalProperties": False
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "web_search",
      "description": "Search the web for current information, documentation, news, or answers. Use when you need up-to-date information not in the codebase.",
      "parameters": {
        "type": "object",
        "properties": {
          "query": {"type": "string", "description": "Search query"},
          "num_results": {"type": "integer", "description": "Number of results (default 8)"}
        },
        "required": ["query"],
        "additionalProperties": False
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "cache_clear",
      "description": "Clear cached file contents",
      "parameters": {
        "type": "object",
        "properties": {},
        "additionalProperties": False
      }
    }
  }
]


def normalize_tool_args(args_raw):
  if isinstance(args_raw, dict):
    return args_raw
  if isinstance(args_raw, str):
    try:
      parsed = json.loads(args_raw)
      return parsed if isinstance(parsed, dict) else {}
    except:
      return {}
  return {}


def truncate_tool_output(output, limit=2000):
  """
  Truncate tool output before storing in conversation history
  to prevent token explosion.
  """
  text = str(output)
  if len(text) <= limit:
    return text
  return text[:limit] + "\n\n... (truncated for context safety)"

# --- Tool Error/Rate Limit Helpers ---
def is_rate_limit_error(error_msg):
  indicators = [
    "rate_limit", "rate limit", "too many requests",
    "429", "quota exceeded", "max requests"
  ]
  msg = str(error_msg).lower()
  return any(ind in msg for ind in indicators)


def handle_tool_error(tool_name, error_msg, messages):
  console.print(
    Panel(
      f"[bold red]Tool Error: {tool_name}[/bold red]\n\n{error_msg}",
      border_style="red",
      title="Error"
    )
  )
  console.print("[bold]Options:[/bold]")
  console.print("  [green]r[/green] Retry")
  console.print("  [red]s[/red] Skip")
  console.print("  [dim]c[/dim] Continue (add error to chat)")
  console.print("  [red]q[/red] Quit tool loop")

  while True:
    key = get_tool_approval_key()
    if key == 'r':
      return 'retry'
    if key == 's':
      return 'skip'
    if key == 'c':
      messages.append({"role": "assistant", "content": f"Tool '{tool_name}' failed: {error_msg}"})
      return 'continue'
    if key == 'q':
      return 'quit'

def generate_unified_diff(file_path, old_content, new_content):
  """Generate a unified diff between old and new content."""
  old_lines = old_content.splitlines(keepends=True) if old_content else []
  new_lines = new_content.splitlines(keepends=True) if new_content else []

  diff = difflib.unified_diff(
    old_lines, new_lines,
    fromfile=f"{file_path} (before)",
    tofile=f"{file_path} (after)",
    lineterm=''
  )
  return ''.join(diff)

def show_tool_execution(tool_name, args, tool_func, plan_mode=False, modified_files=None):
  """
  Execute tool with Claude Code-like UX:
  - Shows tool being called with spinner
  - Displays file diffs for modifications
  - Shows results inline (scrollable)
  - Auto-collapses after completion

  Returns: (result, display_panels, tool_result_text)
  """
  # In plan mode, only skip destructive tools - execute read tools to build context
  if plan_mode:
    read_only_tools = {"list_files", "read_file", "search_code", "cache_clear", "web_search"}
    if tool_name not in read_only_tools:
      return None, "[dark_green]→ PLAN mode: skipped (write tool)[/dark_green]", None
    # Execute read-only tools normally to build context

  # Show execution with spinner
  arg_str = ", ".join([f"{k}={v}" for k, v in args.items()])
  console.print()

  try:
    with console.status(f"[green] {tool_name}[/green] {arg_str}", spinner="dots"):
      result = tool_func(**args)
  except Exception as e:
    error_type = type(e).__name__
    error_msg = f"{tool_name} failed: {error_type}: {str(e)}"
    console.print(Panel(f"[red]{error_msg}[/red]", title=" Tool Error", border_style="red"))
    # Return error message that will be fed back to model for retry
    return None, "", f"TOOL FAILURE: {error_msg}\n\nThe {tool_name} tool failed. Please try again with different parameters or approach."

  result_text = str(result)
  display_panels = []

  # === FILE MODIFICATION HANDLING ===
  if tool_name == "replace_text":
    file_path = args.get("path", "?")
    old_text = args.get("old_text", "")
    new_text = args.get("new_text", "")

    # Track modified file
    if modified_files is not None:
      modified_files.add(file_path)

    # Show clear diff like Claude Code
    old_lines = old_text.split('\n')
    new_lines = new_text.split('\n')

    # Format with line-by-line diff
    diff_lines = []
    diff_lines.append(f"[green]{file_path}[/green]")

    # Show removed lines (red)
    for i, line in enumerate(old_lines, 1):
      if line.strip():
        diff_lines.append(f"[red]{i}: {line[:60]}[/red]")

    diff_lines.append("[dim]    ↓[/dim]")

    # Show added lines (green)
    for i, line in enumerate(new_lines, 1):
      if line.strip():
        diff_lines.append(f"[green]{i}: {line[:60]}[/green]")

    diff_content = "\n".join(diff_lines)

    display_panels.append(
      Panel(
        diff_content,
        title=f"[green]● Replace[/green]",
        border_style="green",
        padding=(0, 1)
      )
    )

  elif tool_name == "write_file":
    file_path = args.get("path", "?")
    content = args.get("content", "")
    lines = len(content.splitlines())

    # Track modified file
    if modified_files is not None:
      modified_files.add(file_path)

    # Compact display
    display_panels.append(
      Panel(
        f"[green]{file_path}[/green]\n[dim]{lines} lines[/dim]",
        title="[green]● Write[/green]",
        border_style="green",
        padding=(0, 1)
      )
    )

  # === STANDARD OUTPUT ===
  else:
    # For read operations, show the output in Claude Code style
    if result_text.strip() and result_text not in ["(Command executed with no output)", "No files found."]:
      # Format based on tool type with Claude Code style
      if tool_name == "list_files":
        title = "Files Found"
        truncated = truncate_tool_output(result_text, limit=500)
        content = truncated

      elif tool_name == "read_file":
        # Show file path with optional offset/limit
        file_path = args.get("path", "?")
        offset = args.get("offset", 0)
        limit = args.get("limit", "")
        location = f"{file_path}:{offset}" + (f"-{limit}" if limit else "")
        title = location

        # Show first few lines as preview
        lines = result_text.split("\n")
        preview_lines = lines[:10]
        truncated = "\n".join(preview_lines)
        if len(lines) > 10:
          truncated += f"\n... ({len(lines)} total lines)"
        content = truncated

      elif tool_name == "search_code":
        # Show search query and matches
        query = args.get("query", args.get("path", "search"))
        title = f"Search: {query}"
        truncated = truncate_tool_output(result_text, limit=500)
        content = truncated

      else:
        title = tool_name
        truncated = truncate_tool_output(result_text, limit=500)
        content = truncated

      display_panels.append(
        Panel(
          content,
          title=f"[green]● {title}[/green]",
          border_style="green",
          padding=(1, 2)
        )
      )

  return result, display_panels, result_text

import re


def get_system_prompt():
  prompt_path = Path(__file__).parent / "OPENCLI.md"
  if prompt_path.exists():
    with open(prompt_path, "r") as f:
      return f.read().strip()
  return "You are OpenCLI, an autonomous coding agent."


def build_system_prompt(mode="safe"):
  """
  Build full system prompt dynamically:
  - Always reload OPENCLI.md (session-level behavior)
  - Always reload TOOLS.md (request-level tool awareness)
  - Inject current execution mode rules
  """
  base = get_system_prompt()

  tools_path = Path(__file__).parent / "TOOLS.md"
  tools_text = ""
  if tools_path.exists():
    with open(tools_path, "r") as f:
      tools_text = f.read().strip()

  strict_rules = """
=== TOOL USAGE RULES (STRICT) ===
When you decide to use a tool:
- Output ONLY a single JSON object.
- Do NOT wrap it in backticks.
- Do NOT prefix it with Thought:, Action:, or Explanation.
- You may include exactly one short plain-English line BEFORE the JSON to tell the user what you are about to do.
- Do NOT include any text after the JSON.
- The format MUST be exactly:

{"tool": "tool_name", "args": { ... }}

After a tool result is returned, you may respond normally.

For requests that require editing, creating, or verifying project files:
- You MUST call at least one real tool before giving a final answer.
- Never claim a file was created or changed unless a write tool was actually executed.

Never simulate tool execution.
Never describe a tool call — only emit valid JSON.

=== CURRENT EXECUTION MODE: {mode_upper} ===
{mode_instructions}

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

=== WEB SEARCH USAGE ===
When you need CURRENT information (news, events, documentation, API references):
- Use `web_search` tool to search the web
- This is MUCH better than guessing or using stale training data
- For questions about recent events, sports, technology updates, etc., ALWAYS search first
- Example: "who won the latest Olympics" → use web_search to get accurate answer
- The tool returns title, URL, and snippet for each result

If no API key is configured, inform the user: "I don't have web search configured. Run --settings to add an EXA or Brave API key."
"""

  mode_upper = mode.upper()
  if mode == "safe":
    mode_instructions = (
      "You are in SAFE mode.\n"
      "- You MAY call any tool. The user will be prompted to approve each destructive action before it runs.\n"
      "- Read-only tools (read_file, list_files, search_code) execute immediately without approval.\n"
      "- Destructive tools (write_file, replace_text, run_shell with write/delete commands) pause and wait for user approval.\n"
      "- Proceed step by step: call a tool, wait for its result, then decide the next step.\n"
      "- Do NOT batch multiple destructive calls — request one at a time so the user can review each change.\n"
      "- You are sandboxed to the working directory. Never access paths outside it."
    )
  elif mode == "unsafe":
    mode_instructions = (
      "You are in UNSAFE mode.\n"
      "- All tools execute automatically without asking for approval.\n"
      "- You are strictly sandboxed to the working directory. Never read, write, or execute anything outside it.\n"
      "- Never use absolute paths outside the project root. Never use ~ or /etc or /usr or any system path.\n"
      "- Never run commands that affect the host system (no brew, pip install --global, npm install -g, etc.).\n"
      "- Never delete files outside the project. Never kill processes. Never open network connections.\n"
      "- Work autonomously: chain tool calls until the task is fully complete, then summarize what was done."
    )
  else:  # plan
    mode_instructions = (
      "You are in PLAN mode. This is READ-ONLY — you must not write, modify, or execute anything.\n"
      "- Permitted tools: read_file, list_files, search_code ONLY.\n"
      "- Do NOT call write_file, replace_text, or run_shell under any circumstances.\n"
      "- Your job: thoroughly understand the codebase, then produce a detailed, actionable plan.\n"
      "  1. Read all relevant files and search for key patterns.\n"
      "  2. After gathering context, STOP reading and think.\n"
      "  3. Output a structured plan: which files change, exactly what changes, and why.\n"
      "  4. Do not repeat searches. Be efficient — 3-5 targeted reads is enough.\n"
      "- The user will switch to SAFE or UNSAFE mode when they are ready to execute your plan."
    )

  mode_block = f"=== CURRENT EXECUTION MODE: {mode_upper} ===\n{mode_instructions}"
  final_rules = strict_rules.replace("=== CURRENT EXECUTION MODE: {mode_upper} ===\n{mode_instructions}", mode_block)
  return f"{base}\n\n{tools_text}\n\n{final_rules}"

def extract_json(text):
  if not text:
    return "", None

  raw = text.strip()

  # Strip common markdown fences
  raw = re.sub(r"```(?:json)?", "", raw, flags=re.IGNORECASE).strip()

  # Strip rich panel borders and box characters
  raw = raw.replace("│", "").replace("╭", "").replace("╰", "").replace("─", "")
  raw = raw.strip()

  # If the entire cleaned text is valid JSON, try that first
  try:
    parsed_full = json.loads(raw)
    if isinstance(parsed_full, dict) and "tool" in parsed_full:
      parsed_full["args"] = normalize_tool_args(parsed_full.get("args", {}))
      return "", parsed_full
    if isinstance(parsed_full, dict) and "name" in parsed_full:
      # OpenAI style: {"name": "tool", "arguments": {...}}
      return "", {"tool": parsed_full.get("name"), "args": normalize_tool_args(parsed_full.get("arguments", {}))}
    if isinstance(parsed_full, dict) and "function" in parsed_full:
      # Anthropic style: {"function": "tool", "arguments": {...}}
      return "", {"tool": parsed_full.get("function"), "args": normalize_tool_args(parsed_full.get("arguments", {}))}
    if isinstance(parsed_full, list):
      # Check for OpenAI-style tool calls in array
      tool_calls = []
      for item in parsed_full:
        if isinstance(item, dict):
          if "tool" in item:
            item["args"] = normalize_tool_args(item.get("args", {}))
            tool_calls.append(item)
          elif "name" in item:
            tool_calls.append({"tool": item.get("name"), "args": normalize_tool_args(item.get("arguments", {}))})
          elif "function" in item:
            tool_calls.append({"tool": item.get("function"), "args": normalize_tool_args(item.get("arguments", {}))})
      if tool_calls:
        return "", tool_calls
  except:
    pass

  # Also check for tool calls anywhere in the text (including thinking)
  # This catches malformed JSON that the model puts in thinking
  # Use a more flexible pattern that handles nested braces
  tool_pattern = r'\{\s*"tool"\s*:\s*"(\w+)"\s*,\s*"args"\s*:\s*(\{.*?\})'
  matches = re.findall(tool_pattern, raw, re.DOTALL)
  if matches:
    tool_calls = []
    for name, args_str in matches:
      try:
        # Try to parse args as JSON - skip if invalid
        args = json.loads(args_str)
        tool_calls.append({"tool": name, "args": args})
      except:
        # Skip malformed tool calls
        continue
    if tool_calls:
      return "", tool_calls

  # Also check for OpenAI style: {"name": "tool", "arguments": {...}}
  openai_pattern = r'\{\s*"name"\s*:\s*"(\w+)"\s*,\s*"arguments"\s*:\s*(\{.*?\})'
  matches = re.findall(openai_pattern, raw, re.DOTALL)
  if matches:
    tool_calls = []
    for name, args_str in matches:
      try:
        args = json.loads(args_str)
        tool_calls.append({"tool": name, "args": args})
      except:
        continue
    if tool_calls:
      return "", tool_calls

  # Find ALL JSON objects (not minimal, but balanced by attempting parse)
  candidates = re.findall(r'\{[\s\S]*?\}', raw)

  for candidate in candidates:
    candidate = candidate.strip()
    try:
      parsed = json.loads(candidate)
      if isinstance(parsed, dict) and "tool" in parsed:
        parsed["args"] = normalize_tool_args(parsed.get("args", {}))
        return "", parsed
      if isinstance(parsed, dict) and "name" in parsed:
        return "", {"tool": parsed.get("name"), "args": normalize_tool_args(parsed.get("arguments", {}))}
      if isinstance(parsed, dict) and "function" in parsed:
        return "", {"tool": parsed.get("function"), "args": normalize_tool_args(parsed.get("arguments", {}))}
    except:
      continue

  # Find JSON arrays for batch tool calls
  array_candidates = re.findall(r'\[[\s\S]*?\]', raw)

  for candidate in array_candidates:
    candidate = candidate.strip()
    try:
      parsed = json.loads(candidate)
      if isinstance(parsed, list):
        tool_calls = []
        for item in parsed:
          if isinstance(item, dict):
            if "tool" in item:
              item["args"] = normalize_tool_args(item.get("args", {}))
              tool_calls.append(item)
            elif "name" in item:
              tool_calls.append({"tool": item.get("name"), "args": normalize_tool_args(item.get("arguments", {}))})
            elif "function" in item:
              tool_calls.append({"tool": item.get("function"), "args": normalize_tool_args(item.get("arguments", {}))})
        if tool_calls:
          return "", tool_calls
    except:
      continue

  return raw, None


def normalize_openai_tool_call(data):
  """
  Normalize OpenAI-style tool_calls or function_call into
  internal {"tool": name, "args": {...}, "id": ...} format.
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
        tool_call_id = call.get("id")  # Get the tool_call_id
        try:
          args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
        except:
          args = {}
        return {"tool": name, "args": args, "id": tool_call_id}

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


def check_ollama(url):
  try:
    base_url = url.split("/v1")[0]
    response = requests.get(base_url, timeout=2)
    return response.status_code == 200
  except: return False

def get_tool_approval_key():
  """Get single-key input: y/n/q (quit)/t (toggle) without requiring Enter."""
  fd = sys.stdin.fileno()
  old_settings = termios.tcgetattr(fd)
  try:
    tty.setraw(fd)
    ch = sys.stdin.read(1)

    # Check for escape sequences
    if ch == '\x1b': # ESC
      # Read all available characters in the escape sequence
      seq = ch
      while len(seq) < 10:
        ready, _, _ = select.select([fd], [], [], 0.05)
        if ready:
          seq += sys.stdin.read(1)
        else:
          break
      
      # Check for Shift+Tab (ESC [ Z) - handle both cases
      if seq.startswith('\x1b[') and seq.endswith(('Z', 'z')):
        return 't'
      # Any other escape - ignore and re-prompt (don't quit!)
      return None
    if ch.lower() in ['y', 'n']:
      return ch.lower()
    elif ch.lower() == 'q':
      return 'q'
    elif ch.lower() == 't':
      return 't'
    elif ch == '\r' or ch == '\n':
      return None
    elif ch == '\x03':
      raise KeyboardInterrupt
    else:
      return None
  finally:
    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

def get_tool_approval(tool_name, args, mode, plan_mode=False):
  """
  Robust tool approval with metadata, single-key input, and mode toggle.
  Returns: True (approve), False (deny), 'toggle' (switch modes), or 'skip' (plan mode).
  Auto-approves read-only tools silently without showing any panel.
  """
  metadata = TOOL_METADATA.get(tool_name, {})
  description = metadata.get("description", "Unknown tool")

  # Dynamically resolve destructive flag based on actual args
  if tool_name == "write_file":
    path = args.get("path", "")
    is_destructive = bool(path and os.path.exists(path))
    description = "Overwrite existing file" if is_destructive else "Create new file"
  elif tool_name == "run_shell":
    cmd = args.get("command", "")
    is_destructive = is_destructive_shell(cmd)
    description = "Destructive shell command" if is_destructive else "Run shell command"
  else:
    is_destructive = metadata.get("destructive", False)

  # Safe read-only shell commands — auto-approve silently
  _SAFE_SHELL_CMDS = {
    "pwd", "echo $PWD", "ls", "ls -la", "ls -l", "ls -a",
    "git status", "git log --oneline -10", "git branch",
    "git diff --stat", "cat", "which", "whereis",
    "python --version", "python3 --version", "node --version",
    "npm --version", "pip --version", "pip3 --version",
  }
  cmd_stripped = args.get("command", "").strip()
  if tool_name == "run_shell" and (
    cmd_stripped in _SAFE_SHELL_CMDS
    or cmd_stripped.startswith(("ls ", "cat ", "which ", "git log", "git show", "git diff"))
  ):
    console.print(f"[dim]  [green]$ {cmd_stripped}[/green][/dim]")
    return True

  # Read-only tools - always auto-approve silently
  read_only_tools = {"list_files", "read_file", "search_code", "cache_clear", "web_search"}
  if tool_name in read_only_tools:
    return True

  # Non-destructive write_file (new file) and non-destructive shell — auto-approve silently
  if not is_destructive and tool_name in ("write_file", "run_shell"):
    arg_str = ", ".join(f"{k}={str(v)[:50]}" for k, v in args.items()) if args else ""
    console.print(f"[dim]  [green]{tool_name}[/green]({arg_str})[/dim]")
    return True

  # UNSAFE mode: auto-approve everything, show brief notice
  if mode == "unsafe" and not plan_mode:
    badge = "[red]DESTRUCTIVE[/red] " if is_destructive else ""
    arg_str = ", ".join(f"{k}={str(v)[:40]}" for k, v in args.items()) if args else ""
    console.print(f"[dim]⚡ {badge}[green]{tool_name}[/green]({arg_str})[/dim]")
    return True

  # Build tool info display with description
  args_display = "\n".join([f" • {k}: {v}" for k, v in args.items()]) if args else " (no args)"
  destructive_badge = "[bold red] DESTRUCTIVE[/bold red] " if is_destructive else ""
  tool_info = f"{destructive_badge}[bold green]{tool_name}[/bold green]\n[dim]{description}[/dim]\n\n{args_display}"

  if plan_mode:
    # PLAN mode: skip all write tools
    console.print()
    console.print(
      Panel(
        tool_info,
        title="[bold dark_green]● PLAN MODE (Skipped)[/bold dark_green]",
        border_style="dark_green",
        padding=(1, 2)
      )
    )
    console.print("[dark_green]→ Skipped (PLAN mode: write tool)[/dark_green]")
    return 'skip'

  if mode == "safe":
    # SAFE mode: approval needed for write/destructive tools only
    # Read-only tools auto-approve silently
    read_only_tools = {"list_files", "read_file", "search_code", "cache_clear", "web_search"}
    if tool_name in read_only_tools:
      return True

    # Show approval for write/destructive tools
    console.print()
    title = f"{('[bold red] DESTRUCTIVE[/bold red] ' if is_destructive else '')}[bold green]Approve?[/bold green]"
    console.print(
      Panel(
        tool_info,
        title=title,
        border_style="red" if is_destructive else "green",
        padding=(1, 2)
      )
    )

    while True:
      response = get_tool_approval_key()
      if response == 't':
        return 'toggle'
      elif response == 'y':
        console.print("[dim]✓ Approved[/dim]")
        return True
      elif response == 'n':
        console.print("[dim]● Denied[/dim]")
        return False
      elif response == 'q':
        console.print("[red] Quit[/red]")
        raise KeyboardInterrupt
      elif response is None:
        continue

def call_model(provider_name, provider_model, key, messages, on_token=None):
  """
  Call the model with streaming. Returns (content, thinking, native_tool_calls).
  on_token(str) is called for each content token as it streams in.
  native_tool_calls is a list of {"tool": name, "args": {...}, "id": ...}
  """
  headers = {"Content-Type": "application/json"}
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
    if not check_ollama(url): return f"Error: Ollama server not found.", "", []
  else: url = config.get("url")

  max_tokens = int(config.get("max_tokens", 4096))

  if provider_name == "OpenRouter":
    headers["Authorization"] = f"Bearer {key}"
    headers["HTTP-Referer"] = "https://github.com/curren/OpenCLI"
    headers["X-Title"] = "OpenCLI"
    payload = {
      "model": provider_model, "messages": messages,
      "stream": True, "max_tokens": max_tokens,
      "tools": TOOL_SCHEMAS, "tool_choice": "auto"
    }
  elif provider_name == "NVIDIA":
    headers["Authorization"] = f"Bearer {key}"
    headers["Accept"] = "text/event-stream"
    payload = {
      "model": provider_model, "messages": messages,
      "temperature": 0.7, "stream": True, "max_tokens": max_tokens,
      "tools": TOOL_SCHEMAS, "tool_choice": "auto"
    }
  elif provider_name == "Anthropic":
    headers["x-api-key"] = key
    headers["anthropic-version"] = "2023-06-01"
    system_msg = ""
    anth_msgs = []
    for m in messages:
      if m["role"] == "system": system_msg += m["content"] + "\n"
      else: anth_msgs.append(m)
    payload = {
      "model": provider_model, "messages": anth_msgs,
      "system": system_msg.strip(), "max_tokens": max_tokens, "stream": True
    }
  elif provider_name == "Google":
    url = f"{url}{provider_model}:streamGenerateContent?key={key}"
    contents = []
    for m in messages:
      role = "user" if m["role"] == "user" else "model"
      contents.append({"role": role, "parts": [{"text": m["content"]}]})
    payload = {"contents": contents, "generationConfig": {"maxOutputTokens": max_tokens, "temperature": 1.0}}
  elif provider_name == "OpenAI":
    headers["Authorization"] = f"Bearer {key}"
    payload = {
      "model": provider_model, "messages": messages,
      "stream": True, "max_tokens": max_tokens,
      "tools": TOOL_SCHEMAS, "tool_choice": "auto"
    }
  elif provider_name == "Ollama":
    payload = {"model": provider_model, "messages": messages, "stream": True, "max_tokens": max_tokens}

  content_buffer = ""
  thinking_buffer = ""
  tool_calls_acc = {}  # index -> {"name": str, "args": str, "id": str}

  try:
    response = requests.post(url, headers=headers, json=payload, stream=True, timeout=60)

    if response.status_code == 429 or "rate_limit" in response.text.lower():
      for attempt in range(3):
        wait_time = (attempt + 1) * 3
        console.print(f"\nRate limited. Waiting {wait_time}s...")
        time.sleep(wait_time)
        response = requests.post(url, headers=headers, json=payload, stream=True, timeout=60)
        if response.status_code == 200: break
        if "rate_limit" not in response.text.lower() and response.status_code != 429: break
      else:
        return "Rate limited. Please wait and try again.", "", []

    if response.status_code != 200:
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
            return f"API Error ({retry.status_code})", "", []
        else:
          return f"API Error ({response.status_code})", "", []
      else:
        try:
          err_body = response.json()
          err_msg = err_body.get("error", {}).get("message", response.text[:200])
        except:
          err_msg = response.text[:200]
        return f"API Error ({response.status_code}): {err_msg}", "", []

    # Stream response
    for line in response.iter_lines():
      if check_stop(): break
      if not line: continue
      chunk = line.decode("utf-8").strip()

      if provider_name == "Anthropic":
        if chunk.startswith("data: "):
          try:
            data = json.loads(chunk[6:])
            if data.get("type") == "content_block_delta":
              token = data["delta"].get("text", "")
              if token:
                content_buffer += token
                if on_token: on_token(token)
          except: pass

      elif provider_name == "Google":
        try:
          data = json.loads(chunk)
          if "candidates" in data:
            token = data["candidates"][0].get("content", {}).get("parts", [{}])[0].get("text", "")
            if token:
              content_buffer += token
              if on_token: on_token(token)
        except: pass

      else:  # OpenAI-compatible (OpenRouter, OpenAI, NVIDIA, Ollama)
        if chunk.startswith("data: "): chunk = chunk[6:]
        if chunk == "[DONE]": break
        try:
          data = json.loads(chunk)
          if "choices" not in data: continue
          delta = data["choices"][0].get("delta", {})

          # Content token
          token = delta.get("content") or ""
          if token:
            content_buffer += token
            if on_token: on_token(token)

          # Reasoning/thinking token (some models like DeepSeek)
          thinking_token = delta.get("reasoning_content") or ""
          if thinking_token:
            thinking_buffer += thinking_token

          # Native tool calls (when tools param is accepted)
          for tc in delta.get("tool_calls", []):
            idx = tc.get("index", 0)
            if idx not in tool_calls_acc:
              tool_calls_acc[idx] = {
                "name": tc.get("function", {}).get("name", ""),
                "args": "",
                "id": tc.get("id", "")
              }
            else:
              fn = tc.get("function", {})
              if fn.get("name"): tool_calls_acc[idx]["name"] = fn["name"]
              if tc.get("id"): tool_calls_acc[idx]["id"] = tc["id"]
            tool_calls_acc[idx]["args"] += tc.get("function", {}).get("arguments", "")
        except: pass

    # Build native tool calls list
    native_tool_calls = []
    for idx in sorted(tool_calls_acc.keys()):
      tc = tool_calls_acc[idx]
      if not tc["name"]: continue
      try:
        args = json.loads(tc["args"]) if tc["args"] else {}
      except:
        args = {}
      native_tool_calls.append({"tool": tc["name"], "args": args, "id": tc.get("id", "")})

    return content_buffer, thinking_buffer, native_tool_calls

  except Exception as e:
    return f"Error: {str(e)}", "", []


def run_onboarding():
  console.clear()
  welcome_text = Text.assemble(
    (r"""
  _ ____ _____ _  _ ____ _   ___ 
 / \| _ \| ___| \ | |/ ___| |  |_ _|
 | | | |_) | _| | \| | |  | |  | | 
 | | | __/| |___| |\ | |___| |___ | | 
 \_/|_|  |_____|_| \_|\____|_____|___|
""", "green"),
    ("\n\nWelcome to OpenCLI!\n", "bold white")
  )
  console.print(Align.center(Panel(welcome_text, box=box.ROUNDED, border_style="green")))
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
  
  # Clear screen on start for consistent UI
  clear_screen()
  
  if "--version" in sys.argv:
    print("OpenCLI v0.1.0")
    return
  if "--settings" in sys.argv:
    interactive_settings_menu()
    config = load_config()
    banner(config.get("mode", "safe"), config.get("model"))
  if not CONFIG_FILE.exists(): run_onboarding()
  
  messages = [{"role": "system", "content": build_system_prompt(config.get("mode", "safe"))}]
  # Unique session per process
  PROCESS_SESSION_ID = str(uuid.uuid4())
  session_id = PROCESS_SESSION_ID
  resumed = False
  _paste_count = 0  # increments each time a multi-line paste is detected

  # Track modified files for session
  modified_files = set()

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
    console.print(f"[bold green]Session:[/bold green] {total_msgs} messages, {tool_count} tool calls")

    conversational = [m for m in messages if m["role"] in ("user", "assistant")]
    for m in conversational[-6:]:
      if m["role"] == "assistant":
        render_message(output=m["content"][:800])
      elif m["role"] == "user":
        console.print(f"[bold green]You[/bold green] › {m['content']}")

  if not resumed:
    cwd = os.getcwd()
    # Gather environment context
    env_lines = [f"WORKING_DIRECTORY: {cwd}"]

    # Git status
    try:
      branch = subprocess.check_output(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=cwd, stderr=subprocess.DEVNULL, text=True
      ).strip()
      status = subprocess.check_output(
        ["git", "status", "--short"],
        cwd=cwd, stderr=subprocess.DEVNULL, text=True
      ).strip()
      env_lines.append(f"GIT_BRANCH: {branch}")
      if status:
        env_lines.append(f"GIT_STATUS (modified/untracked files):\n{status}")
    except:
      pass

    # Detect project type
    project_hints = []
    if Path(cwd, "package.json").exists():
      project_hints.append("Node.js/JavaScript project (package.json found)")
    if Path(cwd, "pyproject.toml").exists() or Path(cwd, "setup.py").exists():
      project_hints.append("Python project (pyproject.toml/setup.py found)")
    if Path(cwd, "Cargo.toml").exists():
      project_hints.append("Rust project (Cargo.toml found)")
    if Path(cwd, "go.mod").exists():
      project_hints.append("Go project (go.mod found)")
    if project_hints:
      env_lines.append("PROJECT_TYPE: " + ", ".join(project_hints))

    env_lines.append(
      "\nYou may use list_files, read_file, and search_code to explore the project."
      "\nDo NOT assume file contents without reading them first."
    )

    messages.append({
      "role": "system",
      "content": "\n".join(env_lines)
    })
  
  mode = config.get("mode", "safe")
  if not resumed:
    banner(mode, config.get("model"))

  # Get initial model for context calculation
  p_model = config.get("model")

  # Calculate initial context %
  _, tokens_after, context_limit = compact_context(messages, p_model)
  context_pct = int((tokens_after / context_limit) * 100) if context_limit > 0 else 0

  # Message queue — messages typed while the model is busy get queued
  _msg_queue = collections.deque()
  _next_queued = None  # set after agent loop to replay a queued message

  # Create prompt_toolkit session (input box + bottom toolbar with mode)
  pt_session = create_prompt_session(msg_queue=_msg_queue)

  while True:
    try:
      config = load_config()
      p_name = config.get("provider", "OpenRouter")
      p_model = config.get("model")
      key_m = {"NVIDIA":"nvidia_key","OpenRouter":"openrouter_key","Anthropic":"anthropic_key","OpenAI":"openai_key","Google":"google_key"}
      key = config.get(key_m.get(p_name))
      if not key and p_name != "Ollama":
        if Prompt.ask("No API key. Setting?", default="y") == "y":
          interactive_settings_menu()
          continue
        return

      # Update context % before each input
      _, tokens_after, context_limit = compact_context(messages, p_model)
      context_pct = int((tokens_after / context_limit) * 100) if context_limit > 0 else 0

      # Update toolbar state
      _ui_state["mode"] = mode
      _ui_state["model"] = p_model
      _ui_state["ctx_pct"] = context_pct

      # Dequeue a previously queued message, or prompt the user
      if _next_queued is not None:
        user_input = _next_queued
        _next_queued = None
      else:
        # Prompt user — bottom toolbar shows mode + Shift+Tab hint
        # patch_stdout ensures Rich output above doesn't corrupt the input line
        try:
          with patch_stdout():
            user_input = pt_session.prompt(HTML('<b><ansigreen>You › </ansigreen></b>'))
        except KeyboardInterrupt:
          session_id = save_history(messages, session_id)
          console.print("\n[green]Goodbye![/green]")
          sys.exit(0)
        except EOFError:
          session_id = save_history(messages, session_id)
          console.print("\n[green]Goodbye![/green]")
          sys.exit(0)

      # Pick up any mode change that happened via Shift+Tab inside the prompt
      mode = _ui_state["mode"]

      # Handle empty input
      if not user_input.strip():
        console.print()
        continue

      # Shift+Tab is handled inside the prompt session key binding now — skip if it leaks
      if user_input in ('\x1b[Z', '\x1b[z'):
        continue

      # Handle quit command
      if user_input.strip().lower() in ['exit', 'quit', 'q']:
        session_id = save_history(messages, session_id)
        console.print("[green]Goodbye![/green]")
        sys.exit(0)

      # Handle mode cycling via /mode command
      if user_input.strip() == "/mode":
        mode = cycle_mode(mode)
        config["mode"] = mode
        _ui_state["mode"] = mode
        save_config(config)
        messages[0] = {"role": "system", "content": build_system_prompt(config.get("mode", "safe"))}
        console.print(f"[green]Mode → {mode.upper()}[/green]")
        continue

      if user_input.strip() in ("/", "/provider"):
        interactive_settings_menu()
        config = load_config()
        mode = config.get("mode", "safe")
        _ui_state["mode"] = mode
        messages[0] = {"role": "system", "content": build_system_prompt(mode)}
        # Reprint recent conversation so context is visible after settings
        console.print()
        console.print("[dim]─── back to conversation ───[/dim]")
        conversational = [m for m in messages[1:] if m["role"] in ("user", "assistant")]
        for m in conversational[-4:]:
          if m["role"] == "assistant":
            content = m.get("content") or ""
            if content:
              render_message(output=content[:600])
          elif m["role"] == "user":
            content = m.get("content") or ""
            if content:
              console.print(f"[bold green]You[/bold green] › {content}")
        console.print()
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
          console.print(f"[bold green]Session Summary[/bold green]")
          console.print(f"Messages: {total_msgs} | Tool Calls: {tool_count} | Working Dir: {os.getcwd()}")
          conversational = [m for m in messages if m["role"] in ("user", "assistant")]
          for m in conversational[-6:]:
            if m["role"] == "assistant":
              content = m.get("content") or ""
              if content:
                render_message(output=content[:800])
            elif m["role"] == "user":
              content = m.get("content") or ""
              if content:
                console.print(f"[bold green]You[/bold green] › {content}")
        continue

      if user_input.strip() == "/clear":
        console.print("\n[bold red]Are you sure you want to clear the session?[/bold red]")
        confirm = Prompt.ask("[red]Type 'y' to confirm, 'n' to cancel:[/red]", default="n")
        if confirm.lower() == 'y':
          messages = [{"role": "system", "content": build_system_prompt(config.get("mode", "safe"))}]
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

      # Don't send empty input
      if not user_input.strip():
        continue

      # Paste detection: 4+ lines = treat as pasted block
      paste_lines = user_input.split("\n")
      if len(paste_lines) >= 4:
        _paste_count += 1
        label = f"Pasted text {_paste_count}"
        line_word = "line" if len(paste_lines) == 1 else "lines"
        preview = "\n".join(paste_lines[:3])
        console.print(
          Panel(
            f"[dim]{preview}\n...[/dim]",
            title=f"[green]{label}[/green]  [dim]{len(paste_lines)} {line_word}[/dim]",
            border_style="green",
            padding=(0, 1),
          )
        )
        # Full content still sent to model, wrapped with label
        message_content = f"[{label} — {len(paste_lines)} lines]\n{user_input}"
      else:
        message_content = user_input

      messages.append({"role": "user", "content": message_content})
      _agent_steps = 0
      session_id = save_history(messages, session_id)
      CURRENT_MESSAGES = messages
      CURRENT_SESSION_ID = session_id
      messages, tokens, limit = compact_context(messages, p_model)

      # Print status bar so mode is always visible during AI processing
      _print_status_bar(mode, p_model, context_pct)

      # Start ESC watcher — ESC at any point cancels model + tool loop
      _esc_event.clear()
      _esc_stop, _esc_thread = _start_esc_watcher()

      while True:
        # Track timing for model response
        start_time = time_module.time()

        # Rate limit spacing
        global _last_request_time
        if '_last_request_time' in globals():
          elapsed = time_module.time() - _last_request_time
          if elapsed < 1.5:
            time.sleep(1.5 - elapsed)
        _last_request_time = time_module.time()

        # Simple thinking indicator using ANSI codes (doesn't break Rich)
        _spinner_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        _spinner_idx = [0]
        _spinner_done = [False]
        
        # Initialize spinner event BEFORE try block
        import threading
        _stop_spinner = threading.Event()
        
        def _update_spinner():
          frame = _spinner_frames[_spinner_idx[0] % len(_spinner_frames)]
          _spinner_idx[0] += 1
          sys.stdout.write(f"\r  {frame} Thinking... ")
          sys.stdout.flush()
        
        _stream_buffer = []
        
        def _on_token(token):
          if _esc_event.is_set():
            raise KeyboardInterrupt
          if not _spinner_done[0]:
            _spinner_done[0] = True
            # Clear spinner line
            sys.stdout.write("\r" + " " * 30 + "\r")
            sys.stdout.flush()
          _stream_buffer.append(token)

        try:
          # Show spinner while waiting for first token
          def _spin():
            while not _stop_spinner.is_set():
              _update_spinner()
              time.sleep(0.1)
            # Clear spinner when done
            sys.stdout.write("\r" + " " * 30 + "\r")
            sys.stdout.flush()
          
          _spin_thread = threading.Thread(target=_spin, daemon=True)
          _spin_thread.start()
          
          result = call_model(p_name, p_model, key, messages, on_token=_on_token)
        except KeyboardInterrupt:
          _stop_spinner.set()
          _esc_event.set()
          sys.stdout.write("\n")
          sys.stdout.flush()
          clear_screen()
          console.print("[dim]⏹ Cancelled[/dim]")
          break
        
        _stop_spinner.set()
        sys.stdout.write("\n")
        sys.stdout.flush()

        console.print()  # Newline after streaming

        # Erase raw streamed output and reprint as Rich Markdown
        streamed_text = "".join(_stream_buffer).strip()
        
        # Unpack 3-tuple to get thinking first
        if isinstance(result, tuple) and len(result) == 3:
          content, thinking_from_api, native_tool_calls = result
          reply = content
        elif isinstance(result, tuple):
          content, thinking_from_api = result
          native_tool_calls = []
          reply = content
        else:
          thinking_from_api = ""
          native_tool_calls = []
          reply = str(result)

        # Show thinking BEFORE the response (from reasoning models)
        if thinking_from_api and thinking_from_api.strip():
          think_preview = thinking_from_api.strip()[:300]
          console.print(f"[dim]● Thinking:[/dim] {think_preview}{'...' if len(thinking_from_api) > 300 else ''}")

        # Reprint streamed output as Rich Markdown (clean formatting)
        if streamed_text:
          console.print(Markdown(streamed_text))

        # Handle error responses
        if reply.startswith("Error") or reply.startswith("API Error"):
          console.print(f"[red]{reply}[/red]")
          break

        response_time = time_module.time() - start_time

        # Build thinking string for tool approval display
        thinking_str = ""
        if thinking_from_api:
          thinking_str = thinking_from_api[:150]
        else:
          think_matches = re.findall(r"<think>(.*?)</think>", reply, re.DOTALL)
          if think_matches:
            thinking_str = " ".join(t.strip()[:100] for t in think_matches[:2])

        # === TOOL CALL DETECTION ===
        # Prefer native tool calls (from tools param), fall back to text JSON
        tool_calls = []
        if native_tool_calls:
          tool_calls = native_tool_calls
        else:
          # Try to parse tool calls from text content
          model_output = reply
          # Strip <think> blocks before parsing
          model_output_clean = re.sub(r"<think>.*?</think>", "", model_output, flags=re.DOTALL).strip()
          _, tool_data = extract_json(model_output_clean)
          if not tool_data:
            json_block_match = re.search(r'\{[\s\S]*\}', model_output_clean)
            if json_block_match:
              try:
                possible = json.loads(json_block_match.group(0))
                if isinstance(possible, dict) and "tool" in possible:
                  tool_data = possible
              except:
                pass
          if tool_data:
            tool_calls = tool_data if isinstance(tool_data, list) else [tool_data]

        if tool_calls:
          # For native tool calls, store assistant turn with tool_calls array (required by OpenAI API)
          if native_tool_calls and p_name not in ("Anthropic", "Google"):
            api_tool_calls = []
            for call in tool_calls:
              api_tool_calls.append({
                "id": call.get("id") or f"call_{call.get('tool')}",
                "type": "function",
                "function": {"name": call.get("tool"), "arguments": json.dumps(call.get("args", {}))}
              })
            messages.append({"role": "assistant", "content": reply or None, "tool_calls": api_tool_calls})
          elif reply and reply.strip() and not native_tool_calls:
            # Text-based: store assistant's raw text (includes the JSON)
            messages.append({"role": "assistant", "content": reply})

          for call in tool_calls:
            tool_name = call.get("tool")
            args = call.get("args", {})
            tool_call_id = call.get("id") or call.get("tool_call_id")  # Get tool_call_id for API

            if tool_name not in TOOLS:
              err_msg = f"Error: Tool {tool_name} not found."
              console.print(Panel(err_msg, style="bold red"))
              messages.append({"role": "assistant", "content": err_msg})
              continue

            approval = get_tool_approval(
              tool_name,
              args,
              mode,
              plan_mode=(mode == "plan"),
            )

            if approval == 'toggle':
              mode = cycle_mode(mode)
              config["mode"] = mode
              _ui_state["mode"] = mode
              save_config(config)
              messages[0] = {"role": "system", "content": build_system_prompt(config.get("mode", "safe"))}
              console.print(f"\n[green]Mode → {mode.upper()}[/green]")
              continue

            if approval == 'skip':
              continue

            if not approval:
              break

            # ESC check before executing tool
            if _esc_event.is_set():
              console.print("[dim]⏹ Cancelled[/dim]")
              break

            # Show running tool indicator
            with console.status(f"[dim]{tool_name}...[/dim]", spinner="dots"):
              result, display_output, tool_result_text = show_tool_execution(
                tool_name,
                args,
                TOOLS[tool_name],
                plan_mode=(mode == "plan"),
                modified_files=modified_files
              )

            # ESC check after tool execution - stop整个循环 if cancelled
            if _esc_event.is_set():
              console.print("[dim]⏹ Cancelled[/dim]")
              break

            # --- Error + Auto-Retry Handling ---
            if isinstance(result, str) and result.startswith("Error"):
              if is_rate_limit_error(result):
                console.print("[red]Rate limited. Waiting 5s and retrying...[/red]")
                time.sleep(5)
                continue  # auto retry same tool

              action = handle_tool_error(tool_name, result, messages)

              if action == 'retry':
                continue
              if action == 'skip':
                continue
              if action == 'quit':
                break

            if isinstance(display_output, list):
              for item in display_output:
                console.print(item)  # items are Rich Panel objects
            elif display_output:
              console.print(display_output)

            if tool_result_text:
              truncated = truncate_tool_output(tool_result_text)

              # Store tool result in the conversation
              if p_name == "Anthropic":
                # Anthropic uses user-role tool_result blocks (simplified: append as user message)
                messages.append({
                  "role": "user",
                  "content": f"Tool result [{tool_name}]:\n{truncated}"
                })
              else:
                msg = {
                  "role": "tool",
                  "name": tool_name,
                  "content": truncated
                }
                if tool_call_id:
                  msg["tool_call_id"] = tool_call_id
                messages.append(msg)

            # ESC check after storing tool result - stop整个循环 if cancelled
            if _esc_event.is_set():
              console.print("[dim]⏹ Cancelled[/dim]")
              break

            # Delay between batched tool calls to prevent rate limiting
            if len(tool_calls) > 1:
              time.sleep(TOOL_BATCH_DELAY)

            # ESC check after batch delay
            if _esc_event.is_set():
              console.print("[dim]⏹ Cancelled[/dim]")
              break

          session_id = save_history(messages, session_id)
          CURRENT_MESSAGES = messages
          CURRENT_SESSION_ID = session_id

          messages, tokens, limit = compact_context(messages, p_model)

          # In PLAN mode, auto-continue to build context until plan is ready
          if mode == "plan" and _agent_steps < 8:
            console.print("[dim]Building context...[/dim]")
            time.sleep(1)
            continue  # Keep planning

          continue  # CRITICAL: never render JSON as assistant output

        # No tool calls — treat as normal assistant text response
        # Content was already streamed live; now store and show timing
        _agent_steps += 1

        if _agent_steps > MAX_AGENT_STEPS:
          console.print(Panel(
            f"[red]Max steps ({MAX_AGENT_STEPS}) reached.[/red]",
            title="Step Limit", border_style="red"
          ))
          break

        # Clean up <think> tags from stored content
        cleaned_output = re.sub(r"<think>.*?</think>", "", reply, flags=re.DOTALL).strip()

        if cleaned_output and cleaned_output != "TASK_COMPLETE":
          output_tokens = len(cleaned_output) // 4
          input_tokens = count_tokens(messages[:-1]) if len(messages) > 1 else 0
          timing_info = f"[dim]{format_time(response_time)} • {format_tokens(output_tokens)} out • {format_tokens(input_tokens)} in[/dim]"
          console.print(Align.right(timing_info))

          messages.append({"role": "assistant", "content": cleaned_output})
          session_id = save_history(messages, session_id)
          CURRENT_MESSAGES = messages
          CURRENT_SESSION_ID = session_id

        break

      # Stop ESC watcher now that agent loop has exited
      _esc_stop.set()
      _esc_event.clear()

      # If queued messages exist, inject the next one so the outer loop picks it up
      if _msg_queue:
        count = len(_msg_queue)
        console.print(f"[dim]▶ {count} queued message{'s' if count > 1 else ''} — processing next...[/dim]")
        # Push it as user_input so the top of the outer loop processes it naturally
        # by pre-populating and skipping the prompt
        _next_queued = _msg_queue.popleft()
      else:
        _next_queued = None

    except KeyboardInterrupt:
      session_id = save_history(messages, session_id)
      CURRENT_MESSAGES = messages
      CURRENT_SESSION_ID = session_id
      console.print("\n[green]Goodbye![/green]")
      sys.exit(0)

if __name__ == "__main__":
  main()
