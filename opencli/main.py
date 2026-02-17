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
from filelock import FileLock
import uuid
import time as time_module

try:
  from prompt_toolkit import PromptSession
  from prompt_toolkit.enums import EditingMode
  HAS_PROMPT_TOOLKIT = True
except ImportError:
  HAS_PROMPT_TOOLKIT = False

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
  "z-ai/glm-4.5-air:free": 128000
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
    content = m.get("content", "")
    # Heuristic: check if content looks like code/JSON
    if any(x in content for x in ['{', '}', 'def ', 'class ', 'function', 'const ']):
      # Code-like content
      total += len(content) // 3
    else:
      # Natural language
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
  sessions = [s for s in sessions if now - s.get("timestamp", 0) < 86400]

  sessions.sort(key=lambda x: x.get("timestamp", 0), reverse=True)

  return {"sessions": sessions[:10]}

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

config = load_config()


def get_theme_color(color_name):
  theme = config.get("theme", "dark")
  colors = {
    "dark": {
      "primary": "green",
      "secondary": "dark_green",
      "warning": "red",
      "error": "red",
      "text": "white",
      "dim": "dim"
    },
    "light": {
      "primary": "green",
      "secondary": "dark_green",
      "warning": "red",
      "error": "red",
      "text": "black",
      "dim": "dim"
    }
  }
  return colors.get(theme, colors["dark"]).get(color_name, "white")

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


def interactive_settings_menu():
  console.clear()
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

TOOL_METADATA = {
  "list_files": {
    "description": "List files in a directory",
    "destructive": False
  },
  "read_file": {
    "description": "Read file contents (safe, read-only)",
    "destructive": False
  },
  "write_file": {
    "description": "Create or overwrite a file",
    "destructive": True
  },
  "replace_text": {
    "description": "Replace text in a file",
    "destructive": True
  },
  "search_code": {
    "description": "Search code with grep",
    "destructive": False
  },
  "run_shell": {
    "description": "Execute shell command",
    "destructive": True
  }
}

def truncate_tool_output(output, limit=2000):
  """
  Truncate tool output before storing in conversation history
  to prevent token explosion.
  """
  text = str(output)
  if len(text) <= limit:
    return text
  return text[:limit] + "\n\n... (truncated for context safety)"

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
  if plan_mode:
    return None, "[dark_green]→ PLAN mode: tool skipped (learning & analyzing)[/dark_green]", None

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

    # Show before/after diff
    diff_content = f"""[bold green]{file_path}[/bold green]

[bold red]REMOVED[/bold red]
[red]{old_text}[/red]

[bold green]ADDED[/bold green]
[green]{new_text}[/green]"""

    display_panels.append(
      Panel(
        diff_content,
        title="[green] Text Replaced[/green]",
        border_style="green",
        padding=(1, 2)
      )
    )

  elif tool_name == "write_file":
    file_path = args.get("path", "?")
    content = args.get("content", "")
    lines = len(content.splitlines())

    # Track modified file
    if modified_files is not None:
      modified_files.add(file_path)

    # Show file creation summary
    preview = content[:200] + ("..." if len(content) > 200 else "")
    display_panels.append(
      Panel(
        f"[green] File created[/green]\n[dim]{file_path}[/dim]\n[dim]{lines} lines[/dim]\n\n[dim]{preview}[/dim]",
        title="[green] File Written[/green]",
        border_style="green",
        padding=(1, 2)
      )
    )

  # === STANDARD OUTPUT ===
  else:
    # For read operations, show the output
    if result_text.strip() and result_text not in ["(Command executed with no output)", "No files found."]:
      truncated = truncate_tool_output(result_text, limit=1000)

      # Format based on tool type
      if tool_name == "list_files":
        title = " Files Found"
        style = "green"
      elif tool_name == "read_file":
        title = " File Contents"
        style = "green"
      elif tool_name == "search_code":
        title = " Search Results"
        style = "green"
      else:
        title = f"{tool_name} Output"
        style = "green"

      display_panels.append(
        Panel(
          f"[dim]{truncated}[/dim]",
          title=f"[{style}]{title}[/{style}]",
          border_style=style,
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


def build_system_prompt():
  """
  Build full system prompt dynamically:
  - Always reload OPENCLI.md (session-level behavior)
  - Always reload TOOLS.md (request-level tool awareness)
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
- Do NOT include any text before or after the JSON.
- The format MUST be exactly:

{"tool": "tool_name", "args": { ... }}

After a tool result is returned, you may respond normally.

Never simulate tool execution.
Never describe a tool call — only emit valid JSON.

=== EXECUTION MODES ===
The CLI has three modes:
- SAFE: You will call tools, but the user must approve each one
- UNSAFE: You will call tools and they auto-execute
- PLAN: You should analyze code and PLAN your approach, but NOT call tools (read-only mode)

In PLAN mode, inspect files and reason about what needs to be done, but do not attempt tool calls.

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

  return f"{base}\n\n{tools_text}\n\n{strict_rules}"

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


def check_ollama(url):
  try:
    base_url = url.split("/v1")[0]
    response = requests.get(base_url, timeout=2)
    return response.status_code == 200
  except: return False

def get_tool_approval_key():
  """Get single-key input: y/n/t (toggle mode) without requiring Enter."""
  fd = sys.stdin.fileno()
  old_settings = termios.tcgetattr(fd)
  try:
    tty.setraw(fd)
    ch = sys.stdin.read(1).lower()

    # Check for Shift+Tab (ESC [ Z sequence)
    if ch == '\x1b': # ESC
      try:
        next_ch = sys.stdin.read(1)
        if next_ch == '[':
          third_ch = sys.stdin.read(1)
          if third_ch == 'Z': # Shift+Tab
            return 't'
      except:
        pass

    # Single key presses
    if ch in ['y', 'n']:
      return ch
    elif ch == '\r' or ch == '\n':
      return 'n' # Default to deny
    elif ch == '\x03': # Ctrl+C
      raise KeyboardInterrupt
    else:
      return None # Invalid key
  finally:
    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

def get_tool_approval(tool_name, args, mode, plan_mode=False):
  """
  Robust tool approval with metadata, single-key input, and mode toggle.
  Returns: True (approve), False (deny), 'toggle' (switch modes), or 'skip' (plan mode).
  """
  metadata = TOOL_METADATA.get(tool_name, {})
  description = metadata.get("description", "Unknown tool")
  is_destructive = metadata.get("destructive", False)

  # Build tool info display with description
  args_display = "\n".join([f" • {k}: {v}" for k, v in args.items()]) if args else " (no args)"
  destructive_badge = "[bold red] DESTRUCTIVE[/bold red] " if is_destructive else ""
  tool_info = f"{destructive_badge}[bold green]{tool_name}[/bold green]\n[dim]{description}[/dim]\n\n{args_display}"

  if plan_mode:
    # PLAN mode: never execute, just show what would happen
    console.print()
    console.print(
      Panel(
        tool_info,
        title="[bold dark_green] PLAN MODE (Read-Only)[/bold dark_green]",
        border_style="green",
        padding=(1, 2)
      )
    )
    console.print("[dark_green]→ Tool execution skipped (PLAN mode: learning & planning only)[/dark_green]")
    return 'skip'

  if mode == "safe":
    # SAFE mode: ALWAYS show approval with metadata
    console.print()
    title = f"{'[bold red] DESTRUCTIVE[/bold red] ' if is_destructive else ''}[bold green]Approve?[/bold green]"
    console.print(
      Panel(
        tool_info,
        title=title,
        border_style="red" if is_destructive else "green",
        padding=(1, 2)
      )
    )

    # Show mode indicator and instructions
    mode_display = get_mode_indicator(mode)
    console.print(f"{mode_display} [dim]y approve • n deny • Shift+Tab cycle mode[/dim]")

    # Get response (supports mode toggle)
    while True:
      response = get_tool_approval_key()
      if response == 't':
        return 'toggle'
      elif response == 'y':
        console.print("[green] Approved[/green]")
        return True
      elif response == 'n':
        console.print(
          Panel(
            "[green] Denied[/green]",
            title="[green]Cancelled[/green]",
            border_style="green"
          )
        )
        return False
      # If None (invalid key), loop and ask again
  else:
    # UNSAFE mode: show but don't ask
    console.print()
    console.print(
      Panel(
        tool_info,
        title="[bold red] Auto-Executing (UNSAFE)[/bold red]",
        border_style="red",
        style="dim"
      )
    )
    instructions = "[dim]→ [bold]Shift+Tab[/bold] to toggle to SAFE mode[/dim]"
    console.print(instructions, justify="center")
    return True

def call_model(provider_name, provider_model, key, messages):
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

      try:
        parsed = json.loads(final_text)
        normalized = normalize_openai_tool_call(parsed)
        if normalized:
          return json.dumps(normalized)
      except:
        pass

      if "<|tool_calls_section_begin|>" in final_text:
        try:
          tool_calls = []

          blocks = re.findall(
            r'<\|tool_call_begin\|>.*?functions\.(\w+).*?(\{.*?\})',
            final_text,
            re.DOTALL
          )

          for name, args_json in blocks:
            try:
              args = json.loads(args_json)
              tool_calls.append({"tool": name, "args": args})
            except Exception:
              continue

          if tool_calls:
            # Return ALL tool calls as a batch (true multi-tool execution)
            return json.dumps(tool_calls)

        except Exception:
          pass

      if thinking_text and full_text:
        return f" Thinking:\n{thinking_text}\n\n{full_text}"
      return thinking_text or full_text

    return "[dim](Empty response)[/dim]"
  except Exception as e: return f"[red]Error: {str(e)}[/red]"

def get_multiline_input(prompt="› "):
  """Input handler: Enter to send, Shift+Tab to cycle mode."""
  console.print(f"[bold {get_theme_color('text')}]{prompt}[/bold {get_theme_color('text')}]", end="", flush=True)

  fd = sys.stdin.fileno()
  old_settings = termios.tcgetattr(fd)
  try:
    tty.setraw(fd)
    text = ""

    while True:
      ch = sys.stdin.read(1)

      # Check for Shift+Tab (ESC [ Z sequence)
      if ch == '\x1b':  # ESC
        try:
          next_ch = sys.stdin.read(1)
          if next_ch == '[':
            third_ch = sys.stdin.read(1)
            if third_ch == 'Z':  # Shift+Tab
              termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
              return "MODE_CYCLE"
        except:
          pass

      # Enter sends
      if ch == '\r' or ch == '\n':
        print()  # New line after input
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return text

      # Ctrl+C to cancel
      if ch == '\x03':
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ""

      # Backspace
      if ch == '\x7f':  # Backspace
        if text:
          text = text[:-1]
          sys.stdout.write('\b \b')  # Delete character on screen
          sys.stdout.flush()

      # Regular character
      elif ord(ch) >= 32:  # Printable character
        text += ch
        sys.stdout.write(ch)
        sys.stdout.flush()

  except KeyboardInterrupt:
    return ""
  finally:
    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

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
  if "--version" in sys.argv:
    print("OpenCLI v0.1.0")
    return
  if "--settings" in sys.argv:
    interactive_settings_menu()
    config = load_config()
    banner(config.get("mode", "safe"), config.get("model"))
  if not CONFIG_FILE.exists(): run_onboarding()
  
  messages = [{"role": "system", "content": build_system_prompt()}]
  # Unique session per process
  PROCESS_SESSION_ID = str(uuid.uuid4())
  session_id = PROCESS_SESSION_ID
  resumed = False

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
    console.print(Panel(
      f"[bold]Session Summary[/bold]\n"
      f"Messages: {total_msgs}\n"
      f"Tool Calls: {tool_count}\n"
      f"Working Dir: {os.getcwd()}",
      border_style="green"
    ))

    conversational = [m for m in messages if m["role"] in ("user", "assistant")]
    for m in conversational[-6:]:
      if m["role"] == "assistant":
        console.print(Panel(m["content"][:800], style="green", title="OpenCLI "))
      elif m["role"] == "user":
        console.print(f"[bold {get_theme_color('text')}]You[/bold {get_theme_color('text')}] › {m['content']}")

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

      # Display mode indicator on left, commands on right
      current_mode = config.get("mode", "safe")
      mode_text = get_mode_indicator(current_mode)
      console.print(
        f"\n[bold {text_c}]You[/bold {text_c}] "
        "[dim](/ → settings • /resume → load chat • /clear → reset • exit → quit)[/dim]"
      )
      console.print(f"{mode_text} [dim]Enter=send • Shift+Tab=cycle mode[/dim]")

      # Get input with mode cycling support
      while True:
        user_input = get_multiline_input("› ")

        # Handle mode cycling via Shift+Tab
        if user_input == "MODE_CYCLE":
          mode = cycle_mode(mode)
          config["mode"] = mode
          save_config(config)
          mode_text = get_mode_indicator(mode)
          console.print(f"\n{mode_text} Mode changed\n")
          # Show mode indicator again and re-prompt
          console.print(f"{mode_text} [dim]Enter=send • Shift+Tab=cycle mode[/dim]")
          continue
        break

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
            border_style="green"
          ))

          conversational = [m for m in messages if m["role"] in ("user", "assistant")]
          for m in conversational[-6:]:
            if m["role"] == "assistant":
              console.print(Panel(m["content"][:800], style="green", title="OpenCLI "))
            elif m["role"] == "user":
              console.print(f"[bold {get_theme_color('text')}]You[/bold {get_theme_color('text')}] › {m['content']}")
        continue
      if user_input.strip() == "/clear":
        messages = [{"role": "system", "content": build_system_prompt()}]
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
      _agent_steps = 0
      session_id = save_history(messages, session_id)
      CURRENT_MESSAGES = messages
      CURRENT_SESSION_ID = session_id
      messages, tokens, limit = compact_context(messages, p_model)
      console.print(Align.right(Text(f"Context: {(tokens*100)//limit}%", style="dim")))
      while True:
        # Track timing for model response
        start_time = time_module.time()
        reply = call_model(p_name, p_model, key, messages)
        response_time = time_module.time() - start_time
        # Extract embedded tool JSON even if model included reasoning text
        visible_text, extracted_tool = extract_json(reply)
        if extracted_tool:
          tool_calls = [extracted_tool]
          reasoning = visible_text
        else:
          tool_calls = None
          reasoning = reply

        # --- Agent step guard (prevents infinite self-loops) ---
        if "_agent_steps" not in locals():
          _agent_steps = 0
        _agent_steps += 1
        MAX_AGENT_STEPS = 20  # Higher limit - keep trying until task completes

        # Show current step progress
        if _agent_steps > 1:
          console.print(Align.right(Text(f"[dim]Step {_agent_steps}/{MAX_AGENT_STEPS}[/dim]", style="dim")))

        # Check if we've exceeded max steps
        if _agent_steps > MAX_AGENT_STEPS:
          console.print(Panel(
            f"[red]Max steps ({MAX_AGENT_STEPS}) reached. Task may be incomplete.[/red]\n\nThe agent has made {_agent_steps} attempts. You can continue by providing feedback or asking the agent to try a different approach.",
            title="Step Limit",
            border_style="red"
          ))
          break

        # reply_stripped = reply.strip()

        executed_tools = False

        if reasoning.strip():

          raw_output = reasoning

          think_matches = re.findall(r"<think>(.*?)</think>", raw_output, re.DOTALL)
          cleaned_output = re.sub(r"<think>.*?</think>", "", raw_output, flags=re.DOTALL).strip()

          for think_block in think_matches:
            think_text = think_block.strip()
            if think_text:
              console.print(
                Panel(
                  think_text,
                  border_style="green",
                  title="Thinking",
                  style="dim"
                )
              )

          if cleaned_output.startswith(" Thinking:"):
            parts = cleaned_output.split("\n\n", 1)
            thinking_part = parts[0].replace(" Thinking:\n", "").strip()
            cleaned_output = parts[1].strip() if len(parts) > 1 else ""

            if thinking_part:
              console.print(
                Panel(
                  thinking_part,
                  border_style="green",
                  title="Thinking",
                  style="dim"
                )
              )

          visible_output = cleaned_output.strip()

          if visible_output and visible_output.strip() != "TASK_COMPLETE":
            console.print(
              Panel(
                Markdown(visible_output),
                border_style="green",
                title="OpenCLI "
              )
            )

            # Show timing and token info
            output_tokens = len(visible_output) // 4
            input_tokens = count_tokens(messages[:-1]) if len(messages) > 1 else 0
            timing_info = f"[dim] {format_time(response_time)} • "
            timing_info += f" {format_tokens(output_tokens)} • "
            timing_info += f" {format_tokens(input_tokens)} tokens[/dim]"
            console.print(Align.right(timing_info))

            # Store ONLY final visible output
            messages.append({"role": "assistant", "content": visible_output})
            session_id = save_history(messages, session_id)
            CURRENT_MESSAGES = messages
            CURRENT_SESSION_ID = session_id

        if tool_calls:
          for tool_call in tool_calls:
            t_name = tool_call["tool"]
            args = tool_call.get("args", {})

            if t_name not in TOOLS:
              err_msg = f"Error: Tool {t_name} not found."
              console.print(Panel(err_msg, style="bold red"))
              messages.append({"role": "assistant", "content": err_msg})
              continue

            # Get approval (handles SAFE, UNSAFE, PLAN modes, and mode toggling)
            approval = get_tool_approval(
              t_name, args, mode,
              plan_mode=(mode == "plan")
            )

            # Handle mode toggle (Shift+Tab) - cycle through all modes
            if approval == 'toggle':
              mode = cycle_mode(mode)
              config["mode"] = mode
              save_config(config)
              mode_display = get_mode_indicator(mode)
              console.print(f"\n[green]Mode switched to: {mode_display}[/green]\n")
              banner(mode, config.get("model"))
              continue

            # Handle PLAN mode (skip execution)
            if approval == 'skip':
              executed_tools = False
              continue

            # Skip if denied
            if not approval:
              continue

            executed_tools = True

            # Execute tool with Claude Code-like UX
            result, display_output, tool_result_text = show_tool_execution(
              t_name, args, TOOLS[t_name],
              plan_mode=(mode == "plan"),
              modified_files=modified_files
            )

            # Display result(s) in chat
            if isinstance(display_output, list):
              for item in display_output:
                console.print(item)
            elif display_output:
              console.print(display_output)

            # Store tool result in messages for context (including errors)
            if tool_result_text:
              truncated = truncate_tool_output(tool_result_text)

              # Treat failures specially so model knows to retry
              is_failure = "TOOL FAILURE:" in truncated

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

              # If tool failed, ensure we loop again (don't break)
              executed_tools = True  # Mark that we tried
            elif result is None:
              # Tool execution resulted in error - still mark as executed
              executed_tools = True

          # After executing ALL tools, compact once and call model again
          session_id = save_history(messages, session_id)
          CURRENT_MESSAGES = messages
          CURRENT_SESSION_ID = session_id

          messages, tokens, limit = compact_context(messages, p_model)
          console.print(Align.right(Text(f"Context: {(tokens*100)//limit}%", style="dim")))

          continue

        # (AUTO-CONTINUE block removed)

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