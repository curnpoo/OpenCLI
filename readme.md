OpenCLI
======

OpenCLI is an autonomous terminal coding agent.

Give it a task, walk away, and it keeps working until the task is complete.

No SaaS lock-in. No black box magic. Just a local, hackable, extensible agent that runs in your terminal.

---

## What It Is

OpenCLI is a developer-first AI coding assistant designed to:
- **Autonomously execute tasks** - Give a command and it works until done
- **Read and modify files** - Full file system access
- **Run shell commands** - npm, git, build tools, etc.
- **Plan before acting** - Analyzes before making changes
- **Ask for approval in safe mode** - Single keystroke (y/n)
- **Run fully autonomous in unsafe mode** - No delays, continuous execution
- **Resume sessions** - Pick up where you left off
- **Handle multiple parallel windows** - Isolated sessions with file locking

It behaves like a competent developer working with you — not a chatbot.

---

## Why It Exists

Claude Code and Codex are powerful. But they're:
- Closed source
- Opinionated
- Not fully inspectable

OpenCLI gives you:
- Full control over prompts
- Full control over tooling
- Full control over models
- Full control over execution behavior
- The ability to run autonomous agents and walk away

It's meant to be modified. If something feels wrong, you can fix it.

---

## Agentic Workflow

OpenCLI is designed as an **autonomous agent**, not a chatbot:

1. **Give a task**: "Fix the login bug" or "Add dark mode to the app"
2. **It works**: Reads files, runs commands, makes changes
3. **Keeps going**: Iterates until the task is complete
4. **Reports back**: Summarizes what was done

**You can give a command and walk away.** It will:
- Explore the codebase to understand the problem
- Make surgical edits
- Run tests and verify fixes
- Continue iterating until the task is fulfilled

Up to 20 steps per task. Each model response + tool execution = 1 step.

---

## Quick Start

```bash
pip install opencli
```

Or clone and install locally:

```bash
git clone https://github.com/yourname/opencli.git
cd opencli
pip install -e .
```

Then run:

```bash
opencli
```

---

## Commands

| Command | Description |
|---------|-------------|
| `/` | Open settings menu |
| `/resume` | Resume a previous session |
| `/clear` | Start a fresh session |
| `exit` or `Ctrl+C` | Quit the application |

---

## Modes

### SAFE (Green) - Default
- Requires approval for each tool call
- Single-key approval (y/n) - no Enter needed
- Good for destructive operations or when you want oversight

### UNSAFE (Red)
- Executes tools automatically
- Meant for trusted environments
- No approval delays
- **Give a task and walk away** - it will keep working

### PLAN (Dark Green)
- Read-only analysis mode
- Learns and understands code
- Never executes tools
- Perfect for exploration and understanding

**Switch modes**: Press `Shift+Tab` or use `/settings`

---

## Features

- **Autonomous execution** - Work until task completion
- **Plan → Tool Execution → Final Reasoning loop**
- **OpenAI-style function calling**
- **Parallel session support** - Multiple windows, isolated files + locking
- **Context compaction** - Smart token management
- **Multiple providers** - Anthropic, OpenRouter, NVIDIA, Gemini, Ollama, etc.
- **Response timing metrics**
- **Token counting** (input/output estimation)
- **File diffs** with color-coded changes
- **Modified file tracking** per session

---

## Parallel Sessions

You can run `opencli` in multiple terminal windows simultaneously.

Each session:
- Gets a unique UUID
- Writes to its own session file
- Uses file locking for safety

No race conditions. No history corruption.

---

## Models

Use whatever model you want. OpenCLI doesn't care.

- Claude Sonnet / Opus
- Kimi
- Gemini
- OpenRouter (any model)
- Ollama (local models)

If it supports function calling, it works here.

---

## Configuration

Run `/` to access settings:
- **Provider** - Choose your API provider
- **Model** - Select the model
- **Ollama URL** - For local models
- **Theme** - UI color preferences
- **Execution Mode** - SAFE / UNSAFE / PLAN

---

## Philosophy

OpenCLI is built around a few rules:
- Do not scan the whole repo unless needed
- Do not act on greetings
- Plan before modifying
- Prefer minimal edits over rewrites
- Never hide what the agent is doing
- Keep thinking visible but separate from output

---

## Is It Production Ready?

It's stable. But it's also meant to evolve.

If you want a frozen appliance, this isn't it.

If you want something you can tune, modify, and push further — it is.

---

## Contributing

Pull requests welcome. If you break it and make it better, even better.

---

## Final Note

OpenCLI is for people who:
- Like understanding their tools
- Want autonomous coding agents
- Don't want to surrender control

If you wanted Claude Code — but open source — this is it.