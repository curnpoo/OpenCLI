OpenCLI

OpenCLI is an open-source terminal coding agent.

If you’ve used Claude Code or Codex and thought:

“I want this… but I want to control it.”

This is that.

No SaaS lock-in.
No black box magic.
Just a local, hackable, extensible agent that runs in your terminal.

⸻

What It Is

OpenCLI is a developer-first AI coding assistant designed to:
	•	Read and modify files
	•	Run shell commands
	•	Batch tool calls
	•	Plan before acting
	•	Ask for approval in safe mode
	•	Run fully autonomous in unsafe mode
	•	Resume sessions
	•	Handle multiple parallel terminal windows safely

It behaves like a serious coding agent, not a chatbot pretending to be one.

⸻

Why It Exists

Claude Code and Codex are powerful.

But they’re:
	•	Closed source
	•	Opinionated
	•	Not fully inspectable
	•	Not easily extendable

OpenCLI gives you:
	•	Full control over prompts
	•	Full control over tooling
	•	Full control over models
	•	Full control over execution behavior
	•	The ability to run multiple agents in parallel

It’s meant to be modified.

If something feels wrong, you can fix it.

⸻

Features
	•	Plan → Batch Tool Execution → Final Reasoning loop
	•	OpenAI-style function calling
	•	Parallel session support (isolated session files + file locking)
	•	Tool approval system (SAFE mode)
	•	Fully autonomous mode (UNSAFE)
	•	PLAN mode (analysis-only, no execution)
	•	Context compaction with smart token counting
	•	Resume previous sessions
	•	Multiple provider support (Anthropic, OpenRouter, NVIDIA, Gemini, Ollama, etc.)
	•	Clean green-themed terminal UI with visible reasoning separation
	•	Response timing metrics (how long each response took)
	•	Token counting (input/output estimation)
	•	File diffs with color-coded changes
	•	Single-key tool approval (y/n, no Enter needed)
	•	Mode toggle with Shift+Tab
	•	Modified file tracking per session
	•	Destructive tool warnings

⸻

Philosophy

OpenCLI is built around a few rules:
	•	Do not scan the whole repo unless needed
	•	Do not act on greetings
	•	Plan before modifying
	•	Prefer minimal edits over rewrites
	•	Never hide what the agent is doing
	•	Keep thinking visible but separate from output

It should feel like a competent developer working with you — not replacing you.

⸻

Install

pip install opencli

Or clone and install locally:

git clone https://github.com/yourname/opencli.git
cd opencli
pip install -e .

Then run:

opencli


⸻

Modes

SAFE (Green)
	•	Requires approval for each tool
	•	Single-key approval (y/n)
	•	Good for destructive operations
	•	Default mode

UNSAFE (Red)
	•	Executes tools automatically
	•	Meant for trusted environments
	•	No approval delays

PLAN (Dark Green)
	•	Read-only analysis mode
	•	Learns and understands code
	•	Never executes tools
	•	Perfect for exploration

Switch with Shift+Tab or via /settings
Approval requires single keystroke - no Enter needed!

⸻

Parallel Sessions

You can run:

opencli

in multiple terminal windows simultaneously.

Each session:
	•	Gets a unique UUID
	•	Writes to its own session file
	•	Uses file locking for safety

No race conditions.
No history corruption.

⸻

Models

Use whatever model you want.

OpenCLI doesn’t care.

Claude Sonnet.
Claude Opus.
Kimi.
Gemini.
OpenRouter.

If it supports function calling, it works here.

⸻

Is It Production Ready?

It’s stable.

But it’s also meant to evolve.

If you want a frozen appliance, this isn’t it.

If you want something you can tune, modify, and push further — it is.

⸻

Contributing

Pull requests welcome.

If you break it and make it better, even better.

⸻

Final Note

OpenCLI is for people who:
	•	Like understanding their tools
	•	Want autonomous coding agents
	•	Don’t want to surrender control

If you wanted Claude Code —
but open source —

this is it.