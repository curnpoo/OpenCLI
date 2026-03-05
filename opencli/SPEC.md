# OpenCLI Rewrite Specification

## Project Overview

**Project Name**: OpenCLI  
**Type**: Autonomous terminal coding agent  
**Core Functionality**: AI-powered coding assistant that autonomously executes tasks until completion  
**Target Users**: Developers who want control, transparency, and extensibility

---

## Architecture

### Current Architecture (to be rewritten)
- Single `main.py` file (~2100 lines)
- Linear while-true loop in `main()` function
- Rich library for terminal UI
- Global state management

### Target Architecture
- **Framework**: Textual (for modern TUI)
- **Structure**: Class-based application with screens/panels
- **Separation of concerns**:
  - `app.py` - Main application class
  - `screens/` - Chat screen, settings screen, etc.
  - `widgets/` - Reusable UI components (footer, message bubble, etc.)
  - `services/` - API calls, file operations, config management

---

## UI Specification

### Layout Structure

```
┌─────────────────────────────────────────────────────┐
│ HEADER: Logo + Status                               │
├─────────────────────────────────────────────────────┤
│                                                     │
│ MAIN CONTENT: Chat messages + tool output           │
│ - Scrollable message history                       │
│ - Tool execution panels                            │
│ - Thinking/reasoning blocks                        │
│                                                     │
├─────────────────────────────────────────────────────┤
│ FOOTER: Mode | Input Prompt | Model + Context %    │
└─────────────────────────────────────────────────────┘
```

### Header
- Logo ASCII art (current banner)
- Current working directory
- Session status indicator

### Main Content Area
- Scrollable message history
- User messages (right-aligned or distinct style)
- Assistant messages with reasoning + tool calls
- Tool output panels
- Thinking blocks (collapsible)

### Footer (Input Area)

The footer is the text input area. Due to terminal limitations, input must be on a separate line.

**Layout**:
```
┌────────────────────────────────────────────────────────────────────────┐
│ SAFE                              Model: gpt-4o    Context: 45%        │
│ You › [user types here..............................................] │
└────────────────────────────────────────────────────────────────────────┘
```

- **Line 1**: Mode (left) | Model + Context % (right)
- **Line 2**: Input prompt "You › " (cyan/bold) + user types

This is the closest we can get to a text box in a terminal. The mode, model, and context are always visible on the line above where you type.

---

## Functionality Specification

### Core Features

1. **Autonomous Execution**
   - Give a task, walk away
   - Up to 20 steps per task
   - Automatic iteration until completion

2. **Tool System**
   | Tool | Purpose |
   |------|---------|
   | `list_files` | List directory contents |
   | `search_code` | Grep/search code |
   | `read_file` | Read file contents |
   | `write_file` | Create new file |
   | `replace_text` | Modify existing text |
   | `run_shell` | Execute shell command |

3. **Mode System**
   - **SAFE**: Approve each tool (y/n)
   - **UNSAFE**: Auto-execute
   - **PLAN**: Read-only analysis
   - Toggle with Shift+Tab

4. **Session Management**
   - Resume previous sessions (`/resume`)
   - Clear session (`/clear`)
   - Auto-save on exit
   - File locking for parallel sessions

5. **Context Management**
   - Token counting
   - Auto-compaction when nearing limit
   - Context % display in footer

6. **Settings**
   - Provider selection (Anthropic, OpenRouter, NVIDIA, Gemini, Ollama)
   - Model selection
   - Theme customization
   - Mode selection

### Commands
| Command | Action |
|---------|--------|
| `/` | Open settings |
| `/resume` | Resume session |
| `/clear` | New session |
| `exit` | Quit |
| `Ctrl+C` | Quit (with save) |

---

## Technical Requirements

### Dependencies
```toml
[project.dependencies]
requests = "*"
rich = "*"
textual = "*"  # NEW - for modern TUI
```

### API Integration
- OpenAI-style function calling
- Support for multiple providers:
  - Anthropic (Claude)
  - OpenRouter
  - NVIDIA
  - Google (Gemini)
  - Ollama (local)

### Data Persistence
- Sessions saved to `~/.opencli/sessions/`
- Config in `~/.opencli/config.json`
- File locking for safety

---

## File Structure (Target)

```
opencli/
├── __init__.py
├── app.py              # Main Textual app
├── config.py            # Config management
├── services/
│   ├── api.py          # Provider API calls
│   ├── tools.py        # Tool implementations
│   ├── session.py      # Session management
│   └── context.py     # Context compaction
├── screens/
│   ├── chat.py         # Main chat screen
│   └── settings.py     # Settings screen
├── widgets/
│   ├── header.py       # Header widget
│   ├── footer.py       # Footer with input
│   ├── message.py      # Message bubbles
│   └── tool_output.py  # Tool execution display
└── prompts/
    ├── system.py       # System prompt
    └── tools.md        # Tool definitions
```

---

## Success Criteria

1. ✅ Autonomous execution - Give task, walk away, task completes
2. ✅ All tools work (list_files, search_code, read_file, write_file, replace_text, run_shell)
3. ✅ Mode switching (SAFE/UNSAFE/PLAN)
4. ✅ Session management (resume, clear, auto-save)
5. ✅ Fixed footer with mode, input, model+context
6. ✅ Settings menu
7. ✅ Context management with % display
8. ✅ Parallel session support
9. ✅ Clean, modern UI (Claude Code / Codex-like)

---

## Notes for Rewrite Agent

1. **Preserve the agentic behavior** - The core value is being able to give a command and walk away
2. **Keep tool definitions** - The TOOLS.md and OPENCLI.md prompts define agent behavior
3. **Maintain compatibility** - Same commands, same tools, same modes
4. **Focus on UX** - The UI should feel professional and modern
5. **Test thoroughly** - Verify autonomous execution works end-to-end
