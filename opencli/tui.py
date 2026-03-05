"""
Textual-based TUI for OpenCLI - gives us windows, proper resize handling, and opencode-like interface.
"""

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import Static, Button, Input, Header, Footer, RichLog, Label
from textual.reactive import reactive
from textual import work
import asyncio


class OpenCLIApp(App):
  """Main OpenCLI Textual application."""
  
  CSS = """
  Screen {
    background: $surface;
  }
  
  # main-container {
    height: 100%;
    width: 100%;
  }
  
  # left-panel {
    width: 35%;
    height: 100%;
    border: solid green;
    padding: 1;
  }
  
  # right-panel {
    width: 65%;
    height: 100%;
    border: solid green;
  }
  
  # banner-area {
    height: auto;
    border: solid green;
    padding: 1;
  }
  
  # chat-area {
    height: 1fr;
    border: solid dim;
    padding: 1;
    overflow-y: auto;
  }
  
  # thinking-area {
    height: auto;
    border: solid yellow;
    padding: 1;
    display: none;
  }
  
  # thinking-area.visible {
    display: block;
  }
  
  # input-area {
    height: auto;
    border: solid green;
    padding: 1;
  }
  
  # status-bar {
    height: 1;
    dock: bottom;
    background: $primary;
    color: white;
  }
  
  .message-user {
    background: $primary-darken-1;
    padding: 1;
    margin: 1;
  }
  
  .message-assistant {
    background: $secondary-darken-1;
    padding: 1;
    margin: 1;
  }
  
  .message-tool {
    background: $accent;
    padding: 1;
    margin: 1;
  }
  
  .ghost-text {
    color: green;
    text-style: bold;
  }
  
  """
  
  # Reactive state
  mode = reactive("safe")
  model = reactive("default")
  folder = reactive("~")
  
  def __init__(self, on_input_callback=None, on_quit_callback=None):
    super().__init__()
    self.on_input_callback = on_input_callback
    self.on_quit_callback = on_quit_callback
    self._thinking = False
    self._thinking_text = ""
  
  def compose(self) -> ComposeResult:
    """Create the layout."""
    # Header with mode and model info
    yield Header(show_clock=True)
    
    # Main container with left (banner) and right (chat) panels
    with Container(id="main-container"):
      # Left panel - banner info
      with Vertical(id="left-panel"):
        yield Static("OpenCLI", id="logo-text")
        yield Static("- by curren -", id="by-text")
        yield Static("", id="folder-text")
        yield Static("", id="model-text")
        yield Static("", id="mode-text")
        yield Static("", id="tips-text")
      
      # Right panel - chat and input
      with Vertical(id="right-panel"):
        # Chat area
        with ScrollableContainer(id="chat-area"):
          yield RichLog(id="chat-log", auto_scroll=True)
        
        # Thinking area (hidden by default)
        with Vertical(id="thinking-area"):
          yield Static("● Thinking", id="thinking-label")
          yield RichLog(id="thinking-log")
        
        # Input area
        with Horizontal(id="input-area"):
          yield Input(placeholder="Message OpenCLI...", id="user-input")
          yield Button("Send", id="send-btn", variant="primary")
    
    # Status bar
    yield Static("SAFE | Model: -- | Tokens: 0", id="status-bar")
  
  def on_mount(self) -> None:
    """Handle mount."""
    # Set up the banner
    self.update_banner()
    
    # Set up input handler
    input_widget = self.query_one("#user-input", Input)
    input_widget.focus()
    
    # Handle enter key in input
    def handle_submit():
      self.send_message()
    
    input_widget.on_submit = handle_submit
    
    # Handle button click
    btn = self.query_one("#send-btn", Button)
    btn.on_click = lambda _: self.send_message()
  
  def update_banner(self, mode="safe", model="default", folder="~"):
    """Update the banner information."""
    self.mode = mode
    self.model = model
    self.folder = folder
    
    # Update left panel
    self.query_one("#logo-text", Static).update(
      "[green]_ ____ _____ _  _ ____ _   ___ \n"
      "/ \\| _ \\| ___| \\ | |/ ___| |  |_ _|\n"
      "| | | |_) | _| | \\| | |  | |  | | \n"
      "| | | __/| |___| |\\| |___| |___ | | \n"
      "\\_/|_|  |_____|_| \\_|\\____|_____|___|[/green]"
    )
    self.query_one("#by-text", Static).update("[dim]- by curren -[/dim]")
    self.query_one("#folder-text", Static).update(f"[green]Folder: {folder}[/green]")
    self.query_one("#model-text", Static).update(f"[green]Model: {model}[/green]")
    self.query_one("#mode-text", Static).update(f"[bold green]Mode: {mode.upper()}[/bold green]")
    self.query_one("#tips-text", Static).update(
      "[dim]Tip: SAFE asks • UNSAFE auto-runs • PLAN reads only\n"
      "/settings • /resume • /clear • exit[/dim]"
    )
    
    # Update status bar
    self.query_one("#status-bar", Static).update(
      f"{mode.upper()} | Model: {model} | Tokens: 0"
    )
  
  def add_message(self, text: str, msg_type: str = "assistant"):
    """Add a message to the chat log."""
    chat_log = self.query_one("#chat-log", RichLog)
    
    if msg_type == "user":
      chat_log.write(f"[blue]You:[/blue] {text}")
    elif msg_type == "tool":
      chat_log.write(f"[yellow]Tool:[/yellow] {text}")
    elif msg_type == "thinking":
      # Show thinking area
      thinking_area = self.query_one("#thinking-area")
      thinking_area.add_class("visible")
      thinking_log = self.query_one("#thinking-log", RichLog)
      thinking_log.write(text)
    else:
      chat_log.write(f"[green]AI:[/green] {text}")
  
  def clear_thinking(self):
    """Clear thinking area and hide it."""
    thinking_area = self.query_one("#thinking-area")
    thinking_area.remove_class("visible")
    thinking_log = self.query_one("#thinking-log", RichLog)
    thinking_log.clear()
  
  def set_thinking(self, thinking: bool, text: str = ""):
    """Show or hide thinking indicator."""
    self._thinking = thinking
    self._thinking_text = text
    
    thinking_area = self.query_one("#thinking-area")
    
    if thinking:
      thinking_area.add_class("visible")
      self.query_one("#thinking-label", Static).update("● Thinking")
    else:
      thinking_area.remove_class("visible")
      self.query_one("#thinking-log", RichLog).clear()
  
  def send_message(self):
    """Handle send button or enter key."""
    input_widget = self.query_one("#user-input", Input)
    text = input_widget.value.strip()
    
    if not text:
      return
    
    # Add user message to chat
    self.add_message(text, "user")
    
    # Clear input
    input_widget.value = ""
    
    # Call the callback if set
    if self.on_input_callback:
      self.on_input_callback(text)
  
  def add_response(self, text: str):
    """Add assistant response to chat."""
    self.clear_thinking()
    self.add_message(text, "assistant")
  
  def add_tool_result(self, tool_name: str, result: str):
    """Add tool result to chat."""
    self.add_message(f"[{tool_name}]: {result[:200]}...", "tool")
  
  def update_status(self, mode: str, model: str, tokens: int = 0):
    """Update status bar."""
    self.query_one("#status-bar", Static).update(
      f"{mode.upper()} | Model: {model} | Tokens: {tokens}"
    )
  
  def run_chat_mode(self):
    """Run the chat interface (this is called after banner)."""
    # The app is already running, this just keeps it alive
    pass


class ToolApprovalModal(App):
  """Modal for approving tool execution."""
  
  CSS = """
  ToolApprovalModal {
    align: center middle;
  }
  
  # approval-container {
    width: 60;
    height: auto;
    border: solid green;
    padding: 1;
    background: $surface;
  }
  
  # approval-title {
    text-style: bold;
    color: green;
  }
  
  # approval-info {
    margin: 1 0;
  }
  
  # approval-buttons {
    height: 3;
    align: center middle;
  }
  
  Button {
    margin: 0 1;
  }
  """
  
  tool_name = ""
  tool_args = ""
  approved = False
  
  def __init__(self, tool_name: str, tool_args: dict, on_approve=None, on_deny=None):
    super().__init__()
    self.tool_name = tool_name
    self.tool_args = tool_args
    self.on_approve = on_approve
    self.on_deny = on_deny
  
  def compose(self) -> ComposeResult:
    with Vertical(id="approval-container"):
      yield Static(f"● Approve {self.tool_name}?", id="approval-title")
      yield Static(f"Args: {self.tool_args}", id="approval-info")
      with Horizontal(id="approval-buttons"):
        yield Button("y - Approve", id="btn-approve", variant="primary")
        yield Button("n - Deny", id="btn-deny", variant="error")
        yield Button("q - Quit", id="btn-quit", variant="default")
  
  def on_button_pressed(self, event: Button.Pressed) -> None:
    """Handle button press."""
    if event.button.id == "btn-approve":
      self.approved = True
      if self.on_approve:
        self.on_approve()
      self.dismiss(True)
    elif event.button.id == "btn-deny":
      self.approved = False
      if self.on_deny:
        self.on_deny()
      self.dismiss(False)
    elif event.button.id == "btn-quit":
      self.dismiss("quit")


def run_textual_app(on_input_callback=None, on_quit_callback=None):
  """Run the Textual app."""
  app = OpenCLIApp(on_input_callback=on_input_callback, on_quit_callback=on_quit_callback)
  return app


if __name__ == "__main__":
  app = OpenCLIApp()
  app.run()
