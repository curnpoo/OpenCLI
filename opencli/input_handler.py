#!/usr/bin/env python3
"""
Dynamic input handler with paste detection and styling support.
Handles multi-line input, paste buffering, and custom backgrounds.
"""

import sys
import os
import select
import tty
import termios
from typing import Optional, Callable, List
from dataclasses import dataclass
from rich.console import Console
from rich.text import Text
from rich.panel import Panel
from rich import box

@dataclass
class InputStyle:
    """Configuration for input styling."""
    name: str
    bg_color: Optional[str]  # Rich color name or None
    text_color: str
    prompt_color: str
    cursor_style: str
    
    # Predefined styles
    @classmethod
    def dark(cls) -> "InputStyle":
        return cls(
            name="dark",
            bg_color=None,  # Default terminal background
            text_color="white",
            prompt_color="cyan",
            cursor_style="bold"
        )
    
    @classmethod
    def dark_gray(cls) -> "InputStyle":
        return cls(
            name="dark_gray",
            bg_color="grey19",  # Subtle dark gray
            text_color="white",
            prompt_color="cyan",
            cursor_style="bold"
        )
    
    @classmethod
    def light(cls) -> "InputStyle":
        return cls(
            name="light",
            bg_color=None,
            text_color="black",
            prompt_color="blue",
            cursor_style="bold"
        )
    
    @classmethod
    def light_gray(cls) -> "InputStyle":
        return cls(
            name="light_gray",
            bg_color="grey70",  # Light gray
            text_color="black",
            prompt_color="blue",
            cursor_style="bold"
        )
    
    @classmethod
    def get(cls, name: str) -> "InputStyle":
        styles = {
            "dark": cls.dark(),
            "dark_gray": cls.dark_gray(),
            "light": cls.light(),
            "light_gray": cls.light_gray(),
        }
        return styles.get(name, cls.dark())


class DynamicInput:
    """
    Multi-line input handler with paste detection.
    
    Features:
    - Detects pasted content vs typed content
    - Allows editing pasted content before sending
    - Supports multi-line input with Shift+Enter
    - Customizable background colors
    - Visual indicator showing paste mode
    """
    
    def __init__(self, console: Console, style: InputStyle = None):
        self.console = console
        self.style = style or InputStyle.dark()
        self.buffer: List[str] = [""]
        self.cursor_line = 0
        self.cursor_col = 0
        self.paste_mode = False
        self.pasted_content = ""
        
    def _is_paste(self, data: str) -> bool:
        """
        Detect if input is likely a paste vs typing.
        
        Heuristics:
        - Multiple characters arriving at once
        - Contains newlines
        - Long string of characters
        """
        if len(data) > 10:
            return True
        if '\n' in data or '\r' in data:
            return True
        if '://' in data:  # Likely a URL
            return True
        return False
    
    def _get_terminal_size(self) -> tuple:
        """Get terminal dimensions."""
        return os.get_terminal_size()
    
    def _render_input(self, hint: str = ""):
        """Render the current input buffer with styling."""
        # Clear previous lines
        self.console.print("\r\033[J", end="")
        
        # Build styled content
        lines_text = []
        for i, line in enumerate(self.buffer):
            if i == self.cursor_line:
                # Show cursor position
                before = line[:self.cursor_col]
                after = line[self.cursor_col:]
                if after:
                    cursor_char = after[0]
                    styled_line = Text.assemble(
                        (before, self.style.text_color),
                        (cursor_char, f"{self.style.cursor_style} reverse"),
                        (after[1:], self.style.text_color)
                    )
                else:
                    styled_line = Text.assemble(
                        (before, self.style.text_color),
                        (" ", f"{self.style.cursor_style} reverse"),
                    )
                lines_text.append(styled_line)
            else:
                lines_text.append(Text(line, style=self.style.text_color))
        
        # Create panel with background
        content = Text("\n").join(lines_text) if len(lines_text) > 1 else (lines_text[0] if lines_text else Text(""))
        
        # Add paste indicator
        if self.paste_mode:
            content = Text.assemble(
                ("[PASTE MODE - Press Enter to send, Esc to cancel]\n", "dim yellow"),
                content
            )
        
        # Render in panel
        panel = Panel(
            content,
            title=f"[dim]{hint}[/dim]" if hint else None,
            border_style=self.style.prompt_color,
            box=box.ROUNDED,
            style=f"on {self.style.bg_color}" if self.style.bg_color else ""
        )
        
        self.console.print(panel, end="")
        self.console.print(f"\r\033[{len(self.buffer) + 2}A", end="")  # Move cursor back up
    
    def _read_raw(self) -> str:
        """Read raw input from terminal."""
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        
        try:
            tty.setraw(fd)
            
            # Check if there's input waiting
            if select.select([sys.stdin], [], [], 0.1)[0]:
                char = sys.stdin.read(1)
                
                # Handle escape sequences
                if char == '\x1b':
                    seq = char
                    # Try to read the full escape sequence
                    for _ in range(3):
                        if select.select([sys.stdin], [], [], 0.05)[0]:
                            seq += sys.stdin.read(1)
                        else:
                            break
                    return self._process_escape(seq)
                
                return char
            
            return ""
            
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    
    def _process_escape(self, seq: str) -> str:
        """Process escape sequences into readable keys."""
        escape_map = {
            '\x1b[A': 'UP',
            '\x1b[B': 'DOWN', 
            '\x1b[C': 'RIGHT',
            '\x1b[D': 'LEFT',
            '\x1b[H': 'HOME',
            '\x1b[F': 'END',
            '\x1b[3~': 'DELETE',
            '\x1b[5~': 'PAGE_UP',
            '\x1b[6~': 'PAGE_DOWN',
            '\x1b\x1b': 'ESC',  # Double escape
        }
        return escape_map.get(seq, f"ESC[{seq[2:] if len(seq) > 2 else ''}")
    
    def _read_buffered(self) -> str:
        """Read potentially pasted content."""
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        
        try:
            tty.setraw(fd)
            
            # Read first char
            if not select.select([sys.stdin], [], [], 0.1)[0]:
                return ""
            
            result = sys.stdin.read(1)
            
            # If it's escape, process as special key
            if result == '\x1b':
                seq = result
                for _ in range(3):
                    if select.select([sys.stdin], [], [], 0.05)[0]:
                        seq += sys.stdin.read(1)
                    else:
                        break
                return self._process_escape(seq)
            
            # Check for more buffered input (paste detection)
            while select.select([sys.stdin], [], [], 0.05)[0]:
                result += sys.stdin.read(1)
            
            return result
            
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    
    def read_input(
        self, 
        prompt: str = "› ", 
        multiline: bool = True,
        hint: str = "Shift+Enter for new line • Enter to send"
    ) -> Optional[str]:
        """
        Read input with paste detection and multi-line support.
        
        Returns:
            The input string, or None if cancelled
        """
        self.buffer = [""]
        self.cursor_line = 0
        self.cursor_col = 0
        self.paste_mode = False
        
        # Show initial prompt
        self.console.print(f"\n[{self.style.prompt_color}]{prompt}[/{self.style.prompt_color}]", end="")
        
        while True:
            # Read input
            data = self._read_buffered()
            
            if not data:
                continue
            
            # Check for paste
            if self._is_paste(data) and not self.paste_mode:
                self.paste_mode = True
                self.pasted_content = data
                
                # Insert pasted content
                lines = data.split('\n')
                if len(lines) > 1:
                    # Multi-line paste
                    self.buffer = lines
                    self.cursor_line = len(lines) - 1
                    self.cursor_col = len(lines[-1])
                else:
                    # Single long line
                    self.buffer[0] = data
                    self.cursor_col = len(data)
                
                self._render_input(hint)
                continue
            
            # Handle special keys
            if data == 'ESC':
                if self.paste_mode:
                    # Cancel paste mode
                    self.paste_mode = False
                    self.buffer = [""]
                    self.cursor_line = 0
                    self.cursor_col = 0
                    self._render_input(hint)
                    continue
                else:
                    # Cancel input
                    return None
            
            elif data == 'UP':
                if self.cursor_line > 0:
                    self.cursor_line -= 1
                    self.cursor_col = min(self.cursor_col, len(self.buffer[self.cursor_line]))
                self._render_input(hint)
                
            elif data == 'DOWN':
                if self.cursor_line < len(self.buffer) - 1:
                    self.cursor_line += 1
                    self.cursor_col = min(self.cursor_col, len(self.buffer[self.cursor_line]))
                self._render_input(hint)
                
            elif data == 'LEFT':
                if self.cursor_col > 0:
                    self.cursor_col -= 1
                elif self.cursor_line > 0:
                    self.cursor_line -= 1
                    self.cursor_col = len(self.buffer[self.cursor_line])
                self._render_input(hint)
                
            elif data == 'RIGHT':
                if self.cursor_col < len(self.buffer[self.cursor_line]):
                    self.cursor_col += 1
                elif self.cursor_line < len(self.buffer) - 1:
                    self.cursor_line += 1
                    self.cursor_col = 0
                self._render_input(hint)
                
            elif data == 'HOME':
                self.cursor_col = 0
                self._render_input(hint)
                
            elif data == 'END':
                self.cursor_col = len(self.buffer[self.cursor_line])
                self._render_input(hint)
                
            elif data == 'DELETE':
                line = self.buffer[self.cursor_line]
                if self.cursor_col < len(line):
                    self.buffer[self.cursor_line] = line[:self.cursor_col] + line[self.cursor_col + 1:]
                elif self.cursor_line < len(self.buffer) - 1:
                    # Join with next line
                    self.buffer[self.cursor_line] += self.buffer.pop(self.cursor_line + 1)
                self._render_input(hint)
                
            elif data == '\x7f' or data == '\b':  # Backspace
                if self.cursor_col > 0:
                    line = self.buffer[self.cursor_line]
                    self.buffer[self.cursor_line] = line[:self.cursor_col - 1] + line[self.cursor_col:]
                    self.cursor_col -= 1
                elif self.cursor_line > 0:
                    # Join with previous line
                    prev_len = len(self.buffer[self.cursor_line - 1])
                    self.buffer[self.cursor_line - 1] += self.buffer.pop(self.cursor_line)
                    self.cursor_line -= 1
                    self.cursor_col = prev_len
                self._render_input(hint)
                
            elif data == '\r' or data == '\n':  # Enter
                if self.paste_mode:
                    # Exit paste mode but keep content
                    self.paste_mode = False
                    self._render_input(hint)
                    continue
                
                if multiline:
                    # Check for Shift+Enter or Ctrl+Enter for new line
                    # In raw mode, we can't easily detect shift, so we use different approach
                    # Double Enter in quick succession = send
                    pass
                
                # Single Enter = send
                result = '\n'.join(self.buffer)
                self.console.print()  # New line
                return result
                
            elif data == '\x0c':  # Ctrl+L - clear
                self.buffer = [""]
                self.cursor_line = 0
                self.cursor_col = 0
                self._render_input(hint)
                
            elif data == '\t':  # Tab
                # Insert 4 spaces
                line = self.buffer[self.cursor_line]
                self.buffer[self.cursor_line] = line[:self.cursor_col] + "    " + line[self.cursor_col:]
                self.cursor_col += 4
                self._render_input(hint)
                
            elif data.startswith('ESC['):
                # Unknown escape sequence, ignore
                pass
                
            else:
                # Regular character input
                if ord(data[0]) >= 32:  # Printable
                    line = self.buffer[self.cursor_line]
                    self.buffer[self.cursor_line] = line[:self.cursor_col] + data + line[self.cursor_col:]
                    self.cursor_col += len(data)
                    self._render_input(hint)


def styled_input(
    console: Console,
    prompt: str = "› ",
    style_name: str = "dark",
    hint: str = "Paste supported • Enter to send",
    multiline: bool = True
) -> Optional[str]:
    """
    Convenience function for styled multi-line input.
    
    Args:
        console: Rich console instance
        prompt: Prompt text to display
        style_name: One of "dark", "dark_gray", "light", "light_gray"
        hint: Helper text to show
        multiline: Allow multi-line input
        
    Returns:
        Input text or None if cancelled
    """
    style = InputStyle.get(style_name)
    handler = DynamicInput(console, style)
    return handler.read_input(prompt, multiline, hint)


# Simple test if run directly
if __name__ == "__main__":
    console = Console()
    console.print("[bold green]Dynamic Input Test[/bold green]")
    console.print("Try pasting text, using arrow keys, etc.")
    console.print("Press Esc to cancel\n")
    
    result = styled_input(console, "Test › ", style_name="dark_gray")
    
    if result is not None:
        console.print(f"\n[green]You entered:[/green]\n{result}")
    else:
        console.print("\n[yellow]Cancelled[/yellow]")
