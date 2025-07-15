from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.live import Live
from rich.layout import Layout
from rich.tree import Tree
from rich.rule import Rule
from rich.columns import Columns
from rich.align import Align
from typing import List, Dict, Optional, Set
from datetime import datetime, timedelta
from pydantic import ValidationError
import time

from .events import (
    MCPEvent, EventType, Role, ContentType, ConversationSummary,
    ContentBlock, ToolExecution
)


def _get_enum_value(enum_value):
    """
    Helper function to safely get enum values.
    Handles both cases where the value is already a string (due to Pydantic use_enum_values=True)
    and where it's still an enum instance.
    """
    if isinstance(enum_value, str):
        return enum_value
    elif hasattr(enum_value, 'value'):
        return enum_value.value
    else:
        return str(enum_value)


class MCPDisplayConfig:
    """Configuration settings for the MCP display."""
    
    def __init__(
        self,
        show_timestamps: bool = True,
        show_token_counts: bool = True,
        show_content_preview: bool = True,
        max_content_preview_length: int = 800,
        auto_scroll: bool = True,
        color_theme: str = "default"
    ):
        self.show_timestamps = show_timestamps
        self.show_token_counts = show_token_counts
        self.show_content_preview = show_content_preview
        self.max_content_preview_length = max_content_preview_length
        self.auto_scroll = auto_scroll
        self.color_theme = color_theme


class MCPTerminalDisplay:
    """
    Rich terminal display for real-time MCP conversation flow visualization.
    
    This class provides a live, interactive terminal interface that shows MCP events
    as they happen, including message flow, token usage, tool executions, and
    conversation statistics.
    """
    
    def __init__(self, config: Optional[MCPDisplayConfig] = None):
        """
        Initialize the MCP terminal display.
        
        Args:
            config: Optional display configuration. If None, uses default settings.
        """
        self.console = Console()
        self.config = config or MCPDisplayConfig()
        self.events: List[MCPEvent] = []
        self.conversations: Dict[str, List[MCPEvent]] = {}
        self.active_conversations: Set[str] = set()
        self.tool_executions: Dict[str, ToolExecution] = {}  # tool_id -> execution
        
        # Display state
        self.live_display: Optional[Live] = None
        self.start_time = datetime.now()
        
        # Color scheme
        self.colors = self._get_color_scheme()
    
    def _get_color_scheme(self) -> Dict[str, str]:
        """
        Get color scheme for different elements.
        
        Returns:
            Dict mapping element types to Rich color names
        """
        return {
            'user': 'bright_blue',
            'assistant': 'bright_green',
            'system': 'bright_yellow',
            'text': 'white',
            'tool_use': 'bright_cyan',
            'tool_result': 'bright_magenta',
            'success': 'bright_green',
            'error': 'bright_red',
            'warning': 'bright_yellow',
            'timestamp': 'dim white',
            'tokens': 'bright_yellow',
            'separator': 'dim blue'
        }
    
    def add_event(self, event: MCPEvent) -> None:
        """
        Add a new MCP event to the display.
        
        Args:
            event: The MCP event to add and display
        """
        self.events.append(event)
        
        # Organize by conversation
        if event.conversation_id not in self.conversations:
            self.conversations[event.conversation_id] = []
        self.conversations[event.conversation_id].append(event)
        
        # Track active conversations
        if event.event_type != EventType.CONVERSATION_ENDED:
            self.active_conversations.add(event.conversation_id)
        else:
            self.active_conversations.discard(event.conversation_id)
        
        # Track tool executions
        if (event.content_block and 
            event.content_block.tool_execution):
            tool_exec = event.content_block.tool_execution
            self.tool_executions[tool_exec.tool_id] = tool_exec
    
    def _format_timestamp(self, timestamp: datetime) -> str:
        """
        Format timestamp for display.
        
        Args:
            timestamp: The datetime to format
            
        Returns:
            Formatted timestamp string
        """
        if self.config.show_timestamps:
            return f"[{self.colors['timestamp']}]{timestamp.strftime('%H:%M:%S')}[/]"
        return ""
    
    def _format_token_count(self, tokens: int) -> str:
        """
        Format token count for display.
        
        Args:
            tokens: Number of tokens
            
        Returns:
            Formatted token count string
        """
        if self.config.show_token_counts:
            return f"[{self.colors['tokens']}]({tokens} tokens)[/]"
        return ""
    
    def _format_content_preview(self, content: str) -> str:
        """
        Format content preview with length limit.
        
        Args:
            content: The content to preview
            
        Returns:
            Formatted and truncated content preview
        """
        if not self.config.show_content_preview:
            return ""
        
        max_len = self.config.max_content_preview_length
        if len(content) > max_len:
            preview = content[:max_len] + "..."
        else:
            preview = content
        
        # Escape Rich markup and replace newlines
        preview = preview.replace("[", r"\[").replace("\n", " ")
        return f'"{preview}"'
    
    def _create_message_panel(self, event: MCPEvent) -> Panel:
        """
        Create a Rich Panel for displaying a single message event.
        
        Args:
            event: The MCP event to display
            
        Returns:
            Rich Panel containing the formatted event
        """
        # First validate the MCP event is properly formatted
        try:
            event = MCPEvent.model_validate(event)
        except ValidationError as e:
            print(f'Error: {str(e)}')

        # Determine the main content
        if event.role:
            role_value = _get_enum_value(event.role)
            role_color = self.colors.get(role_value, 'white')
            role_text = f"[{role_color}]{role_value.upper()}[/]"
        else:
            role_text = f"[{self.colors['system']}]SYSTEM[/]"
        
        # Build the content lines
        content_lines = []
        
        # Header with timestamp and role
        timestamp = self._format_timestamp(event.timestamp)
        header = f"{timestamp} {role_text}"
        
        if event.content_block:
            block = event.content_block
            content_type_value = _get_enum_value(block.content_type)
            content_type_color = self.colors.get(content_type_value, 'white')
            token_count = self._format_token_count(block.token_usage.estimated_tokens)
            
            # Content type and tokens
            content_header = f"[{content_type_color}]{content_type_value.upper()}[/] {token_count}"
            
            # Content preview
            if block.content_type == ContentType.TEXT:
                preview = self._format_content_preview(block.raw_content)
                content_lines.append(f"  📝 {content_header}")
                if preview:
                    content_lines.append(f"     {preview}")
            
            elif block.content_type == ContentType.TOOL_USE:
                if block.tool_execution:
                    tool_name = block.tool_execution.tool_name
                    content_lines.append(f"  🔧 {content_header}: {tool_name}")
                    # Show tool arguments preview
                    args_preview = str(block.tool_execution.arguments)[:50]
                    if len(str(block.tool_execution.arguments)) > 50:
                        args_preview += "..."
                    content_lines.append(f"     Args: {args_preview}")
                else:
                    content_lines.append(f"  🔧 {content_header}")
            
            elif block.content_type == ContentType.TOOL_RESULT:
                if block.tool_execution:
                    status = "✅" if block.tool_execution.success else "❌"
                    duration = ""
                    if block.tool_execution.duration_seconds:
                        duration = f" ({block.tool_execution.duration_seconds:.1f}s)"
                    content_lines.append(f"  {status} {content_header}{duration}")
                    
                    # Show result preview or error
                    if block.tool_execution.success and block.tool_execution.result_content:
                        result_preview = self._format_content_preview(block.tool_execution.result_content)
                        content_lines.append(f"     Result: {result_preview}")
                    elif block.tool_execution.error_message:
                        error_preview = self._format_content_preview(block.tool_execution.error_message)
                        content_lines.append(f"     [red]Error: {error_preview}[/red]")
                else:
                    content_lines.append(f"  📄 {content_header}")
        
        else:
            # Event without content block
            event_type_value = _get_enum_value(event.event_type)
            event_type_text = event_type_value.replace('_', ' ').title()
            content_lines.append(f"  ℹ️  {event_type_text}")
        
        # Combine header and content
        if content_lines:
            full_content = header + "\n" + "\n".join(content_lines)
        else:
            full_content = header
        
        # Create panel with appropriate border color
        border_style = self.colors.get(_get_enum_value(event.role) if event.role else 'system', 'white')
        
        return Panel(
            full_content,
            border_style=border_style,
            padding=(0, 1),
            expand=False
        )
    
    def _create_summary_panel(self) -> Panel:
        """
        Create a summary panel showing conversation statistics.
        
        Returns:
            Rich Panel with conversation summary
        """
        total_events = len(self.events)
        active_convs = len(self.active_conversations)
        total_tools = len([e for e in self.events if e.content_block and 
                          e.content_block.content_type == ContentType.TOOL_USE])
        
        # Calculate total tokens
        total_tokens = sum(
            e.content_block.token_usage.estimated_tokens 
            for e in self.events 
            if e.content_block
        )
        
        # Runtime
        runtime = datetime.now() - self.start_time
        runtime_str = str(runtime).split('.')[0]  # Remove microseconds
        
        summary_text = (
            f"[bold]MCP Conversation Monitor[/bold]\n"
            f"Runtime: [cyan]{runtime_str}[/cyan] | "
            f"Events: [yellow]{total_events}[/yellow] | "
            f"Active Conversations: [green]{active_convs}[/green] | "
            f"Tool Calls: [magenta]{total_tools}[/magenta] | "
            f"Total Tokens: [bright_yellow]{total_tokens}[/bright_yellow]"
        )
        
        return Panel(
            summary_text,
            title="📊 Status",
            border_style="bright_blue",
            padding=(0, 1)
        )
    
    def _create_layout(self) -> Layout:
        """
        Create the main layout for the live display.
        
        Returns:
            Rich Layout containing all display components
        """
        layout = Layout()
        
        # Split into header and main content
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main")
        )
        
        # Header contains the summary
        layout["header"].update(self._create_summary_panel())
        
        # Main content shows recent events
        recent_events = self.events[-10:]  # Show last 10 events
        if recent_events:
            panels = [self._create_message_panel(event) for event in recent_events]
            main_content = Align.left(Columns(panels, expand=True))
        else:
            main_content = Align.center("Waiting for MCP events...", vertical="middle")
        
        layout["main"].update(main_content)
        
        return layout
    
    def start_live_display(self) -> None:
        """
        Start the live terminal display.
        
        This begins real-time rendering of MCP events in the terminal.
        Events added via add_event() will automatically appear.
        """
        if self.live_display is not None:
            return  # Already started
        
        self.live_display = Live(
            self._create_layout(),
            console=self.console,
            refresh_per_second=4,
            auto_refresh=True,
            screen=False
        )
        self.live_display.start()
    
    def stop_live_display(self) -> None:
        """Stop the live terminal display."""
        if self.live_display:
            self.live_display.stop()
            self.live_display = None
    
    def update_display(self) -> None:
        """
        Manually trigger a display update.
        
        This is called automatically when using live display, but can be
        called manually for testing or custom refresh logic.
        """
        if self.live_display:
            self.live_display.update(self._create_layout())
    
    def print_event(self, event: MCPEvent) -> None:
        """
        Print a single event to the console (non-live mode).
        
        Args:
            event: The MCP event to print
        """
        panel = self._create_message_panel(event)
        self.console.print(panel)
    
    def print_summary(self) -> None:
        """Print conversation summary to the console."""
        summary_panel = self._create_summary_panel()
        self.console.print(summary_panel)
    
    def print_separator(self) -> None:
        """Print a visual separator line."""
        self.console.print(Rule(style=self.colors['separator']))


def create_display(config: Optional[MCPDisplayConfig] = None) -> MCPTerminalDisplay:
    """
    Factory function to create an MCP terminal display.
    
    Args:
        config: Optional display configuration
        
    Returns:
        Configured MCPTerminalDisplay instance
        
    Example:
        >>> display = create_display()
        >>> display.start_live_display()
        >>> # Events will now appear in real-time
    """
    return MCPTerminalDisplay(config)