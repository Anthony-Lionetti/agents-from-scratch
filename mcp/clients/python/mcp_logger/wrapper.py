import json
import asyncio
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from contextlib import AsyncExitStack


from anthropic import Anthropic
from anthropic.types import Message, ContentBlock
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.tree import Tree
from rich.rule import Rule

from .events import (
    MCPEvent, EventType, Role, ContentType, TokenUsage, ToolExecution,
    create_content_block, create_mcp_event, estimate_tokens_from_text
)
from .display import MCPTerminalDisplay, MCPDisplayConfig, create_display


class MCPLogger:
    """
    Core logging functionality for MCP interactions.
    
    Handles event creation, file storage, and event callbacks without
    coupling to specific display implementations.
    """
    
    def __init__(
        self,
        log_directory: str = "logs/mcp_conversations",
        enable_file_logging: bool = True,
        event_callback: Optional[Callable[[MCPEvent], None]] = None
    ):
        """
        Initialize the MCP logger.
        
        Args:
            log_directory: Directory to store conversation log files
            enable_file_logging: Whether to write events to JSON files
            event_callback: Optional callback function called for each event
        """
        self.log_directory = Path(log_directory)
        self.enable_file_logging = enable_file_logging
        self.event_callback = event_callback
        
        # Create log directory
        if self.enable_file_logging:
            self.log_directory.mkdir(parents=True, exist_ok=True)
        
        # State tracking
        self.conversation_logs: Dict[str, List[MCPEvent]] = {}
        self.active_tool_executions: Dict[str, ToolExecution] = {}
    
    def log_event(self, event: MCPEvent) -> None:
        """
        Log an MCP event.
        
        Args:
            event: The MCP event to log
        """
        # Add to conversation log
        if event.conversation_id not in self.conversation_logs:
            self.conversation_logs[event.conversation_id] = []
        self.conversation_logs[event.conversation_id].append(event)
        
        # Write to file if enabled
        if self.enable_file_logging:
            self._write_event_to_file(event)
        
        # Call event callback if provided
        if self.event_callback:
            self.event_callback(event)
    
    def _write_event_to_file(self, event: MCPEvent) -> None:
        """Write a single event to the conversation log file."""
        log_file = self.log_directory / f"{event.conversation_id}.jsonl"
        
        try:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(event.model_dump_json() + '\n')
        except Exception as e:
            print(f"Failed to write event to log file: {e}")
    
    def get_conversation_events(self, conversation_id: str) -> List[MCPEvent]:
        """Get all events for a specific conversation."""
        return self.conversation_logs.get(conversation_id, [])
    
    def load_conversation_from_file(self, conversation_id: str) -> List[MCPEvent]:
        """Load conversation events from log file."""
        log_file = self.log_directory / f"{conversation_id}.jsonl"
        events = []
        
        if log_file.exists():
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            event_data = json.loads(line)
                            event = MCPEvent.model_validate(event_data)
                            events.append(event)
            except Exception as e:
                print(f"Failed to load conversation from file: {e}")
        
        return events


class MCPChatBotWrapper:
    """
    Wrapper around MCP ChatBot that adds comprehensive logging and visualization.
    
    This class wraps the core MCP functionality while intercepting all interactions
    to provide real-time logging, token tracking, and terminal visualization.
    """
    
    def __init__(
        self,
        show_display: bool = True,
        display_config: Optional[MCPDisplayConfig] = None,
        log_directory: str = "logs/mcp_conversations",
        enable_file_logging: bool = True
    ):
        """
        Initialize the MCP ChatBot wrapper.
        
        Args:
            show_display: Whether to show the live terminal display
            display_config: Configuration for the terminal display
            log_directory: Directory for conversation log files
            enable_file_logging: Whether to write logs to files
        """
        # Core MCP components (similar to your original claude.py)
        self.sessions: List[ClientSession] = []
        self.exit_stack = AsyncExitStack()
        self.anthropic = Anthropic()
        self.available_tools: List[Dict[str, Any]] = []
        self.tool_to_session: Dict[str, ClientSession] = {}
        
        # Server information tracking
        self.connected_servers: Dict[str, Dict[str, Any]] = {}
        
        # Logging and display components
        self.logger = MCPLogger(
            log_directory=log_directory,
            enable_file_logging=enable_file_logging,
            event_callback=self._on_event if show_display else None
        )
        
        self.display: Optional[MCPTerminalDisplay] = None
        if show_display:
            self.display = create_display(display_config)
        
        # Conversation tracking
        self.current_conversation_id: Optional[str] = None
        self.message_counter = 0
    
    def _on_event(self, event: MCPEvent) -> None:
        """Event callback to update the display."""
        if self.display:
            self.display.add_event(event)
            self.display.update_display()
    
    def _generate_conversation_id(self) -> str:
        """Generate a unique conversation ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        short_uuid = str(uuid.uuid4())[:8]
        return f"conv_{timestamp}_{short_uuid}"
    
    def _generate_message_id(self) -> str:
        """Generate a unique message ID."""
        self.message_counter += 1
        return f"msg_{self.current_conversation_id}_{self.message_counter:03d}"
    
    def _extract_content_blocks_from_anthropic_message(
        self,
        message: Message,
        message_id: str,
        role: Role
    ) -> List[MCPEvent]:
        """
        Extract content blocks from an Anthropic message and create events.
        
        Args:
            message: The Anthropic message object
            message_id: Unique message identifier
            role: The role (USER or ASSISTANT)
            
        Returns:
            List of MCPEvent objects for each content block
        """
        events = []
        
        for block_index, content in enumerate(message.content):
            if hasattr(content, 'type'):
                # Determine content type and extract content

                # set type and content based on content type
                if content.type == 'text':
                    content_type = ContentType.TEXT
                    raw_content = content.text
                    tool_execution = None
                    
                elif content.type == 'tool_use':
                    content_type = ContentType.TOOL_USE
                    raw_content = json.dumps({
                        'tool': content.name,
                        'arguments': content.input
                    }, indent=2)
                    
                    # Create tool execution object
                    tool_execution = ToolExecution(
                        tool_name=content.name,
                        tool_id=content.id,
                        arguments=content.input,
                        start_time=datetime.now()
                    )
                    
                    # Track this tool execution
                    self.logger.active_tool_executions[content.id] = tool_execution
                
                else:
                    # Handle other content types
                    content_type = ContentType.TEXT
                    raw_content = "Content of type: " + str(content.type)
                    tool_execution = None
                
                # Create content block
                content_block = create_content_block(
                    block_index=block_index,
                    content_type=content_type,
                    raw_content=raw_content,
                    tool_execution=tool_execution
                )
                
                # Create and log event
                event = create_mcp_event(
                    event_type=EventType.CONTENT_BLOCK_PARSED,
                    conversation_id=self.current_conversation_id,
                    message_id=message_id,
                    role=role,
                    content_block=content_block,
                    stop_reason=getattr(message, 'stop_reason', None),
                    model=getattr(message, 'model', None),
                    api_usage_stats=getattr(message, 'usage', None).__dict__ if hasattr(message, 'usage') else None
                )
                
                events.append(event)
        
        return events
    
    def _create_tool_result_event(
        self,
        tool_id: str,
        result_content: List[ContentBlock],
        message_id: str,
        success: bool = True,
        error_message: Optional[str] = None
    ) -> MCPEvent:
        """Create an event for tool execution results."""
        
        # Get the original tool execution
        tool_execution = self.logger.active_tool_executions.get(tool_id)
        if tool_execution:
            # Update tool execution with results
            tool_execution.end_time = datetime.now()
            tool_execution.duration_seconds = (
                tool_execution.end_time - tool_execution.start_time
            ).total_seconds()
            tool_execution.success = success
            tool_execution.error_message = error_message
            tool_execution.result_content = str(result_content)
        
        # Create content block for the result
        raw_content = str(result_content)
        content_block = create_content_block(
            block_index=0,  # Tool results are typically single blocks
            content_type=ContentType.TOOL_RESULT,
            raw_content=raw_content,
            tool_execution=tool_execution
        )
        
        return create_mcp_event(
            event_type=EventType.TOOL_EXECUTION_COMPLETED,
            conversation_id=self.current_conversation_id,
            message_id=message_id,
            role=Role.USER,
            content_block=content_block
        )
    
    async def connect_to_server(self, server_name: str, server_config: dict) -> None:
        """Connect to a single MCP server with logging."""
        try:
            server_params = StdioServerParameters(**server_config)
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            read, write = stdio_transport
            session = await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            await session.initialize()
            self.sessions.append(session)
            
            # Initialize server info
            server_info = {
                "config": server_config,
                "session": session,
                "tools": [],
                "resources": [],
                "prompts": []
            }
            
            # List available tools for this session
            try:
                tools_response = await session.list_tools()
                tools = tools_response.tools
                
                for tool in tools:
                    self.tool_to_session[tool.name] = session
                    tool_info = {
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.inputSchema
                    }
                    self.available_tools.append(tool_info)
                    server_info["tools"].append(tool_info)
                    
            except Exception as e:
                print(f"Warning: Could not list tools for {server_name}: {e}")
            
            # List available resources for this session
            try:
                resources_response = await session.list_resources()
                resources = resources_response.resources
                
                for resource in resources:
                    resource_info = {
                        "uri": resource.uri,
                        "name": resource.name,
                        "description": resource.description,
                        "mimeType": getattr(resource, 'mimeType', None)
                    }
                    server_info["resources"].append(resource_info)
                    
            except Exception as e:
                print(f"Warning: Could not list resources for {server_name}: {e}")
            
            # List available prompts for this session
            try:
                prompts_response = await session.list_prompts()
                prompts = prompts_response.prompts
                
                for prompt in prompts:
                    prompt_info = {
                        "name": prompt.name,
                        "description": prompt.description,
                        "arguments": getattr(prompt, 'arguments', [])
                    }
                    server_info["prompts"].append(prompt_info)
                    
            except Exception as e:
                print(f"Warning: Could not list prompts for {server_name}: {e}")
            
            # Store server information
            self.connected_servers[server_name] = server_info
        
        except Exception as e:
            print(f"Failed to connect to {server_name}: {e}")
            raise

    def _display_server_summary(self) -> None:
        """Display a comprehensive summary of all connected servers and their capabilities."""
        console = Console()
        
        if not self.connected_servers:
            console.print("[red]No servers connected.[/red]")
            return
        
        # Title
        console.print(Rule("[bold blue]MCP Server Connection Summary[/bold blue]", style="blue"))
        console.print()
        
        # Overview table
        overview_table = Table(title="Connected Servers Overview", show_header=True, header_style="bold magenta")
        overview_table.add_column("Server Name", style="cyan", no_wrap=True)
        overview_table.add_column("Tools", justify="center", style="green")
        overview_table.add_column("Resources", justify="center", style="yellow")
        overview_table.add_column("Prompts", justify="center", style="blue")
        overview_table.add_column("Status", justify="center", style="bright_green")
        
        for server_name, server_info in self.connected_servers.items():
            overview_table.add_row(
                server_name,
                str(len(server_info["tools"])),
                str(len(server_info["resources"])),
                str(len(server_info["prompts"])),
                "✅ Connected"
            )
        
        console.print(overview_table)
        console.print()
        
        # Detailed information for each server
        for server_name, server_info in self.connected_servers.items():
            # Server panel
            server_panel = Panel(
                self._create_server_details(server_name, server_info),
                title=f"[bold cyan]{server_name}[/bold cyan]",
                border_style="cyan",
                padding=(1, 2)
            )
            console.print(server_panel)
            console.print()
        
        console.print(Rule("[dim]End of Server Summary[/dim]", style="dim"))

    def _create_server_details(self, server_name: str, server_info: Dict[str, Any]) -> str:
        """Create detailed information text for a server."""
        details = []
        
        # Server configuration
        config = server_info.get("config", {})
        if "command" in config:
            details.append(f"[bold]Command:[/bold] {config['command']}")
        if "args" in config:
            details.append(f"[bold]Args:[/bold] {' '.join(config['args'])}")
        
        details.append("")  # Empty line
        
        # Tools section
        tools = server_info.get("tools", [])
        if tools:
            details.append("[bold green]🔧 Available Tools:[/bold green]")
            for tool in tools:
                details.append(f"  • [cyan]{tool['name']}[/cyan]")
                if tool.get('description'):
                    # Truncate long descriptions
                    desc = tool['description']
                    if len(desc) > 80:
                        desc = desc[:77] + "..."
                    details.append(f"    {desc}")
        else:
            details.append("[bold green]🔧 Available Tools:[/bold green] None")
        
        details.append("")  # Empty line
        
        # Resources section
        resources = server_info.get("resources", [])
        if resources:
            details.append("[bold yellow]📄 Available Resources:[/bold yellow]")
            for resource in resources:
                details.append(f"  • [cyan]{resource['name']}[/cyan]")
                if resource.get('description'):
                    desc = resource['description']
                    if len(desc) > 80:
                        desc = desc[:77] + "..."
                    details.append(f"    {desc}")
                if resource.get('uri'):
                    details.append(f"    URI: [dim]{resource['uri']}[/dim]")
                if resource.get('mimeType'):
                    details.append(f"    Type: [dim]{resource['mimeType']}[/dim]")
        else:
            details.append("[bold yellow]📄 Available Resources:[/bold yellow] None")
        
        details.append("")  # Empty line
        
        # Prompts section
        prompts = server_info.get("prompts", [])
        if prompts:
            details.append("[bold blue]💬 Available Prompts:[/bold blue]")
            for prompt in prompts:
                details.append(f"  • [cyan]{prompt['name']}[/cyan]")
                if prompt.get('description'):
                    desc = prompt['description']
                    if len(desc) > 80:
                        desc = desc[:77] + "..."
                    details.append(f"    {desc}")
                if prompt.get('arguments'):
                    args = [arg.get('name', 'arg') for arg in prompt['arguments']]
                    details.append(f"    Arguments: [dim]{', '.join(args)}[/dim]")
        else:
            details.append("[bold blue]💬 Available Prompts:[/bold blue] None")
        
        return "\n".join(details)

    async def connect_to_servers(self) -> None:
        """Connect to all configured MCP servers."""
        # Ensure live display is stopped during connection phase
        if self.display and self.display.live_display:
            self.display.stop_live_display()
        
        try:
            with open("server_config.json", "r") as file:
                data = json.load(file)
            
            servers = data.get("mcpServers", {})
            
            print("🔗 Connecting to MCP servers...")
            
            for server_name, server_config in servers.items():
                print(f"  • Connecting to {server_name}...")
                await self.connect_to_server(server_name, server_config)
                print(f"  ✅ {server_name} connected successfully")
                
            print("\n📊 Generating server connection summary...")
            # Display server summary after all connections are complete
            self._display_server_summary()
            
            print("\n🎉 All servers connected successfully!")
            input("\nPress Enter to start the chat interface...")
                
        except Exception as e:
            print(f"Error loading server config: {e}")
            raise
    
    async def process_query(self, query: str) -> None:
        """
        Process a user query with comprehensive logging.
        
        Args:
            query: The user's input query
        """
        # Start new conversation if needed
        if not self.current_conversation_id:
            self.start_conversation()
        
        # Log user message
        user_message_id = self._generate_message_id()
        user_event = create_mcp_event(
            event_type=EventType.MESSAGE_RECEIVED,
            conversation_id=self.current_conversation_id,
            message_id=user_message_id,
            role=Role.USER
        )
        
        # Create user content block
        user_content_block = create_content_block(
            block_index=0,
            content_type=ContentType.TEXT,
            raw_content=query
        )
        user_event.content_block = user_content_block
        
        self.logger.log_event(user_event)
        
        # Build messages for Anthropic API
        messages = [{'role': 'user', 'content': query}]
        
        # Call Anthropic API
        response = self.anthropic.messages.create(
            max_tokens=2024,
            model='claude-3-5-sonnet-20241022',
            tools=self.available_tools,
            messages=messages
        )
        
        # Process response and continue conversation loop
        process_query = True
        while process_query:
            assistant_message_id = self._generate_message_id()
            
            # Log assistant response content blocks
            assistant_events = self._extract_content_blocks_from_anthropic_message(
                response, assistant_message_id, Role.ASSISTANT
            )
            
            for event in assistant_events:
                self.logger.log_event(event)
            
            assistant_content = []
            
            for content in response.content:
                if content.type == 'text':
                    print(content.text)
                    assistant_content.append(content)
                    if len(response.content) == 1:
                        process_query = False
                
                elif content.type == 'tool_use':
                    assistant_content.append(content)
                    messages.append({'role': 'assistant', 'content': assistant_content})
                    
                    tool_id = content.id
                    tool_args = content.input
                    tool_name = content.name
                    
                    # Log tool execution start
                    tool_start_event = create_mcp_event(
                        event_type=EventType.TOOL_EXECUTION_STARTED,
                        conversation_id=self.current_conversation_id,
                        message_id=assistant_message_id,
                        role=Role.ASSISTANT,
                        metadata={
                            'tool_name': tool_name,
                            'tool_id': tool_id,
                            'arguments': tool_args
                        }
                    )
                    self.logger.log_event(tool_start_event)
                    
                    try:
                        # Execute tool
                        session = self.tool_to_session[tool_name]
                        result = await session.call_tool(tool_name, arguments=tool_args)
                        
                        # Log successful tool execution
                        tool_result_event = self._create_tool_result_event(
                            tool_id=tool_id,
                            result_content=result.content,
                            message_id=self._generate_message_id(),
                            success=True
                        )
                        self.logger.log_event(tool_result_event)
                        
                        # Add tool result to conversation
                        messages.append({
                            "role": "user",
                            "content": [{
                                "type": "tool_result",
                                "tool_use_id": tool_id,
                                "content": result.content
                            }]
                        })
                        
                    except Exception as e:
                        # Log failed tool execution
                        tool_result_event = self._create_tool_result_event(
                            tool_id=tool_id,
                            result_content=[],
                            message_id=self._generate_message_id(),
                            success=False,
                            error_message=str(e)
                        )
                        self.logger.log_event(tool_result_event)
                        
                        # Add error to conversation
                        messages.append({
                            "role": "user",
                            "content": [{
                                "type": "tool_result",
                                "tool_use_id": tool_id,
                                "content": f"Error: {str(e)}"
                            }]
                        })
                    
                    # Continue conversation
                    response = self.anthropic.messages.create(
                        max_tokens=2024,
                        model='claude-3-5-sonnet-20241022',
                        tools=self.available_tools,
                        messages=messages
                    )
                    
                    if len(response.content) == 1 and response.content[0].type == "text":
                        # Log final assistant response
                        final_events = self._extract_content_blocks_from_anthropic_message(
                            response, self._generate_message_id(), Role.ASSISTANT
                        )
                        for event in final_events:
                            self.logger.log_event(event)
                        
                        print(response.content[0].text)
                        process_query = False
    
    def start_conversation(self) -> str:
        """
        Start a new conversation.
        
        Returns:
            The new conversation ID
        """
        self.current_conversation_id = self._generate_conversation_id()
        self.message_counter = 0
        
        # Log conversation start
        start_event = create_mcp_event(
            event_type=EventType.CONVERSATION_STARTED,
            conversation_id=self.current_conversation_id,
            message_id="start",
            metadata={'start_time': datetime.now().isoformat()}
        )
        self.logger.log_event(start_event)
        
        return self.current_conversation_id
    
    def end_conversation(self) -> None:
        """End the current conversation."""
        if self.current_conversation_id:
            end_event = create_mcp_event(
                event_type=EventType.CONVERSATION_ENDED,
                conversation_id=self.current_conversation_id,
                message_id="end",
                metadata={'end_time': datetime.now().isoformat()}
            )
            self.logger.log_event(end_event)
            
            self.current_conversation_id = None
            self.message_counter = 0
    
    async def chat_loop(self) -> None:
        """Run an interactive chat loop with logging and display."""
        print("🤖 MCP Chatbot with Logging Started!")
        print("Type your queries or 'quit' to exit.\n")
        
        try:
            while True:
                # Always stop live display during input to allow typing
                if self.display and self.display.live_display:
                    self.display.stop_live_display()
                
                # Clear screen and show current state
                if self.display:
                    self.display.console.clear()
                    self.display.print_summary()
                    self.display.print_separator()

                    ## Trying to isoltate issues ##
                    try:
                        events_trunc = self.display.events[-5:]
                    except Exception as e:
                        print("Yup that was the issue!")
                        events_trunc = self.display.events
                    ## Trying to isoltate issues ##

                    
                    # Show recent events
                    recent_events = events_trunc if self.display.events else []
                    for event in recent_events:
                        self.display.print_event(event)
                    
                    if recent_events:
                        self.display.print_separator()
                
                # Get user input
                query = input("🤖 Query: ").strip()
                
                if query.lower() == 'quit':
                    break
                
                # Start live display for processing the user query
                # This will run throughout the entire query processing including tool executions
                if self.display:
                    self.display.start_live_display()
                
                await self.process_query(query)
                
                # Stop live display after query processing is complete
                if self.display and self.display.live_display:
                    self.display.stop_live_display()
                
        except KeyboardInterrupt:
            print("\nExiting...")
        except Exception as e:
            print(f"\nError: {str(e)}")
        finally:
            self.end_conversation()
            if self.display:
                self.display.stop_live_display()
    
    async def cleanup(self) -> None:
        """Clean up all resources."""
        if self.display:
            self.display.stop_live_display()
        await self.exit_stack.aclose()


async def main():
    """Example usage of the MCP ChatBot wrapper."""
    # Configure display
    display_config = MCPDisplayConfig(
        show_timestamps=True,
        show_token_counts=True,
        show_content_preview=True,
        max_content_preview_length=1000
    )
    
    # Create wrapper with logging and display
    chatbot = MCPChatBotWrapper(
        show_display=True,
        display_config=display_config,
        enable_file_logging=True
    )
    
    try:
        await chatbot.connect_to_servers()
        await chatbot.chat_loop()
    finally:
        await chatbot.cleanup()


if __name__ == "__main__":
    asyncio.run(main())