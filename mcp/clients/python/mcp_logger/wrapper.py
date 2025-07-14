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
                    raw_content = str(content)
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
        result_content: List[Any],
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
            tool_execution.result_content = json.dumps(result_content, indent=2)
        
        # Create content block for the result
        raw_content = json.dumps(result_content, indent=2) if success else error_message or "Tool execution failed"
        
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
            role=Role.SYSTEM,
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
            
            # List available tools for this session
            response = await session.list_tools()
            tools = response.tools
            
            for tool in tools:
                self.tool_to_session[tool.name] = session
                self.available_tools.append({
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema
                })
                
            print(f"Connected to {server_name} with tools: {[t.name for t in tools]}")
            
        except Exception as e:
            print(f"Failed to connect to {server_name}: {e}")
            raise
    
    async def connect_to_servers(self) -> None:
        """Connect to all configured MCP servers."""
        try:
            with open("server_config.json", "r") as file:
                data = json.load(file)
            
            servers = data.get("mcpServers", {})
            
            for server_name, server_config in servers.items():
                await self.connect_to_server(server_name, server_config)
                
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
                        role=Role.SYSTEM,
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
        print("MCP Chatbot with Logging Started!")
        
        if self.display:
            self.display.start_live_display()
        
        print("Type your queries or 'quit' to exit.")
        
        try:
            while True:
                query = input("\nQuery: ").strip()
                
                if query.lower() == 'quit':
                    break
                
                await self.process_query(query)
                print("\n")
                
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
        max_content_preview_length=60
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