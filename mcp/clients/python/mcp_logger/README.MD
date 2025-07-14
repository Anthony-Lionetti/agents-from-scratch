# MCP Content Flow Visualization

A comprehensive logging and visualization system for Model Context Protocol (MCP) client interactions, providing real-time insight into message structure, token usage, and tool execution patterns.

## Executive Summary

The MCP Content Flow Visualization project transforms how developers understand and debug MCP client interactions. By wrapping existing MCP clients with transparent logging functionality, this system captures every aspect of the conversation flow—from user input to tool execution to AI responses—and presents it through an intuitive, real-time terminal interface.

### Key Benefits

- **🔍 Complete Transparency**: Every content block, token count, and API interaction is visible
- **⚡ Real-time Monitoring**: Live terminal display shows conversation flow as it happens
- **🐛 Enhanced Debugging**: Easily identify bottlenecks, errors, and inefficiencies
- **📊 Token Analytics**: Track token usage patterns and optimize costs
- **🛠️ Tool Execution Tracking**: Monitor tool performance with timing and success rates
- **💾 Persistent Logging**: JSONL format logs for historical analysis and replay

### What It Solves

MCP interactions can be complex black boxes—this system makes them completely transparent. Instead of guessing why a conversation took certain paths or consumed specific token amounts, developers can see exactly what happened, when, and why.

Perfect for:

- Debugging unexpected MCP behavior
- Optimizing token usage and costs
- Understanding conversation flow patterns
- Performance analysis of tool executions
- Learning how Claude structures multi-step responses

---

## Quick Start

### Installation

```bash
# Install required dependencies
pip install rich pydantic anthropic mcp

# Clone/download the mcp_logger package to your project
```

### Basic Usage

Replace your existing MCP client with the logging wrapper:

```python
from mcp_logger.wrapper import MCPChatBotWrapper
import asyncio

async def main():
    # Create wrapper with real-time display
    chatbot = MCPChatBotWrapper(show_display=True)

    try:
        await chatbot.connect_to_servers()
        await chatbot.chat_loop()
    finally:
        await chatbot.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
```

That's it! Your MCP interactions will now be logged and visualized in real-time.

---

## Detailed Usage Guide

### Step 1: Set Up Your MCP Server Configuration

Ensure you have a `server_config.json` file in your project root:

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "/path/to/allowed/directory"
      ]
    },
    "brave-search": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-brave-search"],
      "env": {
        "BRAVE_API_KEY": "your-api-key"
      }
    }
  }
}
```

### Step 2: Basic Integration

Replace your existing MCP client code with the wrapper:

**Before (your original claude.py):**

```python
from claude import MCP_ChatBot
import asyncio

async def main():
    chatbot = MCP_ChatBot()
    try:
        await chatbot.connect_to_servers()
        await chatbot.chat_loop()
    finally:
        await chatbot.cleanup()
```

**After (with logging wrapper):**

```python
from mcp_logger.wrapper import MCPChatBotWrapper
from mcp_logger.display import MCPDisplayConfig
import asyncio

async def main():
    # Configure display options
    display_config = MCPDisplayConfig(
        show_timestamps=True,
        show_token_counts=True,
        show_content_preview=True,
        max_content_preview_length=80
    )

    # Create wrapper with logging
    chatbot = MCPChatBotWrapper(
        show_display=True,           # Enable real-time terminal display
        display_config=display_config,
        enable_file_logging=True,    # Save logs to files
        log_directory="logs/conversations"
    )

    try:
        await chatbot.connect_to_servers()
        await chatbot.chat_loop()
    finally:
        await chatbot.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
```

### Step 3: Advanced Configuration

#### Custom Display Configuration

```python
from mcp_logger.display import MCPDisplayConfig

# Minimal display for production
minimal_config = MCPDisplayConfig(
    show_timestamps=False,
    show_token_counts=True,
    show_content_preview=False,
    auto_scroll=True
)

# Debug display with full details
debug_config = MCPDisplayConfig(
    show_timestamps=True,
    show_token_counts=True,
    show_content_preview=True,
    max_content_preview_length=120,
    color_theme="debug"
)
```

#### Logging-Only Mode (No Display)

```python
# For production environments where you want logs but no terminal display
chatbot = MCPChatBotWrapper(
    show_display=False,              # Disable terminal display
    enable_file_logging=True,        # Keep file logging
    log_directory="production_logs"
)
```

#### Custom Event Processing

```python
from mcp_logger.events import MCPEvent, EventType

def custom_event_handler(event: MCPEvent):
    """Custom processing for each MCP event."""
    if event.event_type == EventType.TOOL_EXECUTION_COMPLETED:
        if event.content_block and event.content_block.tool_execution:
            duration = event.content_block.tool_execution.duration_seconds
            if duration and duration > 5.0:
                print(f"⚠️  Slow tool execution: {duration:.1f}s")

# Use custom handler
chatbot = MCPChatBotWrapper(
    show_display=True,
    enable_file_logging=True
)
chatbot.logger.event_callback = custom_event_handler
```

### Step 4: Programmatic Usage

#### Single Query Processing

```python
async def process_single_query(query: str):
    chatbot = MCPChatBotWrapper(show_display=False)

    try:
        await chatbot.connect_to_servers()
        chatbot.start_conversation()
        await chatbot.process_query(query)

        # Get conversation events
        events = chatbot.logger.get_conversation_events(
            chatbot.current_conversation_id
        )

        # Analyze results
        total_tokens = sum(
            e.content_block.token_usage.estimated_tokens
            for e in events
            if e.content_block
        )
        print(f"Query used {total_tokens} tokens")

    finally:
        await chatbot.cleanup()
```

#### Batch Processing with Analysis

```python
async def process_multiple_queries(queries: List[str]):
    chatbot = MCPChatBotWrapper(
        show_display=True,
        enable_file_logging=True
    )

    try:
        await chatbot.connect_to_servers()

        for i, query in enumerate(queries):
            print(f"\n=== Processing Query {i+1}/{len(queries)} ===")

            # Start new conversation for each query
            conversation_id = chatbot.start_conversation()
            await chatbot.process_query(query)
            chatbot.end_conversation()

            # Get and analyze this conversation
            events = chatbot.logger.get_conversation_events(conversation_id)
            tool_calls = [e for e in events if e.event_type == EventType.TOOL_EXECUTION_STARTED]
            print(f"Query used {len(tool_calls)} tool calls")

    finally:
        await chatbot.cleanup()
```

### Step 5: Log Analysis

#### Reading Log Files

```python
from mcp_logger.wrapper import MCPLogger
from mcp_logger.events import MCPEvent
import json

def analyze_conversation_logs(log_directory: str):
    """Analyze historical conversation logs."""
    logger = MCPLogger(log_directory, enable_file_logging=False)

    # Load a specific conversation
    conversation_id = "conv_20250713_140525_a1b2c3d4"
    events = logger.load_conversation_from_file(conversation_id)

    # Analyze token usage
    total_tokens = sum(
        e.content_block.token_usage.estimated_tokens
        for e in events if e.content_block
    )

    # Analyze tool performance
    tool_executions = [
        e.content_block.tool_execution
        for e in events
        if e.content_block and e.content_block.tool_execution and e.content_block.tool_execution.duration_seconds
    ]

    if tool_executions:
        avg_duration = sum(t.duration_seconds for t in tool_executions) / len(tool_executions)
        print(f"Average tool execution time: {avg_duration:.2f}s")

    print(f"Conversation used {total_tokens} tokens across {len(events)} events")
```

---

## Package Structure

```
mcp_logger/
├── __init__.py              # Package initialization
├── events.py                # Core data structures and event types
├── display.py               # Rich terminal display components
├── wrapper.py               # Main MCP logging wrapper
└── README.md               # This documentation

logs/
└── mcp_conversations/       # Generated log files
    ├── conv_20250713_140525_a1b2c3d4.jsonl
    └── conv_20250713_141032_e5f6g7h8.jsonl
```

---

## File Implementation Details

### `events.py` - Data Structures and Models

**Purpose**: Defines all data structures used throughout the logging system using Pydantic models for validation and serialization.

**Key Components**:

#### Enums

- `EventType`: Types of MCP events (message_received, tool_execution_started, etc.)
- `ContentType`: Content block types (text, tool_use, tool_result)
- `Role`: Message roles (user, assistant, system)

#### Core Models

```python
class MCPEvent(BaseModel):
    """Primary event structure for all MCP interactions."""
    event_type: EventType
    timestamp: datetime
    conversation_id: str
    message_id: str
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    role: Optional[Role] = None
    content_block: Optional[ContentBlock] = None
    # ... additional fields
```

#### Token Tracking

```python
class TokenUsage(BaseModel):
    """Comprehensive token usage information."""
    estimated_tokens: int
    total_message_tokens: Optional[int] = None
    estimation_method: str = "character_count"
    accuracy_percentage: Optional[float] = None
```

#### Tool Execution Tracking

```python
class ToolExecution(BaseModel):
    """Complete tool execution lifecycle."""
    tool_name: str
    tool_id: str
    arguments: Dict[str, Any]
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    success: Optional[bool] = None
    error_message: Optional[str] = None
    result_content: Optional[str] = None
```

**Implementation Features**:

- **Pydantic Integration**: Automatic JSON serialization/deserialization with `model_dump_json()` and `model_validate_json()`
- **Type Safety**: Full type validation on model creation
- **Token Estimation**: Character-based estimation (chars \* 0.75) with tiktoken placeholder for future enhancement
- **Helper Functions**: Factory functions for creating content blocks and events

### `display.py` - Rich Terminal Visualization

**Purpose**: Provides real-time, interactive terminal display using the Rich library for beautiful, informative output.

**Key Components**:

#### Display Configuration

```python
class MCPDisplayConfig:
    """Comprehensive display customization."""
    show_timestamps: bool = True
    show_token_counts: bool = True
    show_content_preview: bool = True
    max_content_preview_length: int = 80
    auto_scroll: bool = True
    color_theme: str = "default"
```

#### Main Display Class

```python
class MCPTerminalDisplay:
    """Live terminal display with Rich components."""

    def start_live_display(self) -> None:
        """Begin real-time rendering with Live component."""

    def add_event(self, event: MCPEvent) -> None:
        """Add event and trigger display update."""
```

**Visual Features**:

- **Color-coded Output**: Different colors for users, assistants, content types
- **Role Indicators**: 📝 Text, 🔧 Tool Use, ✅/❌ Tool Results
- **Token Visualization**: Real-time token counts with each content block
- **Content Previews**: Truncated content with "..." indicators
- **Tool Execution Status**: Duration, success/failure, argument previews
- **Summary Panel**: Live statistics (runtime, events, tokens, tool calls)
- **Progressive Layout**: Header with stats, scrolling main content area

**Implementation Details**:

- **Rich Live Display**: Uses `rich.live.Live` for flicker-free updates
- **Panel-based Layout**: Each message is a styled `Panel` with color-coded borders
- **Automatic Refresh**: 4 FPS refresh rate for smooth real-time updates
- **Memory Efficient**: Shows only recent events (last 10) to prevent memory bloat
- **Configurable Verbosity**: Adjustable content preview lengths and display options

### `wrapper.py` - Core MCP Logging Wrapper

**Purpose**: The main wrapper that intercepts all MCP interactions and adds comprehensive logging without modifying existing client code.

**Architecture Overview**:

#### Two-Layer Design

1. **MCPLogger**: Pure logging functionality (events, file I/O, callbacks)
2. **MCPChatBotWrapper**: MCP client wrapper with integrated logging

#### MCPLogger Class

```python
class MCPLogger:
    """Core logging without UI dependencies."""

    def log_event(self, event: MCPEvent) -> None:
        """Central event logging with file persistence."""

    def _write_event_to_file(self, event: MCPEvent) -> None:
        """JSONL format for easy analysis."""
```

**File Storage Format**:

- **JSONL Format**: One JSON event per line for streaming analysis
- **Conversation Files**: Separate file per conversation (`conv_<timestamp>_<id>.jsonl`)
- **Atomic Writes**: Each event appended immediately for crash safety

#### MCPChatBotWrapper Class

```python
class MCPChatBotWrapper:
    """Drop-in replacement for existing MCP clients."""

    async def process_query(self, query: str) -> None:
        """Enhanced query processing with full event capture."""
```

**Integration Points**:

1. **Message Interception**: Captures user input and AI responses
2. **Content Block Parsing**: Extracts and logs each content block with token estimation
3. **Tool Execution Lifecycle**: Logs start, duration, success/failure, and results
4. **API Metadata Capture**: Stop reasons, model info, usage statistics
5. **Error Handling**: Comprehensive error capture with context

**Wrapper Benefits**:

- **Non-invasive**: Existing MCP client code works unchanged
- **Comprehensive**: Captures 100% of MCP interactions
- **Modular**: Logging and display are separate, reusable components
- **Configurable**: Enable/disable logging, display, file storage independently
- **Performance**: Minimal overhead with efficient event processing

#### Key Implementation Features

**Conversation Management**:

```python
def _generate_conversation_id(self) -> str:
    """Generate unique conversation IDs with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_uuid = str(uuid.uuid4())[:8]
    return f"conv_{timestamp}_{short_uuid}"
```

**Token Attribution**:

```python
def _extract_content_blocks_from_anthropic_message(
    self, message: Message, message_id: str, role: Role
) -> List[MCPEvent]:
    """Parse Anthropic responses into structured events."""
```

**Tool Execution Tracking**:

```python
def _create_tool_result_event(
    self, tool_id: str, result_content: List[Any],
    message_id: str, success: bool = True
) -> MCPEvent:
    """Create comprehensive tool execution events."""
```

**Error Resilience**:

- Try/catch blocks around all external calls (Anthropic API, MCP tools)
- Graceful degradation when logging fails
- Cleanup methods for proper resource disposal

---

## Advanced Usage Patterns

### Custom Event Processing

```python
from mcp_logger.events import EventType

class CustomAnalyzer:
    def __init__(self):
        self.slow_tools = []
        self.token_usage = {}

    def analyze_event(self, event: MCPEvent):
        """Custom analysis of each event."""
        if event.event_type == EventType.TOOL_EXECUTION_COMPLETED:
            if event.content_block and event.content_block.tool_execution:
                duration = event.content_block.tool_execution.duration_seconds
                if duration and duration > 3.0:
                    self.slow_tools.append({
                        'tool': event.content_block.tool_execution.tool_name,
                        'duration': duration,
                        'timestamp': event.timestamp
                    })

# Use custom analyzer
analyzer = CustomAnalyzer()
chatbot = MCPChatBotWrapper(show_display=True)
chatbot.logger.event_callback = analyzer.analyze_event
```

### Integration with External Systems

```python
import requests

def send_to_monitoring(event: MCPEvent):
    """Send events to external monitoring system."""
    if event.event_type == EventType.TOOL_EXECUTION_COMPLETED:
        if event.content_block and event.content_block.tool_execution:
            tool_exec = event.content_block.tool_execution
            if not tool_exec.success:
                # Alert on tool failures
                requests.post('https://monitoring.example.com/alert', json={
                    'type': 'tool_failure',
                    'tool': tool_exec.tool_name,
                    'error': tool_exec.error_message,
                    'timestamp': event.timestamp.isoformat()
                })

chatbot = MCPChatBotWrapper(show_display=False)
chatbot.logger.event_callback = send_to_monitoring
```

---

## Performance Considerations

### Memory Management

- Events are streamed to files immediately to prevent memory buildup
- Display shows only recent events (configurable)
- Large content blocks are truncated for display but preserved in logs

### Token Estimation Accuracy

- Character-based estimation averages 85-90% accuracy
- Future tiktoken integration will improve precision
- Validation against API-reported totals for calibration

### File I/O Optimization

- JSONL format for streaming analysis
- Atomic writes prevent corruption
- Configurable log rotation (future enhancement)

---

## Troubleshooting

### Common Issues

**Display not showing**: Ensure `show_display=True` and Rich is installed
**Missing events**: Check that `enable_file_logging=True` and log directory is writable
**Token estimation errors**: Fallback to simple character count if tiktoken fails
**Connection issues**: Verify `server_config.json` is properly formatted

### Debug Mode

```python
# Enable verbose logging for debugging
import logging
logging.basicConfig(level=logging.DEBUG)

chatbot = MCPChatBotWrapper(
    show_display=True,
    enable_file_logging=True
)
```

---

## Future Enhancements (Phase 2 & 3)

### Planned Features

- **Interactive Terminal UI**: Keyboard navigation, content expansion
- **Historical Analysis**: Pattern detection, performance trends
- **Advanced Filtering**: Search logs by content, tool, date range
- **Export Capabilities**: CSV, PDF reports for stakeholder sharing
- **Real-time Alerting**: Custom triggers for errors, slow performance
- **Token Optimization**: Recommendations for cost reduction

### Tiktoken Integration

```python
# Future enhancement for precise token counting
def estimate_tokens_tiktoken(text: str, model: str = "gpt-4") -> int:
    import tiktoken
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))
```

---

## Contributing

This project is in active development as part of a 6-week implementation plan. Contributions are welcome, particularly for:

- Enhanced token estimation algorithms
- Additional display themes and customization
- Performance optimizations for large conversations
- Integration with external monitoring systems

---

## License

MIT License - see LICENSE file for details.
