from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from enum import Enum
import uuid

class EventType(str, Enum):
    """Types of MCP events we track."""
    MESSAGE_RECEIVED = "message_received"
    CONTENT_BLOCK_PARSED = "content_block_parsed"
    TOOL_EXECUTION_STARTED = "tool_execution_started"
    TOOL_EXECUTION_COMPLETED = "tool_execution_completed"
    MESSAGE_SENT_TO_USER = "message_sent_to_user"
    CONVERSATION_STARTED = "conversation_started"
    CONVERSATION_ENDED = "conversation_ended"



class ContentType(str, Enum):
    """
    Types of content blocks. Can be `text`, `tool_use`, or `tool_result`

    TODO: Extend this in the future to capture images and other types of content
    """
    TEXT = "text"
    TOOL_USE = "tool_use"
    TOOL_RESULT = "tool_result"
    DOCUMENT = "document"
    IMAGE = "image"


class Role(str, Enum):
    """Message roles. Note, there is no 'system' block for Anthropic's messages, only user and assistant"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class TokenUsage(BaseModel):
    """Token usage information."""
    estimated_tokens: int
    total_message_tokens: Optional[int] = None
    estimation_method: str = "character_count"  # "character_count", "tiktoken", "api_reported"
    accuracy_percentage: Optional[float] = None


class ToolExecution(BaseModel):
    """Tool execution details."""
    tool_name: str
    tool_id: str
    arguments: Dict[str, Any]
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    success: Optional[bool] = None
    error_message: Optional[str] = None
    result_content: Optional[str] = None


class ContentBlock(BaseModel):
    """Represents a single content block within a message."""
    block_index: int
    content_type: ContentType
    content_preview: str  # First 100 chars
    raw_content: str
    token_usage: TokenUsage
    tool_execution: Optional[ToolExecution] = None

class MCPEvent(BaseModel):
    """Base event structure for all MCP interactions."""
    model_config = ConfigDict(
        use_enum_values=True,
        json_encoders={
            datetime: lambda v: v.isoformat()
        }
    )
    
    event_type: EventType
    timestamp: datetime = Field(default_factory=datetime.now)
    conversation_id: str
    message_id: str
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    
    # Message context
    role: Optional[Role] = None
    block_index: Optional[int] = None
    
    # Content details
    content_block: Optional[ContentBlock] = None
    
    # API metadata
    stop_reason: Optional[str] = None
    model: Optional[str] = None
    api_usage_stats: Optional[Dict[str, Any]] = None
    
    # Additional context
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ConversationSummary(BaseModel):
    """Summary statistics for a conversation."""
    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat()
        }
    )
    
    conversation_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_duration_seconds: Optional[float] = None
    
    # Message counts
    total_messages: int = 0
    user_messages: int = 0
    assistant_messages: int = 0
    
    # Token statistics
    total_tokens: int = 0
    user_tokens: int = 0
    assistant_tokens: int = 0
    text_block_tokens: int = 0
    tool_use_tokens: int = 0
    tool_result_tokens: int = 0
    
    # Tool statistics
    total_tool_calls: int = 0
    successful_tool_calls: int = 0
    failed_tool_calls: int = 0
    tool_usage_by_name: Dict[str, int] = Field(default_factory=dict)
    average_tool_execution_time: Optional[float] = None
    
    # Performance metrics
    average_response_time: Optional[float] = None
    token_estimation_accuracy: Optional[float] = None


def estimate_tokens_from_text(text: str, method: str = "character_count") -> int:
    """
    Estimate token count from text content.
    
    Args:
        text: The text content to estimate
        method: Estimation method ("character_count" or "tiktoken")
    
    Returns:
        Estimated token count
    """
    if method == "character_count":
        # Simple character-based estimation: chars * 0.75
        return max(1, int(len(text) * 0.75))
    elif method == "tiktoken":
        # TODO: Implement tiktoken estimation in future
        # For now, fallback to character count
        return max(1, int(len(text) * 0.75))
    else:
        raise ValueError(f"Unknown estimation method: {method}")
    


def create_content_block(
    block_index: int,
    content_type: ContentType,
    raw_content: str,
    tool_execution: Optional[ToolExecution] = None
) -> ContentBlock:
    """
    Create a ContentBlock with automatic token estimation and content preview.
    
    This function handles the common task of creating content blocks from raw content,
    automatically generating a preview, estimating token usage, and optionally
    associating tool execution details.
    
    Args:
        block_index: Zero-based index of this content block within its message
        content_type: Type of content (TEXT, TOOL_USE, or TOOL_RESULT)
        raw_content: The full content text to be processed
        tool_execution: Optional tool execution details if this block represents
                       a tool use or tool result
    
    Returns:
        ContentBlock: A fully populated content block with token estimation
        
    Example:
        >>> block = create_content_block(
        ...     block_index=0,
        ...     content_type=ContentType.TEXT,
        ...     raw_content="Hello, how can I help you today?"
        ... )
        >>> print(block.token_usage.estimated_tokens)
        24
    """
    # Create preview (first 100 chars)
    content_preview = raw_content[:100]
    if len(raw_content) > 100:
        content_preview += "..."
    
    # Estimate tokens
    estimated_tokens = estimate_tokens_from_text(raw_content)
    token_usage = TokenUsage(
        estimated_tokens=estimated_tokens,
        estimation_method="character_count"
    )
    
    return ContentBlock(
        block_index=block_index,
        content_type=content_type,
        content_preview=content_preview,
        raw_content=raw_content,
        token_usage=token_usage,
        tool_execution=tool_execution
    )


def create_mcp_event(
    event_type: EventType,
    conversation_id: str,
    message_id: str,
    role: Optional[Role] = None,
    content_block: Optional[ContentBlock] = None,
    **kwargs
) -> MCPEvent:
    """
    Create an MCPEvent with automatic timestamp generation.
    
    This is the primary factory function for creating MCP events. It automatically
    sets the current timestamp and generates a unique event ID, while allowing
    all other event properties to be specified.
    
    Args:
        event_type: The type of MCP event being created
        conversation_id: Unique identifier for the conversation this event belongs to
        message_id: Unique identifier for the message this event relates to
        role: Optional role (USER, ASSISTANT, SYSTEM) if this event relates to a message
        content_block: Optional content block if this event represents content parsing
        **kwargs: Additional keyword arguments for MCPEvent fields (stop_reason,
                 model, api_usage_stats, metadata, etc.)
    
    Returns:
        MCPEvent: A fully populated MCP event with current timestamp and unique ID
        
    Example:
        >>> event = create_mcp_event(
        ...     event_type=EventType.MESSAGE_RECEIVED,
        ...     conversation_id="conv_123",
        ...     message_id="msg_456",
        ...     role=Role.USER,
        ...     metadata={"source": "terminal"}
        ... )
        >>> print(event.event_type)
        EventType.MESSAGE_RECEIVED
    """
    return MCPEvent(
        event_type=event_type,
        conversation_id=conversation_id,
        message_id=message_id,
        role=role,
        content_block=content_block,
        **kwargs
    )