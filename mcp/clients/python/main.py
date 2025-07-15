from mcp_logger.wrapper import MCPChatBotWrapper
from mcp_logger.display import MCPDisplayConfig
from dotenv import load_dotenv
import asyncio

load_dotenv()

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