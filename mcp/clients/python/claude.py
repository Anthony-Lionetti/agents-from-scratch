from dotenv import load_dotenv
from anthropic import Anthropic
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
from typing import List, Dict, TypedDict
from contextlib import AsyncExitStack
import logger
import json
import asyncio

# setup loggering
logger = logger.setup_logging(environment='dev')

load_dotenv()

class ToolDefinition(TypedDict):
    name: str
    description: str
    input_schema: dict

class MCP_ChatBot:

    def __init__(self):
        # Initialize session and client objects
        self.sessions: List[ClientSession] = [] # new
        self.exit_stack = AsyncExitStack() # new
        self.anthropic = Anthropic()
        self.available_tools: List[ToolDefinition] = [] # new
        self.tool_to_session: Dict[str, ClientSession] = {} # new


    async def connect_to_server(self, server_name: str, server_config: dict) -> None:
        """Connect to a single MCP server."""
        try:
            server_params = StdioServerParameters(**server_config)
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            ) # new
            read, write = stdio_transport
            session = await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            ) # new
            await session.initialize()
            self.sessions.append(session)
            
            # List available tools for this session
            response = await session.list_tools()
            tools = response.tools

            logger.debug(f"[MCP_ChatBot.connect_to_server] - Connected to {server_name}")
            logger.debug(f"[MCP_ChatBot.connect_to_server] - With tools {[t.name for t in tools]}")
            
            for tool in tools: # new
                self.tool_to_session[tool.name] = session
                self.available_tools.append({
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema
                })
        except Exception as e:
            logger.error(f"[MCP_ChatBot.connect_to_server] - Failed to connect to {server_name}: {e}")

    async def connect_to_servers(self): # new
        """Connect to all configured MCP servers via server_config.json file."""
        try:
            with open("server_config.json", "r") as file:
                data = json.load(file)
            
            servers = data.get("mcpServers", {})
            
            for server_name, server_config in servers.items():
                await self.connect_to_server(server_name, server_config)
        except Exception as e:
            logger.error(f"[MCP_ChatBot.connect_to_servers] - Error loading server config: {e}")
            raise
    
    async def process_query(self, query):
        logger.info(f"[MCP_ChatBot.process_query] - Processing Query: {query}")
        messages = [{'role':'user', 'content':query}]
        response = self.anthropic.messages.create(max_tokens = 2024,
                                      model = 'claude-3-7-sonnet-20250219', 
                                      tools = self.available_tools,
                                      messages = messages)

        logger.debug(f"[MCP_ChatBot.process_query] - Initial query raw response: {response}")

        process_query = True
        while process_query:
            assistant_content = []


            logger.debug(f"[MCP_ChatBot.process_query] - Response content: {response.content}\n\n")
            for content in response.content:
                logger.debug(f"[MCP_ChatBot.process_query] - Current content: {content}")
                

                if content.type =='text':
                    print(content.text)
                    assistant_content.append(content)
                    if(len(response.content) == 1):
                        process_query= False

                elif content.type == 'tool_use':
                    logger.debug(f"[MCP_ChatBot.process_query] - Current content: {content}")

                    assistant_content.append(content)
                    messages.append({'role':'assistant', 'content':assistant_content})
                    tool_id = content.id
                    tool_args = content.input
                    tool_name = content.name
                    
    
                    logger.debug(f"[MCP_ChatBot.process_query] - Calling tool: {tool_name}")
                    logger.debug(f"[MCP_ChatBot.process_query] - Tool args: {tool_args}")
                    
                    # Call a tool
                    session = self.tool_to_session[tool_name] # new
                    result = await session.call_tool(tool_name, arguments=tool_args)
                    messages.append({"role": "user", 
                                      "content": [
                                          {
                                              "type": "tool_result",
                                              "tool_use_id":tool_id,
                                              "content": result.content
                                          }
                                      ]
                                    })
                    response = self.anthropic.messages.create(max_tokens = 2024,
                                      model = 'claude-3-7-sonnet-20250219', 
                                      tools = self.available_tools,
                                      messages = messages) 
                    
                    logger.debug(f"[MCP_ChatBot.process_query] - Raw response: {response}")

                    if(len(response.content) == 1 and response.content[0].type == "text"):
                        logger.debug(f"[MCP_ChatBot.process_query] - Response text: {response.content[0].text}")
                        process_query= False

    
    
    async def chat_loop(self):
        """Run an interactive chat loop"""
        logger.info(f"[MCP_ChatBot.chat_loop] - MCP Chatbot Started!")

        print("Type your queries or 'quit' to exit.")
        
        while True:
            try:
                query = input("\nQuery: ").strip()
        
                if query.lower() == 'quit':
                    break
                    
                await self.process_query(query)
                print("\n")
                    
            except Exception as e:
                print(f"\nError: {str(e)}")
    
    async def cleanup(self): # new
        """Cleanly close all resources using AsyncExitStack."""
        await self.exit_stack.aclose()


async def main():
    chatbot = MCP_ChatBot()
    try:
        # the mcp clients and sessions are not initialized using "with"
        # like in the previous lesson
        # so the cleanup should be manually handled
        await chatbot.connect_to_servers() # new! 
        await chatbot.chat_loop()
    finally:
        await chatbot.cleanup() #new! 


if __name__ == "__main__":
    asyncio.run(main())