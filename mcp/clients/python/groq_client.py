from dotenv import load_dotenv
from groq import Groq
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
from typing import List, Dict, TypedDict
from contextlib import AsyncExitStack
import logger
import json
import asyncio

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
        self.groq = Groq()
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
                    "type": "function",
                    "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {
                        "type": tool.inputSchema['type'],
                        "properties": tool.inputSchema['properties']
                    },
                    "requried": tool.inputSchema['required']
                    }
                })
            
        except Exception as e:
            logger.error(f"[MCP_ChatBot.connect_to_server] - Failed to connect to {server_name}: {e}")

    async def connect_to_servers(self): # new
        """Connect to all configured MCP servers."""
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

        # Take user query and generate assistant response
        messages = [{'role':'user', 'content':query}]
        response = self.groq.chat.completions.create(
                                        messages=messages,
                                        tools=self.available_tools,
                                        model="meta-llama/llama-4-scout-17b-16e-instruct"
                                    )
        
        logger.debug(f"[MCP_ChatBot.process_query] - Initial query raw response: {response}")
        
        process_query = True
        while process_query:
            response_message = response.choices[0].message
            tool_names = [call.function.name for call in response_message.tool_calls] if response_message.tool_calls else []
            logger.debug(f"[MCP_ChatBot.process_query] - Response content: Role = {response_message.role} | Content = {response_message.content} {"| Tools " + str(tool_names) if tool_names else ""}\n\n")

            # Process the Assistant response
            for choice in response.choices:
                logger.debug(f"[MCP_ChatBot.process_query] - Current choice: {choice}")

                # Get text content from assistant response
                response_content = choice.message.content
                messages.append({'role':'assistant', 'content':response_content})

                # Get tool use invokations from assistant response
                tool_calls = choice.message.tool_calls

                ####
                # Process each tool call
                if tool_calls is not None:
                    for tool_call in tool_calls:
                        logger.debug(f"[MCP_ChatBot.process_query] - Tool Details: {tool_call}\n")
                        tool_name = tool_call.function.name
                        tool_args:object = json.loads(tool_call.function.arguments)
                    
                        logger.debug(f"[MCP_ChatBot.process_query] - Calling tool: {tool_name}")
                        logger.debug(f"[MCP_ChatBot.process_query] - Tool args: {tool_args}")
                        
                        # Call a tool
                        session = self.tool_to_session[tool_name] # new
                        # Await result of tool call
                        result = await session.call_tool(tool_name, arguments=tool_args)

                        # Append tool call to chat history
                        messages.append(
                            {
                                "tool_call_id": tool_call.id, 
                                "role": "tool", # Indicates this message is from tool use
                                "name": tool_name,
                                "content": result.content
                            }
                        )

                # generate next response 
                response = self.groq.chat.completions.create(
                                    messages=messages,
                                    tools=self.available_tools,
                                    model="meta-llama/llama-4-scout-17b-16e-instruct"
                                    )
            
                logger.debug(f"[MCP_ChatBot.process_query] - Raw response: {response}")
                
                if len(response.choices[0].message.tool_calls) == 0:
                    logger.debug(f"[MCP_ChatBot.process_query] - Response text: {response.choices[0].message.content}")
                    print(response.choices[0].message.content)
                    process_query= False

    
    
    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Chatbot Started!")
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