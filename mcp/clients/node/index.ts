import { Groq } from "groq-sdk/index.mjs";
import { Anthropic } from "@anthropic-ai/sdk";

import {
  ChatCompletionTool,
  ChatCompletionMessageParam,
} from "groq-sdk/resources/chat/completions.mjs";
import { ChatCompletionMessageToolCall } from "groq-sdk/src/resources/chat.js";

import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import readline from "readline/promises";
import dotenv from "dotenv";

dotenv.config();

const GROQ_API_KEY = process.env.GROQ_API_KEY;
if (!GROQ_API_KEY) {
  throw new Error("GROQ_API_KEY is not set");
}

// Groq function tool definition
export type Tool = ChatCompletionTool;
export type MessageParam = ChatCompletionMessageParam;

class MCPClient {
  private mcp: Client;
  private groq: Groq;
  private anthropic: Anthropic;
  private model: string;
  private transport: StdioClientTransport | null = null;
  private tools: Tool[] = [];

  constructor() {
    this.groq = new Groq({ apiKey: GROQ_API_KEY });
    this.anthropic = new Anthropic();
    this.mcp = new Client({ name: "mcp-client-cli", version: "1.0.0" });
    this.model = "meta-llama/llama-4-scout-17b-16e-instruct";
  }

  // methods will go here
  async connectToServer(serverScriptPath: string) {
    try {
      const isJs = serverScriptPath.endsWith(".js");
      const isPy = serverScriptPath.endsWith(".py");
      if (!isJs && !isPy) {
        throw new Error("Server script must be a .js or .py file");
      }
      const command = isPy
        ? process.platform === "win32"
          ? "python"
          : "python3"
        : process.execPath;

      this.transport = new StdioClientTransport({
        command,
        args: [serverScriptPath],
      });
      await this.mcp.connect(this.transport);

      const toolsResult = await this.mcp.listTools();
      this.tools = toolsResult.tools.map((tool) => ({
        type: "function",
        function: {
          name: tool.name,
          description: tool.description,
          parameters: tool.inputSchema,
        },
      }));
      console.log(
        "Connected to server with tools:",
        this.tools.map(({ function: { name } }) => name)
      );
    } catch (e) {
      console.log("Failed to connect to MCP server: ", e);
      throw e;
    }
  }

  async processQuery(query: string) {
    const messages: MessageParam[] = [
      {
        role: "user",
        content: query,
      },
    ];

    // const test = await this.anthropic.messages.create()
    const response = await this.groq.chat.completions.create({
      model: this.model,
      max_tokens: 1000,
      messages,
      tools: this.tools,
    });

    const finalText = [];

    // loop through responses
    for (const choice of response.choices) {
      // if the context is a simple text, just push it onto the final text
      if (choice.message.role in ["system", "user", "assistant"]) {
        finalText.push(choice.message.content);

        // if the context is a tool use, get the name and input args for the tool.
      } else if (choice.message.role in ["tool", "function"]) {
        // if function get name, if tool get
        const toolCalls: Array<ChatCompletionMessageToolCall> =
          choice.message.tool_calls ?? [];

        // guard agains no tools being called, even though this should not be possible
        if (toolCalls.length === 0) continue;

        const toolName = toolCalls[0].function.name;
        const toolArgs = toolCalls[0].function.arguments
          ? (JSON.parse(toolCalls[0].function.arguments) as {
              [x: string]: unknown;
            })
          : undefined;

        const result = await this.mcp.callTool({
          name: toolName,
          arguments: toolArgs,
        });

        finalText.push(
          `[Calling tool ${toolName} with args ${JSON.stringify(toolArgs)}]`
        );

        messages.push({
          role: "user",
          content: result.content as string,
        });

        const response = await this.groq.chat.completions.create({
          model: this.model,
          max_tokens: 1000,
          messages,
        });

        finalText.push(
          response.choices[0].message.role === "assistant"
            ? response.choices[0].message.content
            : ""
        );
      }
    }

    return finalText.join("\n");
  }

  async chatLoop() {
    const rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout,
    });

    try {
      console.log("\nMCP Client Started!");
      console.log("Type your queries or 'quit' to exit.");

      while (true) {
        const message = await rl.question("\nQuery: ");
        if (message.toLowerCase() === "quit") {
          break;
        }
        const response = await this.processQuery(message);
        console.log("\n" + response);
      }
    } finally {
      rl.close();
    }
  }

  async cleanup() {
    await this.mcp.close();
  }
}

async function main() {
  if (process.argv.length < 3) {
    console.log("Usage: node build/index.js <path_to_server_script>");
    return;
  }

  const mcpClient = new MCPClient();

  try {
    await mcpClient.connectToServer(process.argv[2]);
    await mcpClient.chatLoop();
  } finally {
    await mcpClient.cleanup();
    process.exit(0);
  }
}

main();
