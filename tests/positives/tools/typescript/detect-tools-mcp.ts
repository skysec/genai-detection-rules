/**
 * Positive test cases for MCP tool detection rule
 * These should all be detected by the detect-tools-mcp rule
 */

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { CallToolRequestSchema, ListToolsRequestSchema, Tool } from "@modelcontextprotocol/sdk/types.js";

const server = new Server({
  name: "test-mcp-server",
  version: "1.0.0"
}, {
  capabilities: {
    tools: {}
  }
});

// Test 1: CallToolRequest handler
// ruleid: detect-tools-mcp
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;

  switch (name) {
    case "calculate":
      return { content: [{ type: "text", text: String(args.a + args.b) }] };
    default:
      throw new Error(`Unknown tool: ${name}`);
  }
});

// Test 2: ListToolsRequest handler
// ruleid: detect-tools-mcp
server.setRequestHandler(ListToolsRequestSchema, async () => {
  return {
    tools: [
      {
        name: "calculate",
        description: "Calculate sum of two numbers",
        inputSchema: {
          type: "object",
          properties: {
            a: { type: "number" },
            b: { type: "number" }
          },
          required: ["a", "b"]
        }
      }
    ]
  };
});

// Test 3: Tool array definition
// ruleid: detect-tools-mcp
const tools: Tool[] = [
  {
    name: "search",
    description: "Search for information",
    inputSchema: {
      type: "object",
      properties: {
        query: { type: "string" }
      }
    }
  },
  {
    name: "fetch_data",
    description: "Fetch data from URL",
    inputSchema: {
      type: "object",
      properties: {
        url: { type: "string" }
      }
    }
  }
];

// Test 4: Tool switch statement
// ruleid: detect-tools-mcp
const mcp = new Server({ name: "tools-server", version: "1.0.0" }, { capabilities: { tools: {} } });

mcp.setRequestHandler(CallToolRequestSchema, async (request) => {
  // ruleid: detect-tools-mcp
  switch (request.params.name) {
    case "read_file":
      return { content: [{ type: "text", text: "file contents" }] };
    case "write_file":
      return { content: [{ type: "text", text: "success" }] };
    default:
      throw new Error("Unknown tool");
  }
});

// Test 5: Tool handler function
// ruleid: detect-tools-mcp
async function handleToolCall(name: string, args: any) {
  switch (name) {
    case "get_weather":
      return { temperature: 72, condition: "sunny" };
    case "get_time":
      return { time: new Date().toISOString() };
    default:
      throw new Error(`Tool not found: ${name}`);
  }
}

// Test 6: Tool list with detailed schemas
// ruleid: detect-tools-mcp
const toolList = [
  {
    name: "query_database",
    description: "Execute a SQL query",
    inputSchema: {
      type: "object",
      properties: {
        query: {
          type: "string",
          description: "SQL query to execute"
        },
        database: {
          type: "string",
          description: "Database name"
        }
      },
      required: ["query"]
    }
  }
];

// Test 7: Import of MCP tool types
// ruleid: detect-tools-mcp
import { CallToolRequestSchema as ToolRequest } from "@modelcontextprotocol/sdk/types.js";

// Test 8: Tool handler with arrow function
// ruleid: detect-tools-mcp
const toolHandler = async (toolName: string, arguments: any) => {
  if (toolName === "process") {
    return { result: "processed" };
  }
  throw new Error("Unknown tool");
};

// Test 9: Multiple tool definitions
const apiServer = new Server({ name: "api-server", version: "1.0.0" }, { capabilities: { tools: {} } });

// ruleid: detect-tools-mcp
apiServer.setRequestHandler(ListToolsRequestSchema, async () => ({
  tools: [
    { name: "tool1", description: "First tool", inputSchema: { type: "object" } },
    { name: "tool2", description: "Second tool", inputSchema: { type: "object" } },
    { name: "tool3", description: "Third tool", inputSchema: { type: "object" } }
  ]
}));

// Test 10: Tool execution handler
// ruleid: detect-tools-mcp
apiServer.setRequestHandler(CallToolRequestSchema, async (req) => {
  const toolName = req.params.name;
  const toolArgs = req.params.arguments;
  return { content: [{ type: "text", text: `Executed ${toolName}` }] };
});

// Test 11: Nested tool object
// ruleid: detect-tools-mcp
const complexTool = {
  name: "complex_operation",
  description: "Perform complex operation",
  inputSchema: {
    type: "object",
    properties: {
      input: { type: "string" },
      options: {
        type: "object",
        properties: {
          format: { type: "string" },
          verbose: { type: "boolean" }
        }
      }
    }
  }
};

// Test 12: Tool handler with type safety
interface ToolArgs {
  [key: string]: any;
}

// ruleid: detect-tools-mcp
async function typedToolHandler(name: string, args: ToolArgs): Promise<any> {
  switch (name) {
    case "calculate":
      return { result: args.a + args.b };
    default:
      throw new Error("Not found");
  }
}
