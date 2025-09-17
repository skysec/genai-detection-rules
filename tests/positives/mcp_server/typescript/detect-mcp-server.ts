// POSITIVE TEST CASES - Should be detected by the rule

// =============================================================================
// MCP Core Imports - SHOULD MATCH (HIGH CONFIDENCE)
// =============================================================================

// ruleid: detect-mcp-server-typescript
import { Server } from "@modelcontextprotocol/sdk/server";
// ruleid: detect-mcp-server-typescript
import { Server, StdioServerTransport } from "@modelcontextprotocol/sdk/server";
// ruleid: detect-mcp-server-typescript
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio";
// ruleid: detect-mcp-server-typescript
import { SSEServerTransport } from "@modelcontextprotocol/sdk/server/sse";
// ruleid: detect-mcp-server-typescript
import { Server as MCPServer } from "@modelcontextprotocol/sdk/server";

// =============================================================================
// MCP Types Imports - SHOULD MATCH (HIGH CONFIDENCE)
// =============================================================================

// ruleid: detect-mcp-server-typescript
import { ListToolsRequestSchema } from "@modelcontextprotocol/sdk/types";
// ruleid: detect-mcp-server-typescript
import { CallToolRequestSchema } from "@modelcontextprotocol/sdk/types";
// ruleid: detect-mcp-server-typescript
import { ListResourcesRequestSchema } from "@modelcontextprotocol/sdk/types";
// ruleid: detect-mcp-server-typescript
import { ReadResourceRequestSchema } from "@modelcontextprotocol/sdk/types";
// ruleid: detect-mcp-server-typescript
import { ListPromptsRequestSchema } from "@modelcontextprotocol/sdk/types";
// ruleid: detect-mcp-server-typescript
import { GetPromptRequestSchema } from "@modelcontextprotocol/sdk/types";

// Multiple imports
// ruleid: detect-mcp-server-typescript
import {
  ListToolsRequestSchema,
  CallToolRequestSchema,
  ListResourcesRequestSchema
} from "@modelcontextprotocol/sdk/types";

// =============================================================================
// MCP Server Class Instantiation - SHOULD MATCH (HIGH CONFIDENCE)
// =============================================================================

// ruleid: detect-mcp-server-typescript
const server = new Server({
  name: "weather-server",
  version: "1.0.0"
});

// ruleid: detect-mcp-server-typescript
const mcpServer = new Server({
  name: "calculator-server",
  version: "2.1.0",
  capabilities: {
    tools: {},
    resources: {},
    prompts: {}
  }
});

// ruleid: detect-mcp-server-typescript
let weatherServer = new Server({
  name: "weather",
  version: "1.0.0"
});

// =============================================================================
// MCP Request Handlers - SHOULD MATCH (HIGH CONFIDENCE)
// =============================================================================

// Tool handlers
// ruleid: detect-mcp-server-typescript
server.setRequestHandler(ListToolsRequestSchema, async () => {
  return {
    tools: [
      {
        name: "get_weather",
        description: "Get weather for a location",
        inputSchema: {
          type: "object",
          properties: {
            location: { type: "string", description: "Location name" }
          },
          required: ["location"]
        }
      }
    ]
  };
});

// ruleid: detect-mcp-server-typescript
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;

  if (name === "get_weather") {
    const location = args.location;
    return {
      content: [
        {
          type: "text",
          text: `Weather for ${location}: Sunny, 72°F`
        }
      ]
    };
  }

  throw new Error(`Unknown tool: ${name}`);
});

// Resource handlers
// ruleid: detect-mcp-server-typescript
server.setRequestHandler(ListResourcesRequestSchema, async () => {
  return {
    resources: [
      {
        uri: "file:///config.json",
        name: "Configuration",
        description: "Server configuration file",
        mimeType: "application/json"
      }
    ]
  };
});

// ruleid: detect-mcp-server-typescript
server.setRequestHandler(ReadResourceRequestSchema, async (request) => {
  const { uri } = request.params;

  if (uri === "file:///config.json") {
    return {
      contents: [
        {
          uri: uri,
          mimeType: "application/json",
          text: JSON.stringify({ setting: "value" }, null, 2)
        }
      ]
    };
  }

  throw new Error(`Resource not found: ${uri}`);
});

// Prompt handlers
// ruleid: detect-mcp-server-typescript
server.setRequestHandler(ListPromptsRequestSchema, async () => {
  return {
    prompts: [
      {
        name: "weather_prompt",
        description: "Generate weather report prompt",
        arguments: [
          {
            name: "location",
            description: "Location for weather report",
            required: true
          }
        ]
      }
    ]
  };
});

// ruleid: detect-mcp-server-typescript
server.setRequestHandler(GetPromptRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;

  if (name === "weather_prompt") {
    return {
      description: "Weather report prompt",
      messages: [
        {
          role: "user",
          content: {
            type: "text",
            text: `Please provide a detailed weather report for ${args?.location || "unknown location"}`
          }
        }
      ]
    };
  }

  throw new Error(`Unknown prompt: ${name}`);
});

// =============================================================================
// MCP Transport Setup Patterns - SHOULD MATCH (HIGH CONFIDENCE)
// =============================================================================

// ruleid: detect-mcp-server-typescript
const transport = new StdioServerTransport();

// ruleid: detect-mcp-server-typescript
const sseTransport = new SSEServerTransport({
  port: 3000,
  path: "/mcp"
});

// ruleid: detect-mcp-server-typescript
server.connect(transport);

// ruleid: detect-mcp-server-typescript
server.connect(new StdioServerTransport());

// ruleid: detect-mcp-server-typescript
server.connect(new SSEServerTransport({
  port: 8080,
  path: "/mcp"
}));

// ruleid: detect-mcp-server-typescript
await server.connect(transport);

// =============================================================================
// Tool Definition Patterns - SHOULD MATCH (HIGH CONFIDENCE)
// =============================================================================

// ruleid: detect-mcp-server-typescript
const weatherTool = {
  name: "get_weather",
  description: "Get current weather for a location",
  inputSchema: {
    type: "object",
    properties: {
      location: {
        type: "string",
        description: "The location to get weather for"
      },
      units: {
        type: "string",
        enum: ["celsius", "fahrenheit"],
        description: "Temperature units"
      }
    },
    required: ["location"]
  }
};

// ruleid: detect-mcp-server-typescript
const calculatorTool = {
  name: "calculate",
  description: "Perform mathematical calculations",
  inputSchema: {
    type: "object",
    properties: {
      expression: {
        type: "string",
        description: "Mathematical expression to evaluate"
      }
    },
    required: ["expression"]
  }
};

// =============================================================================
// Resource Definition Patterns - SHOULD MATCH (HIGH CONFIDENCE)
// =============================================================================

// ruleid: detect-mcp-server-typescript
const configResource = {
  uri: "file:///app/config.json",
  name: "Application Configuration",
  description: "Main application configuration file",
  mimeType: "application/json"
};

// ruleid: detect-mcp-server-typescript
const logResource = {
  uri: "file:///var/log/app.log",
  name: "Application Logs",
  description: "Current application log file",
  mimeType: "text/plain"
};

// =============================================================================
// Prompt Definition Patterns - SHOULD MATCH (HIGH CONFIDENCE)
// =============================================================================

// ruleid: detect-mcp-server-typescript
const codeReviewPrompt = {
  name: "code_review",
  description: "Generate a code review prompt",
  arguments: [
    {
      name: "code",
      description: "Code to review",
      required: true
    },
    {
      name: "language",
      description: "Programming language",
      required: false
    }
  ]
};

// ruleid: detect-mcp-server-typescript
const dataAnalysisPrompt = {
  name: "analyze_data",
  description: "Generate data analysis prompt",
  arguments: [
    {
      name: "dataset",
      description: "Dataset description",
      required: true
    }
  ]
};

// =============================================================================
// Complete MCP Server Example - SHOULD MATCH (HIGH CONFIDENCE)
// =============================================================================

// ruleid: detect-mcp-server-typescript
import { Server } from "@modelcontextprotocol/sdk/server";
// ruleid: detect-mcp-server-typescript
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio";
// ruleid: detect-mcp-server-typescript
import {
  ListToolsRequestSchema,
  CallToolRequestSchema,
} from "@modelcontextprotocol/sdk/types";

// ruleid: detect-mcp-server-typescript
const mcpApp = new Server({
  name: "example-server",
  version: "1.0.0",
});

// ruleid: detect-mcp-server-typescript
mcpApp.setRequestHandler(ListToolsRequestSchema, async () => {
  return {
    tools: [
      {
        name: "echo",
        description: "Echo back the input",
        inputSchema: {
          type: "object",
          properties: {
            message: { type: "string" }
          },
          required: ["message"]
        }
      }
    ]
  };
});

// ruleid: detect-mcp-server-typescript
mcpApp.setRequestHandler(CallToolRequestSchema, async (request) => {
  if (request.params.name === "echo") {
    return {
      content: [
        {
          type: "text",
          text: request.params.arguments?.message || "No message"
        }
      ]
    };
  }
  throw new Error("Unknown tool");
});

async function main() {
  // ruleid: detect-mcp-server-typescript
  const transport = new StdioServerTransport();
  // ruleid: detect-mcp-server-typescript
  await mcpApp.connect(transport);
}

main().catch(console.error);