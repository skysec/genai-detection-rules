// Positive test cases for TypeScript MCP client detection

// Test case 1: Core MCP client imports
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { Client as MCPClient } from "@modelcontextprotocol/sdk/client";

// Test case 2: MCP transport imports
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import { StdioClientTransport as StdioTransport } from "@modelcontextprotocol/sdk/client/stdio";
import { SSEClientTransport } from "@modelcontextprotocol/sdk/client/sse.js";
import { SSEClientTransport as SSETransport } from "@modelcontextprotocol/sdk/client/sse";
import { HttpClientTransport } from "@modelcontextprotocol/sdk/client/http.js";
import { HttpClientTransport as HttpTransport } from "@modelcontextprotocol/sdk/client/http";

// Test case 3: MCP types and schema imports
import {
  CallToolResultSchema,
  ListToolsResultSchema,
  ListResourcesResultSchema,
  GetPromptResultSchema
} from "@modelcontextprotocol/sdk/types.js";

import {
  CallToolResultSchema as CallToolSchema,
  ListToolsResultSchema as ListToolsSchema
} from "@modelcontextprotocol/sdk/types";

// Test case 4: Multi-server management import
import { MultiServerManager } from "@modelcontextprotocol/sdk/client";

// Test case 5: Basic MCP client instantiation
function createBasicMCPClient() {
  const client = new Client({
    name: "mcp-client",
    version: "1.0.0"
  });

  return client;
}

// Test case 6: MCP client with capabilities
function createMCPClientWithCapabilities() {
  const client = new Client(
    { name: "example-client", version: "1.0.0" },
    {
      capabilities: {
        prompts: {},
        resources: {},
        tools: {}
      }
    }
  );

  return client;
}

// Test case 7: Class-based MCP client implementation
class MCPClientManager {
  private mcp: Client;
  private transport: StdioClientTransport | SSEClientTransport;

  constructor() {
    this.mcp = new Client({
      name: "mcp-client-cli",
      version: "1.0.0"
    });
  }

  // Test case 8: STDIO transport setup
  async connectToStdioServer(serverScript: string): Promise<void> {
    this.transport = new StdioClientTransport({
      command: "python",
      args: [serverScript]
    });

    await this.mcp.connect(this.transport);
  }

  // Test case 9: SSE transport setup
  async connectToSSEServer(url: string): Promise<void> {
    const transport = new SSEClientTransport(url);
    await this.mcp.connect(transport);
  }

  // Test case 10: HTTP transport setup
  async connectToHttpServer(baseUrl: string): Promise<void> {
    const transport = new HttpClientTransport({
      url: baseUrl
    });
    await this.mcp.connect(transport);
  }

  // Test case 11: MCP tool operations
  async performToolOperations(): Promise<void> {
    // List available tools
    const tools = await this.mcp.listTools();
    const availableTools = await this.mcp.listTools();

    // Call specific tools
    const result = await this.mcp.callTool("calculate", { a: 5, b: 3 });
    const weatherResult = await this.mcp.callTool("get_weather", {
      location: "San Francisco"
    });
  }

  // Test case 12: MCP resource operations
  async performResourceOperations(): Promise<void> {
    // List available resources
    const resources = await this.mcp.listResources();
    const availableResources = await this.mcp.listResources();

    // Read specific resources
    const content = await this.mcp.readResource("file:///example.txt");
    const document = await this.mcp.readResource({
      uri: "file:///document.pdf"
    });
  }

  // Test case 13: MCP prompt operations
  async performPromptOperations(): Promise<void> {
    // List available prompts
    const prompts = await this.mcp.listPrompts();
    const availablePrompts = await this.mcp.listPrompts();

    // Get specific prompts
    const prompt = await this.mcp.getPrompt("code_review", {
      language: "typescript"
    });
    const greeting = await this.mcp.getPrompt("greeting", {});
  }

  // Test case 14: Connection management
  async cleanup(): Promise<void> {
    if (this.mcp) {
      await this.mcp.close();
    }
  }
}

// Test case 15: Low-level MCP request patterns
async function performLowLevelRequests(client: Client): Promise<void> {
  // Tools requests
  const toolsResult = await client.request(
    { method: "tools/list" },
    ListToolsResultSchema
  );

  const callResult = await client.request(
    { method: "tools/call" },
    CallToolResultSchema
  );

  // Resources requests
  const resourcesResult = await client.request(
    { method: "resources/list" },
    ListResourcesResultSchema
  );

  const readResult = await client.request(
    { method: "resources/read" },
    ListResourcesResultSchema // Schema placeholder
  );

  // Prompts requests
  const promptsResult = await client.request(
    { method: "prompts/list" },
    GetPromptResultSchema
  );

  const getPromptResult = await client.request(
    { method: "prompts/get" },
    GetPromptResultSchema
  );
}

// Test case 16: Alternative transport configurations
function createAlternativeTransports() {
  // Node.js server
  const nodeTransport = new StdioClientTransport({
    command: "node",
    args: ["server.js"]
  });

  // NPX-based server
  const npxTransport = new StdioClientTransport({
    command: "npx",
    args: ["-y", "kubernetes-mcp-server@latest"]
  });

  // SSE with URL object
  const sseTransport = new SSEClientTransport(
    new URL("http://localhost:3000/sse")
  );

  return { nodeTransport, npxTransport, sseTransport };
}

// Test case 17: Multi-server management
class MultiServerMCPClient {
  private manager: MultiServerManager;

  constructor() {
    this.manager = new MultiServerManager();
  }

  async setupMultipleServers(): Promise<void> {
    // Setup would involve multiple server connections
    // This pattern is specific to MCP multi-server scenarios
  }
}

// Test case 18: MCP with AI integration pattern
import { Anthropic } from "@anthropic-ai/sdk";

class MCPAnthropicClient {
  private mcp: Client;
  private anthropic: Anthropic;

  constructor() {
    this.anthropic = new Anthropic();
    this.mcp = new Client({
      name: "mcp-anthropic-client",
      version: "1.0.0"
    });
  }

  async initializeMCPConnection(serverPath: string): Promise<void> {
    const transport = new StdioClientTransport({
      command: "python",
      args: [serverPath]
    });

    await this.mcp.connect(transport);
  }

  async callMCPTool(toolName: string, args: any): Promise<any> {
    const result = await this.mcp.callTool(toolName, args);
    return result;
  }
}

// Test case 19: Error handling with MCP operations
async function mcpWithErrorHandling(): Promise<void> {
  const client = new Client({
    name: "error-handling-client",
    version: "1.0.0"
  });

  const transport = new StdioClientTransport({
    command: "python",
    args: ["server.py"]
  });

  try {
    await client.connect(transport);
    const tools = await client.listTools();
    const result = await client.callTool("example_tool", {});
  } catch (error) {
    console.error("MCP operation failed:", error);
  } finally {
    await client.close();
  }
}

// Test case 20: Complex capabilities configuration
function createClientWithDetailedCapabilities() {
  const client = new Client(
    { name: "advanced-client", version: "2.0.0" },
    {
      capabilities: {
        prompts: {
          listChanged: true
        },
        resources: {
          subscribe: true,
          listChanged: true
        },
        tools: {
          listChanged: true
        }
      }
    }
  );

  return client;
}

// Export for potential use
export {
  MCPClientManager,
  MultiServerMCPClient,
  MCPAnthropicClient,
  createBasicMCPClient,
  createMCPClientWithCapabilities
};