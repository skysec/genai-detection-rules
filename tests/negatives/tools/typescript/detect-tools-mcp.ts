/**
 * Negative test cases for MCP tool detection rule
 * These should NOT be detected by the detect-tools-mcp rule
 */

// Test 1: Regular function
function regularFunction(a: number, b: number): number {
  return a + b;
}

// Test 2: Async function without MCP
async function fetchData(url: string): Promise<any> {
  const response = await fetch(url);
  return response.json();
}

// Test 3: Class methods
class ToolManager {
  private tools: Map<string, Function> = new Map();

  addTool(name: string, handler: Function) {
    this.tools.set(name, handler);
  }

  executeTool(name: string, args: any) {
    const tool = this.tools.get(name);
    if (tool) {
      return tool(args);
    }
    throw new Error(`Tool not found: ${name}`);
  }
}

// Test 4: Interface definitions
interface Tool {
  name: string;
  description: string;
  execute: (args: any) => Promise<any>;
}

interface ToolConfig {
  tools: Tool[];
  version: string;
}

// Test 5: Type definitions
type ToolHandler = (name: string, args: any) => Promise<any>;
type ToolSchema = {
  type: string;
  properties: Record<string, any>;
};

// Test 6: Comments mentioning tools
// This code uses MCP tools for processing
// The server.setRequestHandler pattern is useful
/* Multi-line comment:
   Tool definitions go here
   Use CallToolRequestSchema for tool calls
*/

// Test 7: String containing tool patterns
const documentation = `
To define a tool:
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  // handler code
});
`;

const example = "Use CallToolRequestSchema for tool handling";

// Test 8: Mock or test implementations
class MockServer {
  setRequestHandler(schema: any, handler: Function) {
    // Mock implementation
  }
}

const mockServer = new MockServer();
mockServer.setRequestHandler("schema", async () => {});

// Test 9: Configuration objects (not tool definitions)
const config = {
  tools: ["tool1", "tool2", "tool3"],
  enabled: true
};

const toolsConfig = {
  name: "my-tool",
  description: "Tool description",
  parameters: {}
};

// Test 10: Regular switch statements (not tool-related)
function processCommand(command: string) {
  switch (command) {
    case "start":
      return "Starting...";
    case "stop":
      return "Stopping...";
    default:
      return "Unknown command";
  }
}

// Test 11: Function that uses tools (not defines them)
async function invokeTool(toolName: string, params: any) {
  // Code that invokes/calls a tool
  return { result: "invoked" };
}

// Test 12: HTTP request handlers (not MCP)
import express from 'express';

const app = express();
app.post('/api/tool', async (req, res) => {
  const { name, args } = req.body;
  res.json({ result: "executed" });
});

// Test 13: Event handlers
class EventEmitter {
  on(event: string, handler: Function) {}
}

const emitter = new EventEmitter();
emitter.on('tool:execute', (data) => {
  console.log('Tool executed:', data);
});

// Test 14: Generic handler function
function createHandler<T>(processor: (data: T) => any) {
  return async (input: T) => {
    return processor(input);
  };
}

// Test 15: Decorator pattern (not MCP)
function logged(target: any, propertyKey: string, descriptor: PropertyDescriptor) {
  const originalMethod = descriptor.value;
  descriptor.value = function (...args: any[]) {
    console.log(`Calling ${propertyKey}`);
    return originalMethod.apply(this, args);
  };
}

class Service {
  @logged
  processData(data: any) {
    return data;
  }
}

// Test 16: Builder pattern
class ToolBuilder {
  private name: string = "";
  private description: string = "";

  setName(name: string) {
    this.name = name;
    return this;
  }

  setDescription(desc: string) {
    this.description = desc;
    return this;
  }

  build() {
    return { name: this.name, description: this.description };
  }
}

// Test 17: Abstract classes
abstract class BaseTool {
  abstract execute(params: any): Promise<any>;
}

// Test 18: Enum definitions
enum ToolType {
  QUERY = "query",
  TRANSFORM = "transform",
  FETCH = "fetch"
}

// Test 19: Const assertions
const toolNames = ["tool1", "tool2", "tool3"] as const;

// Test 20: Generic type parameters
function processTool<T extends { name: string }>(tool: T): T {
  console.log(`Processing ${tool.name}`);
  return tool;
}
