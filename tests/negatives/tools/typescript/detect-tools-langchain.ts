/**
 * Negative test cases for LangChain TypeScript tools detection.
 * These patterns should NOT be detected by the detect-tools-langchain.yaml rule.
 */

import * as fs from 'fs';
import * as path from 'path';

// Generic classes with similar method names (but not LangChain)
class GenericTool {
    name: string;
    description: string;

    constructor(name: string = "generic") {
        this.name = name;
        this.description = "Generic tool";
    }

    async call(input: string): Promise<string> {
        // Generic call method (not LangChain)
        return `Processing: ${input}`;
    }

    async _call(args: string): Promise<string> {
        // Generic _call method (not LangChain)
        return "generic result";
    }
}

// Generic agent classes (not LangChain)
class SimpleAgent {
    private tools: any[] = [];

    async run(query: string): Promise<string> {
        // Generic agent run (not LangChain)
        return `Agent response to: ${query}`;
    }

    async invoke(query: string): Promise<string> {
        // Generic invoke method (not LangChain)
        return `Invoked with: ${query}`;
    }

    async call(input: any): Promise<any> {
        // Generic call method (not LangChain)
        return this.run(input);
    }
}

// Generic initialization functions (not LangChain)
async function initializeSystem(tools?: any[], config?: any): Promise<SimpleAgent> {
    // Generic system initialization (not LangChain)
    return new SimpleAgent();
}

async function setupAgent(tools?: any[]): Promise<{ agent: SimpleAgent; tools: any[] }> {
    // Generic agent setup (not LangChain)
    return { agent: new SimpleAgent(), tools: tools || [] };
}

// Generic tool registry patterns
class ToolRegistry {
    private tools: any[] = [];

    registerFunction(func: Function): void {
        // Generic function registration (not LangChain)
        this.tools.push(func);
    }

    async callFunction(name: string, args: any): Promise<string> {
        // Generic function calling (not LangChain)
        return `Called ${name} with ${args}`;
    }

    addTool(tool: any): void {
        // Generic tool addition (not LangChain)
        this.tools.push(tool);
    }
}

// Generic API client
class APIClient {
    private endpoint: string;

    constructor(endpoint: string) {
        this.endpoint = endpoint;
    }

    async run(request: any): Promise<string> {
        // Generic API run (not LangChain)
        return `API response for: ${request}`;
    }

    async invoke(method: string, params: any): Promise<any> {
        // Generic API invoke (not LangChain)
        return { result: "success", method, params };
    }

    async call(data: any): Promise<any> {
        // Generic API call (not LangChain)
        return this.run(data);
    }
}

// Generic conversation patterns
class ConversationManager {
    private history: Array<{ input: string; output: string }> = [];

    async run(message: string): Promise<string> {
        // Generic conversation run (not LangChain)
        const response = `Response to: ${message}`;
        this.history.push({ input: message, output: response });
        return response;
    }

    async call(input: string): Promise<string> {
        return this.run(input);
    }
}

// Generic function calling patterns
async function callFunction(funcName: string, args: any): Promise<string> {
    // Generic function calling (not LangChain)
    return `Function ${funcName} called with ${args}`;
}

function registerFunction(name: string, func: Function): any {
    // Generic function registration (not LangChain)
    return { name, function: func };
}

// Generic database operations
class DatabaseManager {
    private connection: string;

    constructor(connectionString: string) {
        this.connection = connectionString;
    }

    async run(query: string): Promise<string> {
        // Generic database run (not LangChain)
        return `Query result: ${query}`;
    }

    async _call(sql: string): Promise<any> {
        // Generic database _call (not LangChain)
        return { rows: [], count: 0 };
    }

    async call(operation: string): Promise<any> {
        return this.run(operation);
    }
}

// Generic file operations
class FileManager {
    async run(operation: string, filename: string): Promise<string> {
        // Generic file run (not LangChain)
        return `File operation ${operation} on ${filename}`;
    }

    async _call(command: string): Promise<string> {
        // Generic file _call (not LangChain)
        return "file operation complete";
    }

    async call(params: any): Promise<string> {
        return this.run(params.operation, params.filename);
    }
}

// Generic web scraping
class WebScraper {
    async run(url: string): Promise<string> {
        // Generic scraper run (not LangChain)
        return `Scraped content from ${url}`;
    }

    async _call(target: string): Promise<string> {
        // Generic scraper _call (not LangChain)
        return "scraped data";
    }

    async call(input: string): Promise<string> {
        return this.run(input);
    }
}

// Generic calculator
class Calculator {
    async run(expression: string): Promise<string> {
        // Generic calculator run (not LangChain)
        try {
            return eval(expression).toString();
        } catch {
            return "Error";
        }
    }

    async _call(calc: string): Promise<string> {
        // Generic calculator _call (not LangChain)
        return "42";
    }

    async call(input: string): Promise<string> {
        return this.run(input);
    }
}

// Generic workflow systems
class WorkflowEngine {
    private steps: any[] = [];

    async run(workflow: any): Promise<string> {
        // Generic workflow run (not LangChain)
        return `Executed workflow: ${workflow}`;
    }

    async invoke(stepName: string): Promise<string> {
        // Generic workflow invoke (not LangChain)
        return `Step ${stepName} completed`;
    }

    async call(params: any): Promise<string> {
        return this.run(params);
    }
}

// Generic imports with similar names
import { GenericBaseTool } from 'some-library';
import { GenericAgent } from 'another-library';

// Generic variable assignments (should not be detected)
const tools = ["tool1", "tool2", "tool3"];
const agent = "string agent";
const baseTool = "not a class";

// Generic function parameters
async function processData(data: any, tools?: any[], agent?: any): Promise<any> {
    // Generic function with tools/agent parameters (not LangChain)
    if (tools) {
        console.log(`Processing with tools: ${tools}`);
    }
    return data;
}

async function createService(name: string, tools?: any[]): Promise<any> {
    // Generic service creation (not LangChain)
    return { name, tools: tools || [] };
}

// Generic class methods
class GenericProcessor {
    private tools: any[] = [];

    async run(inputData: any): Promise<string> {
        // Generic run
        return `Processed: ${inputData}`;
    }

    async invoke(command: string): Promise<string> {
        // Generic invoke
        return `Invoked: ${command}`;
    }

    async call(data: any): Promise<string> {
        // Generic call
        return this.run(data);
    }
}

// Generic HTTP client
class HTTPClient {
    async run(request: any): Promise<any> {
        // Generic HTTP run (not LangChain)
        return fetch(request.url);
    }

    async invoke(method: string, url: string, data?: any): Promise<any> {
        // Generic HTTP invoke (not LangChain)
        return { status: 200, data: "response" };
    }

    async call(config: any): Promise<any> {
        return this.run(config);
    }
}

// Generic task runner
class TaskRunner {
    private tasks: any[] = [];

    async run(task: any): Promise<string> {
        // Generic task run (not LangChain)
        return `Task ${task} completed`;
    }

    async _call(taskId: string): Promise<any> {
        // Generic task _call (not LangChain)
        return { taskId, status: "done" };
    }

    async call(params: any): Promise<string> {
        return this.run(params);
    }
}

// Generic configuration
const CONFIG = {
    tools: ["tool1", "tool2"],
    agentType: "simple",
    timeout: 30
};

// Generic utilities
async function runCommand(command: string): Promise<number> {
    // Generic command runner (not LangChain)
    return 0; // Mock exit code
}

async function invokeService(serviceName: string, params: any): Promise<string> {
    // Generic service invoker (not LangChain)
    return `Service ${serviceName} invoked with ${params}`;
}

// Generic plugin system
class PluginManager {
    private plugins: any[] = [];

    async run(pluginName: string): Promise<string> {
        // Generic plugin run (not LangChain)
        return `Plugin ${pluginName} executed`;
    }

    registerFunction(func: Function): void {
        // Generic plugin registration (not LangChain)
        this.plugins.push(func);
    }

    async callFunction(name: string): Promise<string> {
        // Generic plugin call (not LangChain)
        return `Plugin function ${name} called`;
    }

    async call(config: any): Promise<string> {
        return this.run(config.name);
    }
}

// Generic machine learning
class MLModel {
    async run(features: number[]): Promise<number[]> {
        // Generic ML run (not LangChain)
        return [0.1, 0.9]; // Mock prediction
    }

    async _call(inputData: any): Promise<any> {
        // Generic ML _call (not LangChain)
        return { prediction: 0.8, confidence: 0.95 };
    }

    async call(data: any): Promise<any> {
        return this.run(data);
    }
}

// Generic data structures
const toolList: any[] = [];
const agentConfig: any = {};
const functionRegistry: any = {};

// Generic functions that might have similar names
async function runAnalysis(data: any, tools?: any[]): Promise<any> {
    // Generic analysis function (not LangChain)
    return { result: "analysis complete" };
}

async function invokePipeline(steps: any[], agent?: any): Promise<string> {
    // Generic pipeline invoker (not LangChain)
    return "pipeline complete";
}

// Generic chain pattern (not LangChain)
class ProcessingChain {
    private steps: Function[];

    constructor(steps: Function[]) {
        this.steps = steps;
    }

    async run(inputData: any): Promise<any> {
        let result = inputData;
        for (const step of this.steps) {
            result = await step(result);
        }
        return result;
    }

    async invoke(data: any): Promise<any> {
        return this.run(data);
    }

    async call(input: any): Promise<any> {
        return this.run(input);
    }
}

// Generic logging
const logger = {
    info: (message: string) => console.log(message),
    error: (error: any) => console.error(error)
};

function logToolUsage(toolName: string, result: any): void {
    // Generic logging (not LangChain)
    logger.info(`Tool ${toolName} returned: ${result}`);
}

// Generic testing utilities
class TestRunner {
    async run(testCase: any): Promise<any> {
        // Generic test run (not LangChain)
        return { passed: true, test: testCase };
    }

    async invoke(testSuite: any): Promise<string> {
        // Generic test invoke (not LangChain)
        return "all tests passed";
    }

    async call(config: any): Promise<any> {
        return this.run(config);
    }
}

// Generic business logic
async function processOrder(orderData: any, tools?: any[]): Promise<any> {
    // Generic business process (not LangChain)
    return { orderId: 123, status: "processed" };
}

async function handleRequest(request: any, agent?: any): Promise<any> {
    // Generic request handler (not LangChain)
    return { response: "handled", agent };
}

// Generic interfaces
interface ToolConfig {
    name: string;
    description: string;
    handler: (input: any) => Promise<any>;
}

interface AgentConfig {
    type: string;
    tools: any[];
    options: any;
}

// Generic factory functions
function createTool(config: ToolConfig): GenericTool {
    return new GenericTool(config.name);
}

async function createAgent(config: AgentConfig): Promise<SimpleAgent> {
    return new SimpleAgent();
}

// Export generic items (not LangChain)
export {
    GenericTool,
    SimpleAgent,
    ToolRegistry,
    APIClient,
    ConversationManager,
    ProcessingChain,
    createTool,
    createAgent
};