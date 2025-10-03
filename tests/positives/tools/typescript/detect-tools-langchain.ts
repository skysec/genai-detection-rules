/**
 * Positive test cases for LangChain TypeScript tools detection.
 * These patterns should be detected by the detect-tools-langchain.yaml rule.
 */

// Core LangChain tools imports - VERY HIGH CONFIDENCE
import { BaseTool } from "@langchain/core/tools";
import { DynamicTool } from "@langchain/core/tools";
import { Tool } from "@langchain/core/tools";

// Agent imports with tools
import { AgentExecutor } from "langchain/agents";
import { initializeAgentExecutorWithOptions } from "langchain/agents";
import { createReactAgent } from "langchain/agents";

// Specific tool imports
import { Calculator } from "@langchain/community/tools/calculator";
import { WebBrowser } from "langchain/tools/webbrowser";
import { SerpAPI } from "@langchain/community/tools/serpapi";

// Tool class definitions - VERY HIGH CONFIDENCE
class CustomCalculatorTool extends BaseTool {
    name = "calculator";
    description = "Performs mathematical calculations";

    async _call(input: string): Promise<string> {
        try {
            const result = eval(input);
            return result.toString();
        } catch (error) {
            return `Error: ${error}`;
        }
    }
}

class WeatherTool extends BaseTool {
    name = "get_weather";
    description = "Gets current weather for a location";

    async _call(location: string): Promise<string> {
        // Mock weather API call
        return `Weather in ${location}: 72°F, sunny`;
    }
}

class DatabaseTool extends BaseTool {
    name = "database_query";
    description = "Queries the database";

    async _call(query: string): Promise<string> {
        // Mock database operation
        return `Database result for: ${query}`;
    }
}

// Agent initialization patterns - HIGH CONFIDENCE
const tools = [
    new Calculator(),
    new CustomCalculatorTool(),
    new WeatherTool()
];

const agent = await initializeAgentExecutorWithOptions(
    tools,
    llm,
    {
        agentType: "zero-shot-react-description",
        verbose: true
    }
);

const searchAgent = await initializeAgentExecutorWithOptions(
    [new SerpAPI()],
    llm,
    { agentType: "zero-shot-react-description" }
);

// LangChain specific tool patterns - HIGH CONFIDENCE
const dynamicTool = new DynamicTool({
    name: "dynamic_search",
    description: "Searches for information",
    func: async (input: string) => {
        return `Search results for: ${input}`;
    }
});

// Tools array patterns
const basicTools = [
    new CustomCalculatorTool(),
    new WeatherTool()
];

const webTools = [
    new WebBrowser(),
    new SerpAPI()
];

// Real-world usage examples
async function setupAgentWithTools() {
    // Define custom tools
    const calculator = new CustomCalculatorTool();
    const weather = new WeatherTool();
    const database = new DatabaseTool();

    // Create tool array
    const tools = [calculator, weather, database];

    // Initialize agent
    const agent = await initializeAgentExecutorWithOptions(
        tools,
        llm,
        {
            agentType: "zero-shot-react-description",
            verbose: true,
            maxIterations: 5
        }
    );

    return agent;
}

async function createFileManagementAgent() {
    // File management tools
    const fileTools = [
        new DynamicTool({
            name: "read_file",
            description: "Reads content from a file",
            func: async (filename: string) => {
                // Mock file reading
                return `Content of ${filename}`;
            }
        }),
        new DynamicTool({
            name: "write_file",
            description: "Writes content to a file",
            func: async (input: string) => {
                const [filename, content] = input.split("|");
                return `Written to ${filename}: ${content}`;
            }
        })
    ];

    // Agent with file tools
    const agent = await initializeAgentExecutorWithOptions(
        fileTools,
        llm,
        { agentType: "structured-chat-zero-shot-react-description" }
    );

    return agent;
}

// Advanced tool implementations
class APITool extends BaseTool {
    name = "api_call";
    description = "Makes API calls to external services";

    async _call(input: string): Promise<string> {
        const [endpoint, params] = input.split("|");
        // Mock API call
        return `API response from ${endpoint} with ${params}`;
    }
}

class EmailTool extends BaseTool {
    name = "send_email";
    description = "Sends emails to recipients";

    async _call(input: string): Promise<string> {
        const [recipient, subject, body] = input.split("|");
        // Mock email sending
        return `Email sent to ${recipient} with subject: ${subject}`;
    }
}

// Multi-agent setup with tools
async function createMultiAgentSystem() {
    // Research agent
    const researchTools = [
        new SerpAPI(),
        new WebBrowser()
    ];

    const researchAgent = await initializeAgentExecutorWithOptions(
        researchTools,
        llm,
        { agentType: "zero-shot-react-description" }
    );

    // Data agent
    const dataTools = [
        new DatabaseTool(),
        new CustomCalculatorTool()
    ];

    const dataAgent = await initializeAgentExecutorWithOptions(
        dataTools,
        llm,
        { agentType: "react" }
    );

    // Communication agent
    const commTools = [
        new EmailTool(),
        new APITool()
    ];

    const commAgent = await initializeAgentExecutorWithOptions(
        commTools,
        llm,
        { agentType: "zero-shot-react-description" }
    );

    return { researchAgent, dataAgent, commAgent };
}

// Custom tool with complex logic
class WebScrapingTool extends BaseTool {
    name = "web_scraper";
    description = "Scrapes content from web pages";

    async _call(url: string): Promise<string> {
        try {
            // Mock web scraping
            const response = await fetch(url);
            const text = await response.text();
            return text.substring(0, 1000); // First 1000 chars
        } catch (error) {
            return `Error scraping ${url}: ${error}`;
        }
    }
}

// Tool composition patterns
async function composeSpecializedAgent() {
    // Combine multiple tool types
    const allTools = [
        new CustomCalculatorTool(),
        new WeatherTool(),
        new WebScrapingTool(),
        new DatabaseTool(),
        new APITool(),
        new SerpAPI()
    ];

    // Specialized agent
    const agent = await initializeAgentExecutorWithOptions(
        allTools,
        llm,
        {
            agentType: "zero-shot-react-description",
            verbose: true,
            maxIterations: 5
        }
    );

    return agent;
}

// Dynamic tool loading
async function loadToolsDynamically() {
    const availableTools: BaseTool[] = [];

    // Add tools based on configuration
    availableTools.push(new CustomCalculatorTool());
    availableTools.push(new WeatherTool());

    // Conditionally add tools
    if (needSearch) {
        availableTools.push(new SerpAPI());
    }

    if (needFiles) {
        availableTools.push(
            new DynamicTool({
                name: "file_reader",
                description: "Reads files",
                func: async (filename: string) => `File content: ${filename}`
            })
        );
    }

    // Create agent with dynamic tools
    const agent = await initializeAgentExecutorWithOptions(
        availableTools,
        llm,
        { agentType: "zero-shot-react-description" }
    );

    return agent;
}

// Tool validation and error handling
class ValidatedTool extends BaseTool {
    name = "validated_tool";
    description = "Tool with input validation";

    async _call(inputData: string): Promise<string> {
        // Input validation
        if (!inputData || inputData.length < 3) {
            return "Error: Input too short";
        }

        // Process input
        const result = `Processed: ${inputData.toUpperCase()}`;
        return result;
    }
}

// Specialized agents for different domains
async function createDomainSpecificAgents() {
    // Finance agent
    const financeTools = [
        new CustomCalculatorTool(),
        new DatabaseTool()
    ];

    const financeAgent = await initializeAgentExecutorWithOptions(
        financeTools,
        llm,
        { agentType: "zero-shot-react-description" }
    );

    // Research agent
    const researchTools = [
        new SerpAPI(),
        new WebBrowser(),
        new WebScrapingTool()
    ];

    const researchAgent = await initializeAgentExecutorWithOptions(
        researchTools,
        llm,
        { agentType: "zero-shot-react-description" }
    );

    return { financeAgent, researchAgent };
}

// Tool interface implementations
interface CustomToolConfig {
    name: string;
    description: string;
    func: (input: string) => Promise<string>;
}

function createCustomTool(config: CustomToolConfig): DynamicTool {
    return new DynamicTool({
        name: config.name,
        description: config.description,
        func: config.func
    });
}

// Async tool patterns
const asyncTools = [
    createCustomTool({
        name: "async_processor",
        description: "Processes data asynchronously",
        func: async (input: string) => {
            await new Promise(resolve => setTimeout(resolve, 100));
            return `Async processed: ${input}`;
        }
    })
];

// Export patterns
export {
    CustomCalculatorTool,
    WeatherTool,
    DatabaseTool,
    setupAgentWithTools,
    createMultiAgentSystem,
    tools,
    basicTools
};