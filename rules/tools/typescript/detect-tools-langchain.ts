// POSITIVE TEST CASES - Should be detected by the rule

// Tool imports - SHOULD MATCH
import { Tool } from "@langchain/core/tools";
import { DynamicTool } from "@langchain/core/tools";
import { StructuredTool } from "@langchain/core/tools";

// Agent imports - SHOULD MATCH
import { initializeAgentExecutorWithOptions } from "langchain/agents";
import { AgentExecutor } from "langchain/agents";

// Tool class definitions - SHOULD MATCH
class CustomTool extends Tool {
    name = "custom_tool";
    description = "A custom tool for testing";

    async _call(input: string): Promise<string> {
        return `Processed: ${input}`;
    }
}

class MyStructuredTool extends StructuredTool {
    name = "structured_tool";
    description = "A structured tool";

    async _call(input: string): Promise<string> {
        return input.toUpperCase();
    }
}

// Tool creation patterns - SHOULD MATCH
const searchTool = new DynamicTool({
    name: "web_search",
    description: "Search the web for information",
    func: async (input: string) => {
        return `Search results for: ${input}`;
    },
});

const calculatorTool = new DynamicTool({
    name: "calculator",
    description: "Perform mathematical calculations",
    func: (expression: string) => {
        return eval(expression).toString();
    },
});

// Agent initialization with tools - SHOULD MATCH
const agent = await initializeAgentExecutorWithOptions(
    [searchTool, calculatorTool],
    model,
    { agentType: "zero-shot-react-description" }
);

const executorAgent = initializeAgentExecutorWithOptions(tools, llm);

// Tool execution patterns - SHOULD MATCH
const result1 = await agent.call({ input: "What's the weather?" });
const result2 = await agent.invoke({ input: "Calculate 2+2" });
const toolResult = await searchTool.call("AI news");
const invokeResult = await calculatorTool.invoke("5*5");

// Tool arrays and registration - SHOULD MATCH
const tools = [searchTool, calculatorTool];
tools.push(new CustomTool());

// Function calling patterns - SHOULD MATCH
someObject.registerFunction(myFunction);
toolkit.registerFunction(helperTool);

// NEGATIVE TEST CASES - Should NOT be detected

// Regular imports (not tool-related)
import { Document } from "@langchain/core/documents";
import { BaseMessage } from "@langchain/core/messages";

// Regular class definitions
class DataProcessor {
    process(data: any) {
        return data;
    }
}

// Regular function calls
const data = processor.process(input);
const config = system.getConfiguration();

// Regular arrays
const items = ["item1", "item2"];
const numbers = [1, 2, 3];

// Regular method calls
object.callMethod(param);
service.executeCommand(cmd);