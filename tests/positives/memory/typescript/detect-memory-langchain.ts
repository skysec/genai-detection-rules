/**
 * Positive test cases for LangChain TypeScript memory detection.
 * These patterns should be detected by the detect-memory-langchain.yaml rule.
 */

// Core LangChain memory imports - VERY HIGH CONFIDENCE
import { BufferMemory } from "langchain/memory";
import { ConversationSummaryMemory } from "langchain/memory";
import { ConversationSummaryBufferMemory } from "langchain/memory";
import { ConversationChain } from "langchain/chains";

// Additional memory imports
import { ChatMessageHistory } from "langchain/memory";
import { RedisChatMessageHistory } from "@langchain/redis";
import { UpstashRedisChatMessageHistory } from "@langchain/community/stores";

// Memory instantiation - VERY HIGH CONFIDENCE
const memory = new BufferMemory({
    memoryKey: "chat_history",
    returnMessages: true
});

const bufferMemory = new BufferMemory({
    memoryKey: "chat_history",
    inputKey: "input",
    outputKey: "output"
});

const summaryMemory = new ConversationSummaryMemory({
    llm: llm,
    memoryKey: "chat_history",
    returnMessages: true
});

// Conversation chain with memory - HIGH CONFIDENCE
const conversation = new ConversationChain({
    llm: llm,
    memory: memory,
    verbose: true
});

const chatChain = new ConversationChain({
    llm: llm,
    memory: bufferMemory
});

// Memory-specific method calls - VERY HIGH CONFIDENCE
await memory.saveContext(
    { input: "Hello" },
    { output: "Hi there!" }
);

await memory.saveContext(inputs, outputs);
const variables = await memory.loadMemoryVariables({});
const context = await memory.loadMemoryVariables({ input: "test" });

// Chat message history - HIGH CONFIDENCE
const messageHistory = new ChatMessageHistory([]);

const redisHistory = new RedisChatMessageHistory({
    sessionId: "session_123",
    url: "redis://localhost:6379"
});

const upstashHistory = new UpstashRedisChatMessageHistory({
    sessionId: "session_456",
    config: {
        url: "https://...",
        token: "..."
    }
});

// Real-world usage examples
async function setupConversationMemory() {
    // Initialize memory
    const memory = new BufferMemory({
        memoryKey: "chat_history",
        returnMessages: true,
        inputKey: "input",
        outputKey: "output"
    });

    // Create conversation chain
    const conversation = new ConversationChain({
        llm: llm,
        memory: memory,
        verbose: true
    });

    return conversation;
}

async function conversationWithSummary() {
    // Summary memory for long conversations
    const memory = new ConversationSummaryMemory({
        llm: llm,
        memoryKey: "chat_history",
        returnMessages: true
    });

    // Save conversation context
    await memory.saveContext(
        { input: "What is machine learning?" },
        { output: "Machine learning is a subset of AI..." }
    );

    // Load memory variables
    const history = await memory.loadMemoryVariables({});

    return memory;
}

async function persistentMemoryExample() {
    // Redis-backed persistent memory
    const redisHistory = new RedisChatMessageHistory({
        sessionId: "user_123",
        url: "redis://localhost:6379/0"
    });

    const memory = new BufferMemory({
        chatHistory: redisHistory,
        memoryKey: "chat_history",
        returnMessages: true
    });

    // Create conversation with persistent memory
    const conversation = new ConversationChain({
        llm: llm,
        memory: memory
    });

    return conversation;
}

// Advanced memory configurations
class ChatBot {
    private memory: BufferMemory;
    private conversation: ConversationChain;

    constructor(llm: any) {
        this.memory = new BufferMemory({
            memoryKey: "chat_history",
            returnMessages: true
        });

        this.conversation = new ConversationChain({
            llm: llm,
            memory: this.memory
        });
    }

    async chat(message: string): Promise<string> {
        const response = await this.conversation.call({ input: message });
        return response.response;
    }

    async getHistory(): Promise<any> {
        return await this.memory.loadMemoryVariables({});
    }

    async clearMemory(): Promise<void> {
        this.memory.clear();
    }
}

// Memory with external storage
async function setupMemoryWithRedis() {
    const sessions = ["user_1", "user_2", "user_3"];

    for (const sessionId of sessions) {
        const redisMemory = new BufferMemory({
            chatHistory: new RedisChatMessageHistory({
                sessionId: sessionId,
                url: "redis://localhost:6379"
            }),
            returnMessages: true
        });

        // Use memory in conversation
        const conversation = new ConversationChain({
            llm: llm,
            memory: redisMemory
        });

        await conversation.call({ input: "Hello" });
    }
}

// Summary buffer memory
async function summaryBufferMemory() {
    const memory = new ConversationSummaryBufferMemory({
        llm: llm,
        maxTokenLimit: 500,
        returnMessages: true
    });

    // Save multiple contexts
    await memory.saveContext(
        { input: "Explain quantum computing" },
        { output: "Quantum computing uses quantum mechanics..." }
    );

    return memory;
}

// Memory in different contexts
async function createAgentWithMemory() {
    const memory = new BufferMemory({
        memoryKey: "chat_history",
        returnMessages: true
    });

    const agent = createAgent({
        tools: tools,
        llm: llm,
        memory: memory,
        verbose: true
    });

    return agent;
}

// Multiple memory types
const memoryTypes = {
    buffer: new BufferMemory({
        memoryKey: "chat_history"
    }),
    summary: new ConversationSummaryMemory({
        llm: llm,
        memoryKey: "chat_history"
    }),
    summaryBuffer: new ConversationSummaryBufferMemory({
        llm: llm,
        maxTokenLimit: 200
    })
};

// Async memory operations
async function memoryOperations() {
    const memory = new BufferMemory({
        returnMessages: true
    });

    // Async save and load
    await memory.saveContext(
        { input: "Question 1" },
        { output: "Answer 1" }
    );

    await memory.saveContext(
        { input: "Question 2" },
        { output: "Answer 2" }
    );

    const allVariables = await memory.loadMemoryVariables({});

    return allVariables;
}

// Function with memory parameter
function createConversationChain(llm: any, memory: BufferMemory) {
    return new ConversationChain({
        llm: llm,
        memory: memory,
        verbose: true
    });
}

// Memory configuration options
const advancedMemory = new BufferMemory({
    memoryKey: "conversation_history",
    inputKey: "human_input",
    outputKey: "ai_response",
    returnMessages: true,
    humanPrefix: "Human",
    aiPrefix: "AI"
});

// Export for testing
export {
    memory,
    bufferMemory,
    summaryMemory,
    conversation,
    setupConversationMemory,
    ChatBot,
    memoryTypes
};