// POSITIVE TEST CASES - Should be detected by the rule

// Memory imports - SHOULD MATCH
import { BufferMemory } from "langchain/memory";
import { ConversationSummaryMemory } from "langchain/memory";
import { ConversationSummaryBufferMemory } from "langchain/memory";
import { ConversationChain } from "langchain/chains";
import { ChatMessageHistory } from "langchain/memory";
import { RedisChatMessageHistory } from "@langchain/redis";

// Memory instantiation - SHOULD MATCH
const memory = new BufferMemory({
    returnMessages: true,
    memoryKey: "chat_history",
    inputKey: "input",
    outputKey: "output"
});

const summaryMemory = new ConversationSummaryMemory({
    llm: model,
    memoryKey: "history",
    returnMessages: true
});

const bufferMemory = new ConversationSummaryBufferMemory({
    llm: model,
    maxTokenLimit: 2000
});

// Conversation chain with memory - SHOULD MATCH
const chain = new ConversationChain({
    llm: model,
    memory: memory,
    verbose: true
});

const conversationChain = new ConversationChain({
    llm: chatModel,
    memory: summaryMemory,
    prompt: conversationPrompt
});

// Memory usage patterns - SHOULD MATCH
const response1 = await chain.call({ input: "Hello, how are you?" });
const response2 = await chain.predict({ input: "What's my name?" });
const result = chain.call({ input: userMessage });

// Memory method calls - SHOULD MATCH
await memory.saveContext({ input: "User message" }, { output: "AI response" });
const memoryVars = await memory.loadMemoryVariables({});
await memory.clear();

memory.saveContext(inputs, outputs);
const variables = memory.loadMemoryVariables({});

// Chat message history - SHOULD MATCH
const messageHistory = new ChatMessageHistory([
    { role: "user", content: "Hello" },
    { role: "assistant", content: "Hi there!" }
]);

const redisHistory = new RedisChatMessageHistory({
    sessionId: "user123",
    config: { url: "redis://localhost:6379" }
});

// Memory configuration examples - SHOULD MATCH
const configuredMemory = new BufferMemory({
    chatHistory: messageHistory,
    returnMessages: true,
    memoryKey: "conversation",
    humanPrefix: "User",
    aiPrefix: "Assistant"
});

// Function with memory usage - SHOULD MATCH
async function createConversationWithMemory() {
    const memory = new BufferMemory({ returnMessages: true });

    const chain = new ConversationChain({
        llm: model,
        memory: memory,
        verbose: false
    });

    return chain;
}

// NEGATIVE TEST CASES - Should NOT be detected

// Regular imports
import { Document } from "@langchain/core/documents";
import { BaseMessage } from "@langchain/core/messages";
import { OpenAI } from "@langchain/openai";

// Regular class instantiation
const processor = new DataProcessor({
    config: settings,
    timeout: 5000
});

const service = new ApiService({
    baseUrl: "https://api.example.com",
    apiKey: process.env.API_KEY
});

// Regular method calls
const data = await service.getData();
const result = processor.process(input);
const config = system.getConfiguration();

// Regular storage operations
localStorage.setItem("key", "value");
sessionStorage.clear();
const cache = new Map();

// Regular async operations
const response = await fetch(url);
const json = await response.json();

// Regular conversation patterns (not memory-related)
const messages = [
    { role: "user", content: "Hello" },
    { role: "assistant", content: "Hi" }
];

const chatResponse = await openai.chat.completions.create({
    model: "gpt-4",
    messages: messages
});