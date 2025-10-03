/**
 * Negative test cases for LangChain TypeScript memory detection.
 * These patterns should NOT be detected by the detect-memory-langchain.yaml rule.
 */

import * as fs from 'fs';
import * as redis from 'redis';

// Generic memory/storage operations (not LangChain)
interface GenericMemory {
    data: Record<string, any>;
}

// Generic classes with similar method names (but not LangChain)
class GenericMemoryManager {
    private data: Record<string, any> = {};

    async saveContext(key: string, value: any): Promise<void> {
        // Not LangChain - generic save
        this.data[key] = value;
    }

    async loadMemoryVariables(filters?: any): Promise<Record<string, any>> {
        // Not LangChain - generic load
        return this.data;
    }

    clear(): void {
        // Generic clear method
        this.data = {};
    }
}

// Generic conversation classes (not LangChain)
class ChatBot {
    private history: Array<{ user: string; bot: string }> = [];

    async saveContext(input: any, output: any): Promise<void> {
        // Not LangChain save_context
        this.history.push({ user: input.text, bot: output.text });
    }

    async loadMemoryVariables(query?: any): Promise<any> {
        // Not LangChain load_memory_variables
        return { history: this.history };
    }

    clear(): void {
        this.history = [];
    }
}

// Generic database operations
class DatabaseConnection {
    constructor(private connectionString: string) {}

    async saveContext(table: string, data: any): Promise<void> {
        // Not LangChain - database save
        console.log(`Saving to ${table}:`, data);
    }

    async loadMemoryVariables(table: string): Promise<any> {
        // Not LangChain - database load
        return {};
    }
}

// Generic file operations
async function saveContextToFile(filename: string, context: any): Promise<void> {
    // Generic file save (not LangChain)
    await fs.promises.writeFile(filename, JSON.stringify(context));
}

async function loadMemoryVariablesFromFile(filename: string): Promise<any> {
    // Generic file load (not LangChain)
    const data = await fs.promises.readFile(filename, 'utf-8');
    return JSON.parse(data);
}

// Generic cache/memory management
class CacheManager {
    private cache: Map<string, any> = new Map();

    async saveContext(key: string, value: any, ttl: number = 3600): Promise<void> {
        // Not LangChain - cache save
        this.cache.set(key, { value, ttl });
    }

    async loadMemoryVariables(key: string): Promise<any> {
        // Not LangChain - cache load
        return this.cache.get(key);
    }

    clear(): void {
        this.cache.clear();
    }
}

// System memory operations
function getSystemMemoryUsage(): any {
    // System memory monitoring (not LangChain)
    return process.memoryUsage();
}

function forceGarbageCollection(): void {
    // Garbage collection (not LangChain)
    if (global.gc) {
        global.gc();
    }
}

// Generic AI/ML operations (not LangChain memory)
class TextProcessor {
    private memory: string[] = [];

    processText(text: string): string[] {
        // Not LangChain processing
        return text.split(' ');
    }

    async saveContext(text: string, result: string[]): Promise<void> {
        // Not LangChain - generic save
        this.memory.push(`${text}: ${result.join(' ')}`);
    }

    getMemory(): string[] {
        return this.memory;
    }
}

// Generic conversation tracking
class ConversationTracker {
    private conversations: Array<{
        user: string;
        bot: string;
        timestamp: number;
    }> = [];

    addExchange(userInput: string, botResponse: string): void {
        // Not LangChain exchange
        this.conversations.push({
            user: userInput,
            bot: botResponse,
            timestamp: Date.now()
        });
    }

    getHistory(): any[] {
        return this.conversations;
    }

    clear(): void {
        this.conversations = [];
    }
}

// Generic Redis operations
async function redisOperations(): Promise<void> {
    const client = redis.createClient({ url: 'redis://localhost:6379' });
    await client.connect();

    // Generic Redis operations (not LangChain)
    await client.set("session_123", "some_data");
    const data = await client.get("session_123");
    await client.del("session_123");

    await client.disconnect();
}

// Generic variable assignments (should not be detected)
const memory = "just a string variable";
const chatMemory: any[] = [];
const messageHistory: Record<string, any> = {};
const conversationHistory = ["msg1", "msg2"];

// Generic function parameters
function processData(data: any, memory?: any, chatMemory?: any): any {
    // Generic function with memory parameters (not LangChain)
    if (memory) {
        console.log(`Processing with memory: ${memory}`);
    }
    return data;
}

function createSession(sessionId: string, memoryStore?: any): any {
    // Generic session creation (not LangChain)
    return { id: sessionId, store: memoryStore };
}

// Generic class methods
class GenericProcessor {
    private memory: any[] = [];

    async saveContext(item: any): Promise<void> {
        // Not LangChain - just appending to array
        this.memory.push(item);
    }

    async loadMemoryVariables(): Promise<any[]> {
        // Not LangChain - just returning array
        return this.memory;
    }

    clear(): void {
        // Generic clear
        this.memory = [];
    }
}

// Generic imports with similar names
import { GenericBuffer } from 'some-library';
import { MemoryManager } from 'another-library';

// Generic conversation patterns
async function simpleChat(): Promise<void> {
    const history: Array<{ user: string; bot: string }> = [];

    const userInput = "Hello";

    // Generic response (not LangChain)
    const response = `Response to: ${userInput}`;

    // Generic history tracking (not LangChain)
    history.push({ user: userInput, bot: response });

    console.log(`Bot: ${response}`);
}

// Generic configuration
const CONFIG = {
    memoryLimit: 1000,
    sessionTimeout: 3600,
    persistence: true
};

// Generic utilities
async function saveToFile(data: any, filename: string): Promise<void> {
    await fs.promises.writeFile(filename, JSON.stringify(data));
}

async function loadFromFile(filename: string): Promise<any> {
    const data = await fs.promises.readFile(filename, 'utf-8');
    return JSON.parse(data);
}

// Generic API client
class APIClient {
    private sessionData: Record<string, any> = {};

    constructor(private endpoint: string) {}

    async saveContext(sessionId: string, data: any): Promise<void> {
        // Generic API save (not LangChain)
        this.sessionData[sessionId] = data;
    }

    async loadMemoryVariables(sessionId: string): Promise<any> {
        // Generic API load (not LangChain)
        return this.sessionData[sessionId] || {};
    }
}

// Generic logging
async function logConversation(userInput: string, botResponse: string): Promise<void> {
    // Generic logging (not LangChain)
    console.log(`User: ${userInput}, Bot: ${botResponse}`);
}

// Generic data structures
const conversationBuffer: any[] = [];
const messageQueue: any[] = [];
const sessionStore: Record<string, any> = {};

// Generic functions that might have similar names
function predict(inputData: any, context?: any): string {
    // Generic prediction function (not LangChain)
    return `Prediction for: ${inputData}`;
}

function processInput(text: string, memoryContext?: any): string {
    // Generic text processing (not LangChain)
    return text.toUpperCase();
}

// Generic chain pattern (not LangChain)
class ProcessingChain {
    constructor(private steps: Array<(input: any) => any>) {}

    async run(inputData: any): Promise<any> {
        let result = inputData;
        for (const step of this.steps) {
            result = await step(result);
        }
        return result;
    }
}

// Generic factory patterns
class ComponentFactory {
    static create(type: string, options?: any): any {
        // Generic factory (not LangChain)
        switch (type) {
            case 'memory':
                return new GenericMemoryManager();
            case 'processor':
                return new GenericProcessor();
            default:
                return null;
        }
    }
}

// Generic async operations
async function asyncMemoryOperation(): Promise<void> {
    const data = await loadFromFile('data.json');
    await processData(data);
    await saveToFile(data, 'output.json');
}

// Generic interfaces
interface ConversationContext {
    sessionId: string;
    memory: any[];
    timestamp: number;
}

interface MemoryProvider {
    saveContext(input: any, output: any): Promise<void>;
    loadMemoryVariables(query?: any): Promise<any>;
    clear(): void;
}

// Generic implementations
class LocalMemoryProvider implements MemoryProvider {
    private data: any[] = [];

    async saveContext(input: any, output: any): Promise<void> {
        this.data.push({ input, output });
    }

    async loadMemoryVariables(query?: any): Promise<any> {
        return this.data;
    }

    clear(): void {
        this.data = [];
    }
}

// Export generic items (not LangChain)
export {
    GenericMemoryManager,
    ChatBot,
    CacheManager,
    ConversationTracker,
    ProcessingChain,
    LocalMemoryProvider
};