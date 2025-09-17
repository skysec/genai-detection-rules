// Negative test cases for TypeScript MCP client detection - these should NOT trigger the rule

// Test case 1: Regular HTTP client imports
import axios from 'axios';
import fetch from 'node-fetch';
import { HttpClient } from '@angular/common/http';
import { Client as GraphQLClient } from 'graphql-request';

// Test case 2: Regular WebSocket clients
import WebSocket from 'ws';
import { io, Socket } from 'socket.io-client';

// Test case 3: Regular database clients
import { MongoClient } from 'mongodb';
import { Client as PgClient } from 'pg';
import { createClient } from 'redis';

// Test case 4: Regular API clients
import { TwitterApi } from 'twitter-api-v2';
import { OpenAI } from 'openai';
import { Anthropic } from '@anthropic-ai/sdk';

// Test case 5: Regular HTTP client class
class HttpApiClient {
  private baseUrl: string;
  private client: any;

  constructor(baseUrl: string) {
    this.baseUrl = baseUrl;
    this.client = axios.create({ baseURL: baseUrl });
  }

  async connect(): Promise<void> {
    // Regular HTTP connection check
    await this.client.get('/health');
  }

  async close(): Promise<void> {
    // No specific close needed for HTTP
  }

  async listItems(): Promise<any[]> {
    const response = await this.client.get('/items');
    return response.data;
  }

  async callEndpoint(endpoint: string, data: any): Promise<any> {
    const response = await this.client.post(`/${endpoint}`, data);
    return response.data;
  }

  async getResource(id: string): Promise<any> {
    const response = await this.client.get(`/resources/${id}`);
    return response.data;
  }
}

// Test case 6: Regular REST API client
class RestApiClient {
  private baseUrl: string;

  constructor(baseUrl: string) {
    this.baseUrl = baseUrl;
  }

  async listUsers(): Promise<any[]> {
    const response = await fetch(`${this.baseUrl}/users`);
    return response.json();
  }

  async callApi(endpoint: string, method: string, body?: any): Promise<any> {
    const response = await fetch(`${this.baseUrl}/${endpoint}`, {
      method,
      body: body ? JSON.stringify(body) : undefined,
      headers: { 'Content-Type': 'application/json' }
    });
    return response.json();
  }

  async getDocument(id: string): Promise<any> {
    const response = await fetch(`${this.baseUrl}/documents/${id}`);
    return response.json();
  }
}

// Test case 7: GraphQL client
class GraphQLApiClient {
  private client: GraphQLClient;

  constructor(endpoint: string) {
    this.client = new GraphQLClient(endpoint);
  }

  async listProjects(): Promise<any> {
    const query = `
      query {
        projects {
          id
          name
        }
      }
    `;
    return this.client.request(query);
  }

  async callQuery(query: string, variables?: any): Promise<any> {
    return this.client.request(query, variables);
  }
}

// Test case 8: WebSocket client
class WebSocketClient {
  private ws: WebSocket | null = null;

  async connect(url: string): Promise<void> {
    this.ws = new WebSocket(url);
    return new Promise((resolve) => {
      this.ws!.on('open', resolve);
    });
  }

  async close(): Promise<void> {
    if (this.ws) {
      this.ws.close();
    }
  }

  async listChannels(): Promise<any> {
    return new Promise((resolve) => {
      this.ws!.send(JSON.stringify({ type: 'list_channels' }));
      this.ws!.on('message', (data) => {
        resolve(JSON.parse(data.toString()));
      });
    });
  }

  async callMethod(method: string, params: any): Promise<any> {
    return new Promise((resolve) => {
      this.ws!.send(JSON.stringify({ type: method, params }));
      this.ws!.on('message', (data) => {
        resolve(JSON.parse(data.toString()));
      });
    });
  }
}

// Test case 9: Socket.IO client
class SocketIOClient {
  private socket: Socket;

  constructor(url: string) {
    this.socket = io(url);
  }

  async connect(): Promise<void> {
    return new Promise((resolve) => {
      this.socket.on('connect', resolve);
    });
  }

  async close(): Promise<void> {
    this.socket.disconnect();
  }

  async listRooms(): Promise<any> {
    return new Promise((resolve) => {
      this.socket.emit('list_rooms', (response: any) => {
        resolve(response);
      });
    });
  }

  async callEvent(event: string, data: any): Promise<any> {
    return new Promise((resolve) => {
      this.socket.emit(event, data, (response: any) => {
        resolve(response);
      });
    });
  }
}

// Test case 10: Database client
class DatabaseClient {
  private client: MongoClient;

  constructor(connectionString: string) {
    this.client = new MongoClient(connectionString);
  }

  async connect(): Promise<void> {
    await this.client.connect();
  }

  async close(): Promise<void> {
    await this.client.close();
  }

  async listCollections(): Promise<any> {
    const db = this.client.db();
    return db.listCollections().toArray();
  }

  async callProcedure(name: string, params: any): Promise<any> {
    const db = this.client.db();
    return db.collection('procedures').findOne({ name, params });
  }

  async getDocument(collection: string, id: string): Promise<any> {
    const db = this.client.db();
    return db.collection(collection).findOne({ _id: id });
  }
}

// Test case 11: Redis client
class CacheClient {
  private client: any;

  constructor() {
    this.client = createClient();
  }

  async connect(): Promise<void> {
    await this.client.connect();
  }

  async close(): Promise<void> {
    await this.client.quit();
  }

  async listKeys(): Promise<string[]> {
    return this.client.keys('*');
  }

  async callCommand(command: string, ...args: any[]): Promise<any> {
    return this.client.sendCommand([command, ...args]);
  }

  async getValue(key: string): Promise<any> {
    return this.client.get(key);
  }
}

// Test case 12: OpenAI client (similar AI client but not MCP)
class OpenAIClient {
  private client: OpenAI;

  constructor(apiKey: string) {
    this.client = new OpenAI({ apiKey });
  }

  async listModels(): Promise<any> {
    return this.client.models.list();
  }

  async callCompletion(prompt: string): Promise<any> {
    return this.client.completions.create({
      model: 'text-davinci-003',
      prompt,
      max_tokens: 100
    });
  }

  async getEmbedding(text: string): Promise<any> {
    return this.client.embeddings.create({
      model: 'text-embedding-ada-002',
      input: text
    });
  }
}

// Test case 13: Anthropic client (similar AI client but not MCP)
class AnthropicClient {
  private client: Anthropic;

  constructor(apiKey: string) {
    this.client = new Anthropic({ apiKey });
  }

  async listModels(): Promise<any> {
    // Anthropic doesn't have a public models endpoint
    return ['claude-3-opus-20240229', 'claude-3-sonnet-20240229'];
  }

  async callMessage(content: string): Promise<any> {
    return this.client.messages.create({
      model: 'claude-3-opus-20240229',
      max_tokens: 1000,
      messages: [{ role: 'user', content }]
    });
  }

  async getResponse(prompt: string): Promise<any> {
    return this.client.messages.create({
      model: 'claude-3-sonnet-20240229',
      max_tokens: 500,
      messages: [{ role: 'user', content: prompt }]
    });
  }
}

// Test case 14: Generic API client with similar method names
class GenericApiClient {
  private baseUrl: string;
  private httpClient: any;

  constructor(baseUrl: string) {
    this.baseUrl = baseUrl;
    this.httpClient = axios.create({ baseURL: baseUrl });
  }

  async connect(): Promise<void> {
    // Generic connection test
    await this.httpClient.get('/ping');
  }

  async close(): Promise<void> {
    // Generic cleanup
  }

  // These method names are similar to MCP but in different context
  async listTools(): Promise<any> {
    // Regular API endpoint for development tools
    const response = await this.httpClient.get('/dev/tools');
    return response.data;
  }

  async callTool(toolName: string, params: any): Promise<any> {
    // Regular API call to a development tool endpoint
    const response = await this.httpClient.post(`/dev/tools/${toolName}`, params);
    return response.data;
  }

  async listResources(): Promise<any> {
    // Regular API endpoint for cloud resources
    const response = await this.httpClient.get('/cloud/resources');
    return response.data;
  }

  async readResource(resourceId: string): Promise<any> {
    // Regular API call to read cloud resource
    const response = await this.httpClient.get(`/cloud/resources/${resourceId}`);
    return response.data;
  }

  async listPrompts(): Promise<any> {
    // Regular API endpoint for UI prompts/templates
    const response = await this.httpClient.get('/ui/prompts');
    return response.data;
  }

  async getPrompt(promptId: string): Promise<any> {
    // Regular API call to get UI prompt
    const response = await this.httpClient.get(`/ui/prompts/${promptId}`);
    return response.data;
  }
}

// Test case 15: Message queue client
class MessageQueueClient {
  private connection: any;

  async connect(url: string): Promise<void> {
    // Generic message queue connection
  }

  async close(): Promise<void> {
    // Generic cleanup
  }

  async request(message: any, schema?: any): Promise<any> {
    // Generic message queue request
    return { status: 'processed', data: message };
  }
}

// Test case 16: Configuration objects that might look similar
const apiConfig = {
  capabilities: {
    caching: true,
    compression: true,
    retries: 3
  }
};

const serviceConfig = {
  capabilities: {
    monitoring: {},
    logging: {},
    metrics: {}
  }
};

// Test case 17: Class with private client property (generic)
class ServiceManager {
  private client: any;
  private httpClient: any;

  constructor() {
    this.client = axios.create();
    this.httpClient = fetch;
  }
}

// Test case 18: Transport-like classes (but not MCP)
class HttpTransport {
  constructor(options: { url: string }) {
    // Generic HTTP transport
  }
}

class WebSocketTransport {
  constructor(url: string) {
    // Generic WebSocket transport
  }
}

// Test case 19: Child process management (not MCP stdio)
import { spawn } from 'child_process';

class ProcessManager {
  async startProcess(command: string, args: string[]): Promise<void> {
    const child = spawn(command, args);
    // This is generic process management, not MCP
  }
}

// Test case 20: Regular schema definitions
const ApiSchema = {
  type: 'object',
  properties: {
    result: { type: 'string' }
  }
};

const ResponseSchema = {
  type: 'object',
  properties: {
    data: { type: 'array' },
    status: { type: 'string' }
  }
};

// Export for potential use
export {
  HttpApiClient,
  RestApiClient,
  GraphQLApiClient,
  WebSocketClient,
  SocketIOClient,
  DatabaseClient,
  CacheClient,
  OpenAIClient,
  AnthropicClient,
  GenericApiClient,
  MessageQueueClient,
  ServiceManager,
  ProcessManager
};