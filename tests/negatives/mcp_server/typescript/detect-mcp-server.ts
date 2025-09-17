// NEGATIVE TEST CASES - Should NOT be detected by the MCP server rule

// Regular TypeScript/JavaScript imports - should NOT match
import express from "express";
import { Request, Response } from "express";
import * as fs from "fs";
import { readFile } from "fs/promises";
import axios from "axios";
import { createServer } from "http";
import { WebSocket } from "ws";

// Standard library imports - should NOT match
import path from "path";
import os from "os";
import crypto from "crypto";
import { EventEmitter } from "events";
import { Transform } from "stream";

// Third-party library imports - should NOT match
import lodash from "lodash";
import moment from "moment";
import { v4 as uuidv4 } from "uuid";
import bcrypt from "bcrypt";
import jwt from "jsonwebtoken";

// React/Vue/Angular imports - should NOT match
import React from "react";
import { Component } from "react";
import { useState, useEffect } from "react";
import { createApp } from "vue";
import { NgModule } from "@angular/core";

// Database imports - should NOT match
import { MongoClient } from "mongodb";
import { Pool } from "pg";
import { createConnection } from "mysql2";
import { Sequelize } from "sequelize";
import { PrismaClient } from "@prisma/client";

// Testing framework imports - should NOT match
import { describe, it, expect } from "vitest";
import { test } from "@jest/globals";
import { suite } from "mocha";

// Regular class definitions - should NOT match
class HttpServer {
  private port: number;

  constructor(port: number) {
    this.port = port;
  }

  start(): void {
    console.log(`Server starting on port ${this.port}`);
  }

  stop(): void {
    console.log("Server stopping");
  }
}

class APIController {
  async handleRequest(req: Request, res: Response): Promise<void> {
    res.json({ status: "success" });
  }

  processData(data: any): any {
    return { processed: data };
  }
}

class DatabaseManager {
  private connection: any;

  async connect(): Promise<void> {
    this.connection = await createConnection();
  }

  async query(sql: string): Promise<any> {
    return this.connection.query(sql);
  }
}

// Regular function definitions - should NOT match
function processRequest(data: any): any {
  return data.toString().toUpperCase();
}

async function fetchApiData(url: string): Promise<any> {
  const response = await axios.get(url);
  return response.data;
}

function calculateSum(a: number, b: number): number {
  return a + b;
}

async function handleWebSocketMessage(message: string): Promise<void> {
  console.log(`Received: ${message}`);
}

// Regular interfaces and types - should NOT match
interface User {
  id: string;
  name: string;
  email: string;
  createdAt: Date;
}

type RequestHandler = (req: Request, res: Response) => void;

interface ApiResponse<T> {
  data: T;
  status: number;
  message: string;
}

interface DatabaseConfig {
  host: string;
  port: number;
  username: string;
  password: string;
}

// Express.js patterns - should NOT match
const app = express();

app.get("/api/users", (req: Request, res: Response) => {
  res.json({ users: [] });
});

app.post("/api/data", async (req: Request, res: Response) => {
  const result = await processRequest(req.body);
  res.json(result);
});

app.listen(3000, () => {
  console.log("Server running on port 3000");
});

// WebSocket patterns - should NOT match
const wss = new WebSocket.Server({ port: 8080 });

wss.on("connection", (ws: WebSocket) => {
  ws.on("message", (message: string) => {
    ws.send(`Echo: ${message}`);
  });
});

// Regular server patterns - should NOT match
const httpServer = createServer((req, res) => {
  res.writeHead(200, { "Content-Type": "text/plain" });
  res.end("Hello World");
});

httpServer.listen(8000);

// Regular object patterns - should NOT match
const config = {
  name: "app-config",
  description: "Application configuration",
  settings: {
    port: 3000,
    debug: true
  }
};

const userSchema = {
  type: "object",
  properties: {
    name: { type: "string" },
    age: { type: "number" }
  },
  required: ["name"]
};

const apiEndpoint = {
  name: "getUserData",
  description: "Get user data by ID",
  method: "GET",
  path: "/api/users/:id"
};

// Regular decorators (if using experimental decorators) - should NOT match
function logged(target: any, propertyName: string, descriptor: PropertyDescriptor) {
  const method = descriptor.value;
  descriptor.value = function (...args: any[]) {
    console.log(`Calling ${propertyName}`);
    return method.apply(this, args);
  };
}

class UserService {
  @logged
  async getUser(id: string): Promise<User | null> {
    return null;
  }

  @logged
  async createUser(data: Partial<User>): Promise<User> {
    return data as User;
  }
}

// Regular async patterns - should NOT match
async function main(): Promise<void> {
  const data = await fetchApiData("https://api.example.com");
  const processed = processRequest(data);
  console.log(processed);
}

async function handleHttpRequest(req: Request, res: Response): Promise<void> {
  try {
    const result = await processData(req.body);
    res.json({ success: true, data: result });
  } catch (error) {
    res.status(500).json({ error: "Internal server error" });
  }
}

// Regular utility functions - should NOT match
function formatResponse(data: any): any {
  return {
    timestamp: new Date().toISOString(),
    data: data
  };
}

function validateInput(input: any): boolean {
  return typeof input === "string" && input.length > 0;
}

function generateId(): string {
  return uuidv4();
}

function hashPassword(password: string): string {
  return bcrypt.hashSync(password, 10);
}

// Regular event handling - should NOT match
const eventEmitter = new EventEmitter();

eventEmitter.on("data", (data: any) => {
  console.log("Data received:", data);
});

eventEmitter.on("error", (error: Error) => {
  console.error("Error occurred:", error);
});

// Regular file operations - should NOT match
async function readConfigFile(path: string): Promise<any> {
  const content = await readFile(path, "utf-8");
  return JSON.parse(content);
}

function writeLogFile(message: string): void {
  fs.appendFileSync("app.log", `${new Date().toISOString()}: ${message}\n`);
}

// Regular testing patterns - should NOT match
describe("User Service", () => {
  it("should create a user", async () => {
    const userService = new UserService();
    const user = await userService.createUser({ name: "John" });
    expect(user.name).toBe("John");
  });

  it("should get a user by ID", async () => {
    const userService = new UserService();
    const user = await userService.getUser("123");
    expect(user).toBeNull();
  });
});

// Regular class with methods - should NOT match
class FileProcessor {
  async processFile(filePath: string): Promise<string> {
    const content = await readFile(filePath, "utf-8");
    return content.toUpperCase();
  }

  async saveResult(data: string, outputPath: string): Promise<void> {
    await fs.promises.writeFile(outputPath, data);
  }
}

// Regular context patterns - should NOT match
interface Context {
  user: User;
  request: Request;
  timestamp: Date;
}

function withContext<T>(fn: (ctx: Context) => T): T {
  const context: Context = {
    user: { id: "1", name: "User", email: "user@example.com", createdAt: new Date() },
    request: {} as Request,
    timestamp: new Date()
  };
  return fn(context);
}

async function handleWithContext(data: any, context: Context): Promise<any> {
  console.log(`Processing for user: ${context.user.name}`);
  return { processed: data, user: context.user.id };
}

// Regular schema definitions - should NOT match
const postSchema = {
  type: "object",
  properties: {
    title: { type: "string", maxLength: 100 },
    content: { type: "string" },
    author: { type: "string" },
    tags: { type: "array", items: { type: "string" } }
  },
  required: ["title", "content", "author"]
};

const responseSchema = {
  type: "object",
  properties: {
    success: { type: "boolean" },
    data: { type: "object" },
    error: { type: "string" }
  }
};

// Regular REST API patterns - should NOT match
const routes = {
  users: {
    list: { method: "GET", path: "/users" },
    create: { method: "POST", path: "/users" },
    get: { method: "GET", path: "/users/:id" },
    update: { method: "PUT", path: "/users/:id" },
    delete: { method: "DELETE", path: "/users/:id" }
  }
};

// Regular middleware patterns - should NOT match
function authMiddleware(req: Request, res: Response, next: Function): void {
  const token = req.headers.authorization;
  if (!token) {
    res.status(401).json({ error: "Unauthorized" });
    return;
  }
  next();
}

function loggingMiddleware(req: Request, res: Response, next: Function): void {
  console.log(`${req.method} ${req.path}`);
  next();
}

// Regular database operations - should NOT match
const dbClient = new MongoClient("mongodb://localhost:27017");

async function findUser(id: string): Promise<User | null> {
  const db = dbClient.db("myapp");
  const users = db.collection("users");
  return users.findOne({ _id: id });
}

async function createUser(userData: Partial<User>): Promise<User> {
  const db = dbClient.db("myapp");
  const users = db.collection("users");
  const result = await users.insertOne(userData);
  return { ...userData, id: result.insertedId.toString() } as User;
}

// Regular worker/job patterns - should NOT match
class JobProcessor {
  async processJob(job: any): Promise<void> {
    console.log(`Processing job: ${job.id}`);
    await new Promise(resolve => setTimeout(resolve, 1000));
    console.log(`Job ${job.id} completed`);
  }

  async scheduleJob(jobData: any): Promise<string> {
    const jobId = generateId();
    console.log(`Scheduled job: ${jobId}`);
    return jobId;
  }
}