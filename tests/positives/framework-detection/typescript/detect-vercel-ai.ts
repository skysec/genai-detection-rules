// Positive test cases for Vercel AI (TypeScript)
// ruleid: detect-vercel-ai
import { generateText, streamText } from "ai";

// ruleid: detect-vercel-ai
import { createOpenAI } from "@ai-sdk/openai";

// ruleid: detect-vercel-ai
const { text } = await generateText({ model: openai("gpt-4"), prompt: "Hello" });

// ruleid: detect-vercel-ai
const result = await streamText({ model: openai("gpt-4"), prompt: "Stream" });

// ruleid: detect-vercel-ai
const openai = createOpenAI({ apiKey: "key" });

// ruleid: detect-vercel-ai
const { useChat } = require("ai/react");
