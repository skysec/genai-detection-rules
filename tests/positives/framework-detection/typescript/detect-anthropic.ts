// Positive test cases for Anthropic (TypeScript)
// ruleid: detect-anthropic
import Anthropic from "@anthropic-ai/sdk";

// ruleid: detect-anthropic
const client = new Anthropic({ apiKey: "key" });

// ruleid: detect-anthropic
const message = await client.messages.create({
  model: "claude-3-sonnet-20240229",
  messages: [{ role: "user", content: "Hello" }]
});

// ruleid: detect-anthropic
const stream = await client.messages.stream({
  model: "claude-3-opus-20240229",
  messages: [{ role: "user", content: "Stream test" }]
});
