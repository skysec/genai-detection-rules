/**
 * Positive test cases for OpenAI detection rule (TypeScript)
 * These should all be detected by the detect-openai rule
 */

// Test 1: Basic import
// ruleid: detect-openai
import "openai";

// Test 2: Default import
// ruleid: detect-openai
import OpenAI from "openai";

// Test 3: Named import
// ruleid: detect-openai
import { OpenAI, AzureOpenAI } from "openai";

// Test 4: Require import
// ruleid: detect-openai
const OpenAI = require("openai");

// Test 5: Client instantiation
// ruleid: detect-openai
const client = new OpenAI({ apiKey: "sk-test" });

// Test 6: Azure OpenAI
// ruleid: detect-openai
const azureClient = new AzureOpenAI({
  apiKey: "key",
  endpoint: "https://example.openai.azure.com/"
});

// Test 7: Chat completions create
// ruleid: detect-openai
const completion = await client.chat.completions.create({
  model: "gpt-4",
  messages: [{ role: "user", content: "Hello" }]
});

// Test 8: Chat completions stream
// ruleid: detect-openai
const stream = await client.chat.completions.stream({
  model: "gpt-4",
  messages: [{ role: "user", content: "Hello" }],
  stream: true
});

// Test 9: Embeddings create
// ruleid: detect-openai
const embedding = await client.embeddings.create({
  model: "text-embedding-3-small",
  input: "Text to embed"
});

// Test 10: Audio transcription
// ruleid: detect-openai
const transcription = await client.audio.transcriptions.create({
  model: "whisper-1",
  file: audioFile
});

// Test 11: Image generation
// ruleid: detect-openai
const image = await client.images.generate({
  model: "dall-e-3",
  prompt: "A sunset"
});

// Test 12: Models list
// ruleid: detect-openai
const models = await client.models.list();

// Test 13: Fine-tuning
// ruleid: detect-openai
const fineTune = await client.fineTuning.jobs.create({
  training_file: "file-abc",
  model: "gpt-3.5-turbo"
});

// Test 14: Assistants create
// ruleid: detect-openai
const assistant = await client.beta.assistants.create({
  name: "Assistant",
  model: "gpt-4"
});
