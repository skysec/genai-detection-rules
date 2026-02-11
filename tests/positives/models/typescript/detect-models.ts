/**
 * Positive test cases for AI models detection rule
 * These should all be detected by the detect-ai-models rule
 */

// OpenAI GPT models - current and future versions
// ruleid: detect-ai-models
const response1 = await openai.chat.completions.create({ model: "gpt-4" });
// ruleid: detect-ai-models
const response2 = await openai.chat.completions.create({ model: "gpt-4-turbo" });
// ruleid: detect-ai-models
const response3 = await openai.chat.completions.create({ model: "gpt-3.5-turbo" });
// ruleid: detect-ai-models
const response4 = await openai.chat.completions.create({ model: "gpt-5" }); // Future version
// ruleid: detect-ai-models
const response5 = await openai.chat.completions.create({ model: 'gpt-4-32k' });

// OpenAI o1 models
// ruleid: detect-ai-models
const response6 = await openai.chat.completions.create({ model: "o1-preview" });
// ruleid: detect-ai-models
const response7 = await openai.chat.completions.create({ model: "o1-mini" });
// ruleid: detect-ai-models
const response8 = await openai.chat.completions.create({ model: "o1-2025" }); // Future

// OpenAI o3 models (future)
// ruleid: detect-ai-models
const response9 = await openai.chat.completions.create({ model: "o3-preview" });

// OpenAI DALL-E models
// ruleid: detect-ai-models
const image1 = await openai.images.generate({ model: "dall-e-3" });
// ruleid: detect-ai-models
const image2 = await openai.images.generate({ model: "dall-e-2" });
// ruleid: detect-ai-models
const image3 = await openai.images.generate({ model: "dall-e-4" }); // Future

// OpenAI Whisper models
// ruleid: detect-ai-models
const transcription1 = await openai.audio.transcriptions.create({ model: "whisper-1" });
// ruleid: detect-ai-models
const transcription2 = await openai.audio.transcriptions.create({ model: "whisper-2" }); // Future

// OpenAI TTS models
// ruleid: detect-ai-models
const speech1 = await openai.audio.speech.create({ model: "tts-1" });
// ruleid: detect-ai-models
const speech2 = await openai.audio.speech.create({ model: "tts-1-hd" });
// ruleid: detect-ai-models
const speech3 = await openai.audio.speech.create({ model: "tts-2" }); // Future

// OpenAI embedding models
// ruleid: detect-ai-models
const embedding1 = await openai.embeddings.create({ model: "text-embedding-3-small" });
// ruleid: detect-ai-models
const embedding2 = await openai.embeddings.create({ model: "text-embedding-3-large" });
// ruleid: detect-ai-models
const embedding3 = await openai.embeddings.create({ model: "text-embedding-ada-002" });

// Anthropic Claude models - current and future versions
// ruleid: detect-ai-models
const message1 = await anthropic.messages.create({ model: "claude-3-opus-20240229" });
// ruleid: detect-ai-models
const message2 = await anthropic.messages.create({ model: "claude-3-sonnet-20240229" });
// ruleid: detect-ai-models
const message3 = await anthropic.messages.create({ model: "claude-3-haiku-20240307" });
// ruleid: detect-ai-models
const message4 = await anthropic.messages.create({ model: "claude-3-5-sonnet-20241022" });
// ruleid: detect-ai-models
const message5 = await anthropic.messages.create({ model: "claude-3-5-haiku-20241022" });
// ruleid: detect-ai-models
const message6 = await anthropic.messages.create({ model: "claude-4-opus-20250101" }); // Future
// ruleid: detect-ai-models
const message7 = await anthropic.messages.create({ model: "claude-instant-1.2" });

// Google Gemini models - current and future versions
// ruleid: detect-ai-models
const gemini1 = await model.generateContent({ model: "gemini-pro" });
// ruleid: detect-ai-models
const gemini2 = await model.generateContent({ model: "gemini-1.5-pro" });
// ruleid: detect-ai-models
const gemini3 = await model.generateContent({ model: "gemini-1.5-flash" });
// ruleid: detect-ai-models
const gemini4 = await model.generateContent({ model: "gemini-2.0-ultra" }); // Future
// ruleid: detect-ai-models
const gemini5 = await model.generateContent({ model: "gemini-ultra" });

// Mistral models - current and future versions
// ruleid: detect-ai-models
const mistral1 = await client.chat({ model: "mistral-large-latest" });
// ruleid: detect-ai-models
const mistral2 = await client.chat({ model: "mistral-medium-latest" });
// ruleid: detect-ai-models
const mistral3 = await client.chat({ model: "mistral-small-latest" });
// ruleid: detect-ai-models
const mistral4 = await client.chat({ model: "mistral-7b" });
// ruleid: detect-ai-models
const mistral5 = await client.chat({ model: "mistral-xl" }); // Future

// Mixtral models
// ruleid: detect-ai-models
const mixtral1 = await client.chat({ model: "mixtral-8x7b" });
// ruleid: detect-ai-models
const mixtral2 = await client.chat({ model: "mixtral-8x22b" });
// ruleid: detect-ai-models
const mixtral3 = await client.chat({ model: "mixtral-16x14b" }); // Future

// Meta Llama models - current and future versions
// ruleid: detect-ai-models
const llama1 = await client.chat({ model: "llama-2-70b" });
// ruleid: detect-ai-models
const llama2 = await client.chat({ model: "llama-2-13b" });
// ruleid: detect-ai-models
const llama3 = await client.chat({ model: "llama-3-70b" });
// ruleid: detect-ai-models
const llama4 = await client.chat({ model: "llama-3.1-405b" });
// ruleid: detect-ai-models
const llama5 = await client.chat({ model: "llama-4-1t" }); // Future

// Cohere models - current and future versions
// ruleid: detect-ai-models
const cohere1 = await client.chat({ model: "command" });
// ruleid: detect-ai-models
const cohere2 = await client.chat({ model: "command-light" });
// ruleid: detect-ai-models
const cohere3 = await client.chat({ model: "command-r" });
// ruleid: detect-ai-models
const cohere4 = await client.chat({ model: "command-r-plus" });
// ruleid: detect-ai-models
const cohere5 = await client.chat({ model: "command-2" }); // Future

// Configuration objects
// ruleid: detect-ai-models
const config1 = { model: "gpt-4", temperature: 0.7 };
// ruleid: detect-ai-models
const config2 = { apiKey: "key", model: "claude-3-opus-20240229" };
// ruleid: detect-ai-models
const config3 = { model: "gemini-pro", maxTokens: 1000 };

// Variable assignments
// ruleid: detect-ai-models
const modelName1 = "gpt-4-turbo";
// ruleid: detect-ai-models
let aiModel = "claude-3-sonnet-20240229";
// ruleid: detect-ai-models
const llmModel = "gemini-1.5-pro";

// Single quotes
// ruleid: detect-ai-models
const singleQuote1 = { model: 'gpt-4' };
// ruleid: detect-ai-models
const singleQuote2 = { model: 'claude-3-opus-20240229' };
// ruleid: detect-ai-models
const singleQuote3 = { model: 'gemini-pro' };

// LangChain usage with Vercel AI SDK
import { ChatOpenAI } from "langchain/chat_models/openai";

// ruleid: detect-ai-models
const llm1 = new ChatOpenAI({ modelName: "gpt-4" });
// ruleid: detect-ai-models
const llm2 = new ChatOpenAI({ model: "gpt-4-turbo" });
