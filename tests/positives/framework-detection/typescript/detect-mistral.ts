// Positive test cases for Mistral (TypeScript)
// ruleid: detect-mistral
import { Mistral } from "@mistralai/mistralai";

// ruleid: detect-mistral
const client = new Mistral({ apiKey: "key" });

// ruleid: detect-mistral
const response = await client.chat({ model: "mistral-large-latest", messages: [] });

// ruleid: detect-mistral
const embeddings = await client.embeddings.create({ model: "mistral-embed", input: ["text"] });
