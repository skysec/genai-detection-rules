// Positive test cases for Gemini (TypeScript)
// ruleid: detect-gemini
import { GoogleGenerativeAI } from "@google/generative-ai";

// ruleid: detect-gemini
const genAI = new GoogleGenerativeAI("API_KEY");

// ruleid: detect-gemini
const model = genAI.getGenerativeModel({ model: "gemini-pro" });

// ruleid: detect-gemini
const result = await model.generateContent("Hello");

// ruleid: detect-gemini
const chat = model.startChat();

// ruleid: detect-gemini
const response = await chat.sendMessage("Hi");
