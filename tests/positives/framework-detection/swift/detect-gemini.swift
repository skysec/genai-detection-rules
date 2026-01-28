// Positive test cases for Gemini (Swift)

// ruleid: detect-gemini
import GoogleGenerativeAI

// ruleid: detect-gemini
let model = GenerativeModel(name: "gemini-pro", apiKey: "key")

// ruleid: detect-gemini
let response = try await model.generateContent("Hello")

// ruleid: detect-gemini
let chat = model.startChat()
