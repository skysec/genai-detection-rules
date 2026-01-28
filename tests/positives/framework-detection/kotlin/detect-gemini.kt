// Positive test cases for Gemini (Kotlin)

// ruleid: detect-gemini
import com.google.ai.client.generativeai.GenerativeModel

// ruleid: detect-gemini
val model = GenerativeModel(modelName = "gemini-pro", apiKey = "key")

// ruleid: detect-gemini
val response = model.generateContent("Hello")

// ruleid: detect-gemini
val chat = model.startChat()
