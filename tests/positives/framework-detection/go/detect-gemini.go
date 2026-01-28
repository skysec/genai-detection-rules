package main

// Positive test cases for Gemini (Go)

// ruleid: detect-gemini
import "github.com/google/generative-ai-go/genai"

func main() {
	// ruleid: detect-gemini
	client, _ := genai.NewClient(ctx, option.WithAPIKey("key"))

	// ruleid: detect-gemini
	model := client.GenerativeModel("gemini-pro")

	// ruleid: detect-gemini
	resp, _ := model.GenerateContent(ctx, genai.Text("Hello"))
}
