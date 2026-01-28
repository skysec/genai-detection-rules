package main

// Positive test cases for OpenAI (Go)

// ruleid: detect-openai
import "github.com/sashabaranov/go-openai"

func main() {
	// ruleid: detect-openai
	client := openai.NewClient("sk-test")

	// ruleid: detect-openai
	resp, _ := client.CreateChatCompletion(ctx, openai.ChatCompletionRequest{
		Model: openai.GPT4,
	})

	// ruleid: detect-openai
	client.CreateEmbeddings(ctx, openai.EmbeddingRequest{})
}
