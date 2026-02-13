/**
 * Positive test cases for AI models detection rule
 * These should all be detected by the detect-ai-models rule
 */

package main

import (
	"context"
	openai "github.com/sashabaranov/go-openai"
)

func main() {
	ctx := context.Background()

	// OpenAI GPT models - current and future versions
	// ruleid: detect-ai-models
	req1 := openai.ChatCompletionRequest{Model: "gpt-4"}
	// ruleid: detect-ai-models
	req2 := openai.ChatCompletionRequest{Model: "gpt-4-turbo"}
	// ruleid: detect-ai-models
	req3 := openai.ChatCompletionRequest{Model: "gpt-3.5-turbo"}
	// ruleid: detect-ai-models
	req4 := openai.ChatCompletionRequest{Model: "gpt-5"} // Future version
	// ruleid: detect-ai-models
	req5 := openai.ChatCompletionRequest{Model: `gpt-4-32k`}

	// OpenAI o1 models
	// ruleid: detect-ai-models
	req6 := openai.ChatCompletionRequest{Model: "o1-preview"}
	// ruleid: detect-ai-models
	req7 := openai.ChatCompletionRequest{Model: "o1-mini"}
	// ruleid: detect-ai-models
	req8 := openai.ChatCompletionRequest{Model: "o1-2025"} // Future

	// OpenAI o3 models (future)
	// ruleid: detect-ai-models
	req9 := openai.ChatCompletionRequest{Model: "o3-preview"}

	// OpenAI DALL-E models
	// ruleid: detect-ai-models
	img1 := openai.ImageRequest{Model: "dall-e-3"}
	// ruleid: detect-ai-models
	img2 := openai.ImageRequest{Model: "dall-e-2"}
	// ruleid: detect-ai-models
	img3 := openai.ImageRequest{Model: "dall-e-4"} // Future

	// OpenAI Whisper models
	// ruleid: detect-ai-models
	audio1 := openai.AudioRequest{Model: "whisper-1"}
	// ruleid: detect-ai-models
	audio2 := openai.AudioRequest{Model: "whisper-2"} // Future

	// OpenAI TTS models
	// ruleid: detect-ai-models
	tts1 := openai.CreateSpeechRequest{Model: "tts-1"}
	// ruleid: detect-ai-models
	tts2 := openai.CreateSpeechRequest{Model: "tts-1-hd"}
	// ruleid: detect-ai-models
	tts3 := openai.CreateSpeechRequest{Model: "tts-2"} // Future

	// OpenAI embedding models
	// ruleid: detect-ai-models
	embed1 := openai.EmbeddingRequest{Model: "text-embedding-3-small"}
	// ruleid: detect-ai-models
	embed2 := openai.EmbeddingRequest{Model: "text-embedding-3-large"}
	// ruleid: detect-ai-models
	embed3 := openai.EmbeddingRequest{Model: "text-embedding-ada-002"}

	// Anthropic Claude models - current and future versions
	// ruleid: detect-ai-models
	claude1 := AnthropicRequest{Model: "claude-3-opus-20240229"}
	// ruleid: detect-ai-models
	claude2 := AnthropicRequest{Model: "claude-3-sonnet-20240229"}
	// ruleid: detect-ai-models
	claude3 := AnthropicRequest{Model: "claude-3-haiku-20240307"}
	// ruleid: detect-ai-models
	claude4 := AnthropicRequest{Model: "claude-3-5-sonnet-20241022"}
	// ruleid: detect-ai-models
	claude5 := AnthropicRequest{Model: "claude-3-5-haiku-20241022"}
	// ruleid: detect-ai-models
	claude6 := AnthropicRequest{Model: "claude-4-opus-20250101"} // Future
	// ruleid: detect-ai-models
	claude7 := AnthropicRequest{Model: "claude-instant-1.2"}

	// Google Gemini models - current and future versions
	// ruleid: detect-ai-models
	gemini1 := GeminiRequest{Model: "gemini-pro"}
	// ruleid: detect-ai-models
	gemini2 := GeminiRequest{Model: "gemini-1.5-pro"}
	// ruleid: detect-ai-models
	gemini3 := GeminiRequest{Model: "gemini-1.5-flash"}
	// ruleid: detect-ai-models
	gemini4 := GeminiRequest{Model: "gemini-2.0-ultra"} // Future
	// ruleid: detect-ai-models
	gemini5 := GeminiRequest{Model: "gemini-ultra"}

	// Mistral models - current and future versions
	// ruleid: detect-ai-models
	mistral1 := MistralRequest{Model: "mistral-large-latest"}
	// ruleid: detect-ai-models
	mistral2 := MistralRequest{Model: "mistral-medium-latest"}
	// ruleid: detect-ai-models
	mistral3 := MistralRequest{Model: "mistral-small-latest"}
	// ruleid: detect-ai-models
	mistral4 := MistralRequest{Model: "mistral-7b"}
	// ruleid: detect-ai-models
	mistral5 := MistralRequest{Model: "mistral-xl"} // Future

	// Mixtral models
	// ruleid: detect-ai-models
	mixtral1 := MistralRequest{Model: "mixtral-8x7b"}
	// ruleid: detect-ai-models
	mixtral2 := MistralRequest{Model: "mixtral-8x22b"}
	// ruleid: detect-ai-models
	mixtral3 := MistralRequest{Model: "mixtral-16x14b"} // Future

	// Meta Llama models - current and future versions
	// ruleid: detect-ai-models
	llama1 := LlamaRequest{Model: "llama-2-70b"}
	// ruleid: detect-ai-models
	llama2 := LlamaRequest{Model: "llama-2-13b"}
	// ruleid: detect-ai-models
	llama3 := LlamaRequest{Model: "llama-3-70b"}
	// ruleid: detect-ai-models
	llama4 := LlamaRequest{Model: "llama-3.1-405b"}
	// ruleid: detect-ai-models
	llama5 := LlamaRequest{Model: "llama-4-1t"} // Future

	// Cohere models - current and future versions
	// ruleid: detect-ai-models
	cohere1 := CohereRequest{Model: "command"}
	// ruleid: detect-ai-models
	cohere2 := CohereRequest{Model: "command-light"}
	// ruleid: detect-ai-models
	cohere3 := CohereRequest{Model: "command-r"}
	// ruleid: detect-ai-models
	cohere4 := CohereRequest{Model: "command-r-plus"}
	// ruleid: detect-ai-models
	cohere5 := CohereRequest{Model: "command-2"} // Future

	// Variable assignments
	// ruleid: detect-ai-models
	modelName1 := "gpt-4-turbo"
	// ruleid: detect-ai-models
	var aiModel = "claude-3-sonnet-20240229"
	// ruleid: detect-ai-models
	const llmModel = "gemini-1.5-pro"

	// Using backticks
	// ruleid: detect-ai-models
	bt1 := openai.ChatCompletionRequest{Model: `gpt-4`}
	// ruleid: detect-ai-models
	bt2 := AnthropicRequest{Model: `claude-3-opus-20240229`}
	// ruleid: detect-ai-models
	bt3 := GeminiRequest{Model: `gemini-pro`}

	_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = req1, req2, req3, req4, req5, req6, req7, req8, req9, img1, img2, img3, audio1, audio2, tts1, tts2, tts3, embed1, embed2, embed3, ctx, modelName1
	_, _, _, _, _, _, _, _, _, _, _ = claude1, claude2, claude3, claude4, claude5, claude6, claude7, gemini1, gemini2, gemini3, gemini4
	_, _, _, _, _, _, _, _, _, _, _ = gemini5, mistral1, mistral2, mistral3, mistral4, mistral5, mixtral1, mixtral2, mixtral3, llama1, llama2
	_, _, _, _, _, _, _, _, _ = llama3, llama4, llama5, cohere1, cohere2, cohere3, cohere4, cohere5, aiModel
	_, _, _, _ = llmModel, bt1, bt2, bt3
}

// Type definitions (for compilation)
type AnthropicRequest struct {
	Model string
}

type GeminiRequest struct {
	Model string
}

type MistralRequest struct {
	Model string
}

type LlamaRequest struct {
	Model string
}

type CohereRequest struct {
	Model string
}
