// Positive test cases for OpenAI (C#)

// ruleid: detect-openai
using OpenAI;

// ruleid: detect-openai
using OpenAI.Chat;

public class Example
{
    public async Task Run()
    {
        // ruleid: detect-openai
        var client = new OpenAIClient("sk-test");

        // ruleid: detect-openai
        ChatClient chatClient = new ChatClient("gpt-4", "sk-test");

        // ruleid: detect-openai
        var completion = await chatClient.CompleteChatAsync("Hello");

        // ruleid: detect-openai
        var stream = chatClient.CompleteChatStreamingAsync("Stream test");
    }
}
