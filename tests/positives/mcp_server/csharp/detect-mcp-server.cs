// Positive test cases for C# MCP server detection

using System;
using System.ComponentModel;
using System.Threading.Tasks;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using ModelContextProtocol;
using ModelContextProtocol.Server;
using ModelContextProtocol.Protocol;
using ModelContextProtocol.AspNetCore;
using ModelContextProtocol.Core;

namespace McpServerTests
{
    // Test case 1: Basic MCP service registration
    public class BasicServerSetup
    {
        public void ConfigureServices()
        {
            var builder = Host.CreateApplicationBuilder();
            builder.Services.AddMcpServer();
        }
    }

    // Test case 2: MCP service with transport configuration
    public class ServerWithTransport
    {
        public void ConfigureServices()
        {
            var builder = Host.CreateApplicationBuilder();
            builder.Services
                .AddMcpServer()
                .WithStdioServerTransport()
                .WithToolsFromAssembly();
        }
    }

    // Test case 3: MCP tool class with attributes
    [McpServerToolType]
    public static class CalculatorTools
    {
        [McpServerTool]
        [Description("Adds two numbers together")]
        public static string Add(
            [Description("First number")] int a,
            [Description("Second number")] int b)
        {
            return (a + b).ToString();
        }

        [McpServerTool]
        [Description("Multiplies two numbers")]
        public static async Task<string> MultiplyAsync(
            [Description("First number")] int x,
            [Description("Second number")] int y)
        {
            await Task.Delay(1);
            return (x * y).ToString();
        }
    }

    // Test case 4: MCP prompt class with attributes
    [McpServerPromptType]
    public static class TextPrompts
    {
        [McpServerPrompt]
        [Description("Creates a writing prompt")]
        public static ChatMessage CreateWritingPrompt(
            [Description("Topic for the writing")] string topic)
        {
            return new ChatMessage(ChatRole.User, $"Write about: {topic}");
        }

        [McpServerPrompt]
        [Description("Generates a code review prompt")]
        public static async Task<ChatMessage> CodeReviewPrompt(
            [Description("Code to review")] string code)
        {
            await Task.Delay(1);
            return new ChatMessage(ChatRole.System, $"Please review this code: {code}");
        }
    }

    // Test case 5: MCP resource class with attributes
    [McpServerResourceType]
    public static class FileResources
    {
        [McpServerResource]
        [Description("Reads a file resource")]
        public static async Task<string> ReadFile(
            [Description("File path to read")] string filePath)
        {
            return await System.IO.File.ReadAllTextAsync(filePath);
        }

        [McpServerResource]
        [Description("Lists directory contents")]
        public static string ListDirectory(
            [Description("Directory path")] string directoryPath)
        {
            return string.Join(",", System.IO.Directory.GetFiles(directoryPath));
        }
    }

    // Test case 6: ASP.NET Core MCP integration
    public class WebServerSetup
    {
        public void Configure(IApplicationBuilder app)
        {
            app.MapMcp();
        }

        public void ConfigureServices(IServiceCollection services)
        {
            services.AddMcpServer()
                    .WithHttpServerTransport()
                    .WithPromptsFromAssembly();
        }
    }

    // Test case 7: MCP base class inheritance
    public class CustomTool : McpServerTool
    {
        public override Task<string> ExecuteAsync()
        {
            return Task.FromResult("Custom tool result");
        }
    }

    public class CustomPrompt : McpServerPrompt
    {
        public override ChatMessage Generate()
        {
            return new ChatMessage(ChatRole.Assistant, "Custom prompt");
        }
    }

    public class CustomResource : McpServerResource
    {
        public override Task<string> ReadAsync()
        {
            return Task.FromResult("Custom resource data");
        }
    }

    // Test case 8: MCP static factory methods
    public class FactoryMethods
    {
        public void CreateMcpComponents()
        {
            var prompt = McpServerPrompt.Create("test-prompt", "Test description");
            var resource = McpServerResource.Create("test-resource", "Resource description");
            var tool = McpServerTool.Create("test-tool", "Tool description");
        }
    }

    // Test case 9: Host builder patterns
    public class HostBuilderPatterns
    {
        public void CreateHost()
        {
            var builder1 = Host.CreateApplicationBuilder(args: null);
            var builder2 = Host.CreateEmptyApplicationBuilder(settings: null);
        }
    }

    // Test case 10: Complex MCP setup with dependency injection
    [McpServerToolType]
    public static class ServiceIntegratedTools
    {
        [McpServerTool]
        [Description("Tool that uses dependency injection")]
        public static async Task<string> ProcessWithService(
            [Description("Input text")] string input,
            IMyService myService)
        {
            return await myService.ProcessAsync(input);
        }
    }

    public interface IMyService
    {
        Task<string> ProcessAsync(string input);
    }
}