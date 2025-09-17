// Negative test cases for C# MCP server detection - these should NOT trigger the rule

using System;
using System.ComponentModel;
using System.Threading.Tasks;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Mvc;

namespace NonMcpServerTests
{
    // Test case 1: Regular .NET service registration (not MCP)
    public class RegularServiceSetup
    {
        public void ConfigureServices(IServiceCollection services)
        {
            services.AddControllers();
            services.AddScoped<IMyService, MyService>();
            services.AddLogging();
        }
    }

    // Test case 2: Regular ASP.NET Core controllers
    [ApiController]
    [Route("api/[controller]")]
    public class CalculatorController : ControllerBase
    {
        [HttpPost]
        [Description("Adds two numbers")]
        public ActionResult<int> Add(int a, int b)
        {
            return a + b;
        }

        [HttpGet]
        [Description("Gets calculator status")]
        public async Task<ActionResult<string>> GetStatus()
        {
            await Task.Delay(1);
            return "Calculator is running";
        }
    }

    // Test case 3: Regular classes with Description attributes (not MCP-specific)
    public class DocumentProcessor
    {
        [Description("Processes documents")]
        public string ProcessDocument(string content)
        {
            return content.ToUpper();
        }

        [Description("Validates document format")]
        public async Task<bool> ValidateAsync(string document)
        {
            await Task.Delay(1);
            return !string.IsNullOrEmpty(document);
        }
    }

    // Test case 4: Regular static classes with methods
    public static class StringUtils
    {
        [Description("Capitalizes a string")]
        public static string Capitalize(string input)
        {
            return char.ToUpper(input[0]) + input.Substring(1);
        }

        [Description("Reverses a string")]
        public static async Task<string> ReverseAsync(string input)
        {
            await Task.Delay(1);
            return new string(input.ToCharArray().Reverse().ToArray());
        }
    }

    // Test case 5: Custom attributes that might look similar but are not MCP
    [CustomToolType]
    public class CustomTools
    {
        [CustomTool]
        [Description("Custom tool method")]
        public static string ProcessData(string data)
        {
            return data;
        }
    }

    public class CustomToolTypeAttribute : Attribute { }
    public class CustomToolAttribute : Attribute { }

    // Test case 6: Regular host builder without MCP
    public class RegularHostBuilder
    {
        public void CreateHost()
        {
            var builder = Host.CreateDefaultBuilder();
            builder.ConfigureServices(services =>
            {
                services.AddHttpClient();
                services.AddMemoryCache();
            });
        }
    }

    // Test case 7: Message classes that are not MCP ChatMessage
    public class CustomMessage
    {
        public CustomMessage(string role, string content)
        {
            Role = role;
            Content = content;
        }

        public string Role { get; set; }
        public string Content { get; set; }
    }

    public class MessageHandler
    {
        public CustomMessage CreateMessage(string role, string content)
        {
            return new CustomMessage(role, content);
        }
    }

    // Test case 8: Regular inheritance (not MCP base classes)
    public class CustomProcessor : BaseProcessor
    {
        public override Task<string> ProcessAsync()
        {
            return Task.FromResult("Processed");
        }
    }

    public abstract class BaseProcessor
    {
        public abstract Task<string> ProcessAsync();
    }

    // Test case 9: Regular factory methods
    public class ComponentFactory
    {
        public static IProcessor CreateProcessor(string type)
        {
            return new DefaultProcessor();
        }

        public static IValidator CreateValidator(string name)
        {
            return new DefaultValidator();
        }
    }

    public interface IProcessor
    {
        Task<string> ProcessAsync();
    }

    public interface IValidator
    {
        bool Validate(object input);
    }

    public class DefaultProcessor : IProcessor
    {
        public Task<string> ProcessAsync()
        {
            return Task.FromResult("Default processing");
        }
    }

    public class DefaultValidator : IValidator
    {
        public bool Validate(object input)
        {
            return input != null;
        }
    }

    // Test case 10: Regular service implementations
    public interface IMyService
    {
        Task<string> ProcessAsync(string input);
    }

    public class MyService : IMyService
    {
        [Description("Processes input data")]
        public async Task<string> ProcessAsync(string input)
        {
            await Task.Delay(100);
            return $"Processed: {input}";
        }
    }

    // Test case 11: WPF/WinForms components with Description attributes
    public class UIComponent
    {
        [Description("Button click handler")]
        public void OnButtonClick(object sender, EventArgs e)
        {
            // Handle button click
        }

        [Description("Updates UI display")]
        public async Task UpdateDisplayAsync()
        {
            await Task.Delay(50);
            // Update UI
        }
    }

    // Test case 12: Entity Framework or ORM models
    public class DataModel
    {
        [Description("User identifier")]
        public int Id { get; set; }

        [Description("User name")]
        public string Name { get; set; }

        [Description("User email")]
        public string Email { get; set; }
    }

    // Test case 13: Configuration classes
    public class AppSettings
    {
        [Description("Database connection string")]
        public string ConnectionString { get; set; }

        [Description("API endpoint URL")]
        public string ApiUrl { get; set; }
    }
}