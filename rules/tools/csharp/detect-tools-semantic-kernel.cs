// POSITIVE TEST CASES - Should be detected by the rule

// Semantic Kernel using statements - SHOULD MATCH
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Functions;
using Microsoft.SemanticKernel.Plugins;
using Microsoft.SemanticKernel.ChatCompletion;

public class ToolsTestClass
{
    // KernelFunction attribute - SHOULD MATCH
    [KernelFunction]
    public string GetWeather(string location)
    {
        return $"Weather in {location}: Sunny";
    }

    [KernelFunction("calculate")]
    public double Calculate(double a, double b, string operation)
    {
        return operation switch
        {
            "add" => a + b,
            "multiply" => a * b,
            _ => 0
        };
    }

    [KernelFunction(Description = "Get current time")]
    public DateTime GetCurrentTime()
    {
        return DateTime.Now;
    }

    public async Task TestKernelOperations()
    {
        // Kernel initialization - SHOULD MATCH
        var kernel = Kernel.CreateBuilder()
            .AddOpenAIChatCompletion("gpt-4", "api-key")
            .Build();

        var simpleKernel = new Kernel();

        // Plugin imports and additions - SHOULD MATCH
        kernel.ImportPluginFromType<WeatherPlugin>();
        kernel.ImportPluginFromType<MathPlugin>("MathFunctions");
        kernel.ImportPluginFromObject(new TimePlugin(), "TimeUtils");

        // Function invocation patterns - SHOULD MATCH
        var result1 = await kernel.InvokeAsync("GetWeather", new KernelArguments { ["location"] = "Seattle" });
        var result2 = await kernel.InvokeAsync<string>("calculate");
        var response = kernel.InvokeAsync(weatherFunction, args);

        // Service registration - SHOULD MATCH
        var builder = Kernel.CreateBuilder();
        builder.AddOpenAIChatCompletion("gpt-4", "key");
        builder.AddAzureOpenAIChatCompletion("gpt-4", "endpoint", "key");
        builder.AddChatCompletionService("openai", service);

        // Kernel function creation - SHOULD MATCH
        var promptFunction = KernelFunctionFactory.CreateFromPrompt("Summarize: {{$input}}");
        var customFunction = kernel.CreateFunctionFromPrompt(promptTemplate);

        // Function calling with arguments - SHOULD MATCH
        var arguments = new KernelArguments();
        arguments.Add("location", "New York");
        arguments.Add("date", DateTime.Today);
    }
}

// Plugin class definitions - SHOULD MATCH
public class WeatherPlugin
{
    [KernelFunction]
    public string GetCurrentWeather(string location)
    {
        return $"Current weather in {location}";
    }

    [KernelFunction("get_forecast")]
    public string GetForecast(string location, int days)
    {
        return $"{days}-day forecast for {location}";
    }
}

public class MathPlugin
{
    [KernelFunction]
    public double Add(double a, double b)
    {
        return a + b;
    }

    [KernelFunction(Description = "Multiply two numbers")]
    public double Multiply(double x, double y)
    {
        return x * y;
    }
}

// NEGATIVE TEST CASES - Should NOT be detected

// Regular using statements
using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Microsoft.Extensions.DependencyInjection;

public class RegularClass
{
    // Regular methods without KernelFunction attribute
    public string ProcessData(string input)
    {
        return input.ToUpper();
    }

    public void ConfigureServices(IServiceCollection services)
    {
        services.AddSingleton<IDataService, DataService>();
    }

    // Regular attribute usage
    [Required]
    public string Name { get; set; }

    [JsonProperty("id")]
    public int Identifier { get; set; }

    // Regular method calls
    public async Task RegularOperations()
    {
        var service = serviceProvider.GetService<IDataProcessor>();
        var result = await service.ProcessAsync(data);
        var config = Configuration.GetSection("Settings");

        var builder = new StringBuilder();
        builder.Add("some text");
    }
}