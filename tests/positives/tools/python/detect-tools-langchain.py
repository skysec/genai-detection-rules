#!/usr/bin/env python3
"""
Positive test cases for LangChain tools detection.
These patterns should be detected by the detect-tools-langchain.yaml rule.
"""

# Core LangChain tools imports - VERY HIGH CONFIDENCE
from langchain.tools import BaseTool
from langchain.tools import DuckDuckGoSearchRun
from langchain.tools import WikipediaQueryRun
from langchain.tools import ShellTool
from langchain.tools import PythonREPLTool
import langchain.tools

# Agent imports with tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.agents import create_react_agent
from langchain.agents import AgentExecutor

# Additional tool imports
from langchain.tools.file_management import (
    ReadFileTool,
    WriteFileTool,
    ListDirectoryTool
)

# Tool class definitions - VERY HIGH CONFIDENCE
class CalculatorTool(BaseTool):
    name = "Calculator"
    description = "Useful for math calculations"

    def _run(self, expression: str) -> str:
        """Execute the calculation"""
        try:
            result = eval(expression)
            return str(result)
        except Exception as e:
            return f"Error: {e}"

class WeatherTool(BaseTool):
    name = "get_weather"
    description = "Get current weather for a location"

    def _run(self, location: str) -> str:
        """Get weather information"""
        # Mock weather API call
        return f"Weather in {location}: 72°F, sunny"

class CustomSearchTool(BaseTool):
    name = "custom_search"
    description = "Search for information online"

    async def _arun(self, query: str) -> str:
        """Async search implementation"""
        # Mock async search
        return f"Search results for: {query}"

# Agent initialization patterns - HIGH CONFIDENCE
tools = [
    DuckDuckGoSearchRun(),
    WikipediaQueryRun(),
    CalculatorTool(),
    WeatherTool()
]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.REACT_DOCSTORE,
    verbose=True
)

search_agent = initialize_agent(
    tools=[DuckDuckGoSearchRun()],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION
)

# LangChain specific tool patterns - HIGH CONFIDENCE
agent.tools.append(CustomSearchTool())
custom_tools = [CalculatorTool(), WeatherTool()]

research_agent = initialize_agent(
    tools=custom_tools,
    llm=llm
)

# Real-world usage examples
def setup_agent_with_tools():
    # Define custom tools
    calculator = CalculatorTool()
    weather = WeatherTool()
    search = DuckDuckGoSearchRun()

    # Create tool list
    tools = [calculator, weather, search]

    # Initialize agent
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )

    return agent

def create_file_management_agent():
    # File management tools
    file_tools = [
        ReadFileTool(),
        WriteFileTool(),
        ListDirectoryTool()
    ]

    # Agent with file tools
    agent = initialize_agent(
        tools=file_tools,
        llm=llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION
    )

    return agent

def python_repl_agent():
    # Python REPL tool
    python_tool = PythonREPLTool()

    # Agent with Python execution capability
    agent = initialize_agent(
        tools=[python_tool],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION
    )

    return agent

# Advanced tool implementations
class DatabaseTool(BaseTool):
    name = "database_query"
    description = "Query the database for information"

    def _run(self, query: str) -> str:
        """Execute database query"""
        # Mock database operation
        return f"Database result for: {query}"

class APITool(BaseTool):
    name = "api_call"
    description = "Make API calls to external services"

    def _run(self, endpoint: str, params: str) -> str:
        """Make API call"""
        # Mock API call
        return f"API response from {endpoint} with {params}"

class EmailTool(BaseTool):
    name = "send_email"
    description = "Send emails to recipients"

    async def _arun(self, recipient: str, subject: str, body: str) -> str:
        """Send email asynchronously"""
        # Mock email sending
        return f"Email sent to {recipient} with subject: {subject}"

# Multi-agent setup with tools
def create_multi_agent_system():
    # Research agent
    research_tools = [
        DuckDuckGoSearchRun(),
        WikipediaQueryRun()
    ]

    research_agent = initialize_agent(
        tools=research_tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION
    )

    # Data agent
    data_tools = [
        DatabaseTool(),
        PythonREPLTool()
    ]

    data_agent = initialize_agent(
        tools=data_tools,
        llm=llm,
        agent=AgentType.REACT_DOCSTORE
    )

    # Communication agent
    comm_tools = [
        EmailTool(),
        APITool()
    ]

    comm_agent = initialize_agent(
        tools=comm_tools,
        llm=llm
    )

    return research_agent, data_agent, comm_agent

# Custom tool with complex logic
class WebScrapingTool(BaseTool):
    name = "web_scraper"
    description = "Scrape content from web pages"

    def _run(self, url: str) -> str:
        """Scrape web content"""
        import requests
        from bs4 import BeautifulSoup

        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.content, 'html.parser')
            return soup.get_text()[:1000]  # First 1000 chars
        except Exception as e:
            return f"Error scraping {url}: {e}"

# Tool composition patterns
def compose_specialized_agent():
    # Combine multiple tool types
    all_tools = [
        CalculatorTool(),
        WeatherTool(),
        WebScrapingTool(),
        DatabaseTool(),
        PythonREPLTool(),
        DuckDuckGoSearchRun()
    ]

    # Specialized agent
    agent = initialize_agent(
        tools=all_tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        max_iterations=5
    )

    return agent

# Dynamic tool loading
def load_tools_dynamically():
    available_tools = []

    # Add tools based on configuration
    available_tools.append(CalculatorTool())
    available_tools.append(WeatherTool())

    # Conditionally add tools
    if need_search:
        available_tools.append(DuckDuckGoSearchRun())

    if need_files:
        available_tools.extend([
            ReadFileTool(),
            WriteFileTool()
        ])

    # Create agent with dynamic tools
    agent = initialize_agent(
        tools=available_tools,
        llm=llm
    )

    return agent

# Tool validation and error handling
class ValidatedTool(BaseTool):
    name = "validated_tool"
    description = "Tool with input validation"

    def _run(self, input_data: str) -> str:
        """Validated tool execution"""
        # Input validation
        if not input_data or len(input_data) < 3:
            return "Error: Input too short"

        # Process input
        result = f"Processed: {input_data.upper()}"
        return result

# Specialized agents for different domains
def create_domain_specific_agents():
    # Finance agent
    finance_tools = [
        CalculatorTool(),
        DatabaseTool()
    ]

    finance_agent = initialize_agent(
        tools=finance_tools,
        llm=llm
    )

    # Research agent
    research_tools = [
        DuckDuckGoSearchRun(),
        WikipediaQueryRun(),
        WebScrapingTool()
    ]

    research_agent = initialize_agent(
        tools=research_tools,
        llm=llm
    )

    return finance_agent, research_agent

# Tools array patterns
basic_tools = [CalculatorTool(), WeatherTool()]
search_tools = [DuckDuckGoSearchRun(), WikipediaQueryRun()]
file_tools = [ReadFileTool(), WriteFileTool(), ListDirectoryTool()]

# Agent with tools array
comprehensive_agent = initialize_agent(
    tools=basic_tools + search_tools + file_tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION
)