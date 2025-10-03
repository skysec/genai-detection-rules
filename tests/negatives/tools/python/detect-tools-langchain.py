#!/usr/bin/env python3
"""
Negative test cases for LangChain tools detection.
These patterns should NOT be detected by the detect-tools-langchain.yaml rule.
"""

import os
import json
import requests
from typing import List, Dict, Any

# Generic classes with similar method names (but not LangChain)
class GenericTool:
    def __init__(self, name="generic"):
        self.name = name

    def run(self, input_data):
        # Generic run method (not LangChain)
        return f"Processing: {input_data}"

    def _run(self, args):
        # Generic _run method (not LangChain)
        return "generic result"

    async def _arun(self, args):
        # Generic async run (not LangChain)
        return "async result"

# Generic agent classes (not LangChain)
class SimpleAgent:
    def __init__(self):
        self.tools = []

    def run(self, query):
        # Generic agent run (not LangChain)
        return f"Agent response to: {query}"

    def invoke(self, query):
        # Generic invoke method (not LangChain)
        return f"Invoked with: {query}"

    def __call__(self, query):
        # Generic callable (not LangChain)
        return self.run(query)

# Generic initialization functions (not LangChain)
def initialize_system(tools=None, config=None):
    """Generic system initialization (not LangChain)"""
    return SimpleAgent()

def setup_agent(tools=None):
    """Generic agent setup (not LangChain)"""
    return {"agent": SimpleAgent(), "tools": tools or []}

# Generic tool registration patterns
class ToolRegistry:
    def __init__(self):
        self.tools = []

    def register_function(self, func):
        # Generic function registration (not LangChain)
        self.tools.append(func)

    def call_function(self, name, args):
        # Generic function calling (not LangChain)
        return f"Called {name} with {args}"

    def add_tool(self, tool):
        # Generic tool addition (not LangChain)
        self.tools.append(tool)

# Generic API client
class APIClient:
    def __init__(self, endpoint):
        self.endpoint = endpoint

    def run(self, request):
        # Generic API run (not LangChain)
        return f"API response for: {request}"

    def invoke(self, method, params):
        # Generic API invoke (not LangChain)
        return {"result": "success", "method": method}

# Generic conversation patterns
class ConversationManager:
    def __init__(self):
        self.history = []

    def run(self, message):
        # Generic conversation run (not LangChain)
        response = f"Response to: {message}"
        self.history.append((message, response))
        return response

# Generic function calling patterns
def call_function(func_name, args):
    """Generic function calling (not LangChain)"""
    return f"Function {func_name} called with {args}"

def register_function(name, func):
    """Generic function registration (not LangChain)"""
    return {"name": name, "function": func}

# Generic database operations
class DatabaseManager:
    def __init__(self, connection_string):
        self.connection = connection_string

    def run(self, query):
        # Generic database run (not LangChain)
        return f"Query result: {query}"

    def _run(self, sql):
        # Generic database _run (not LangChain)
        return {"rows": [], "count": 0}

# Generic file operations
class FileManager:
    def run(self, operation, filename):
        # Generic file run (not LangChain)
        return f"File operation {operation} on {filename}"

    def _run(self, command):
        # Generic file _run (not LangChain)
        return "file operation complete"

# Generic web scraping
class WebScraper:
    def run(self, url):
        # Generic scraper run (not LangChain)
        return f"Scraped content from {url}"

    def _run(self, target):
        # Generic scraper _run (not LangChain)
        return "scraped data"

# Generic calculator
class Calculator:
    def run(self, expression):
        # Generic calculator run (not LangChain)
        try:
            return str(eval(expression))
        except:
            return "Error"

    def _run(self, calc):
        # Generic calculator _run (not LangChain)
        return "42"

# Generic workflow systems
class WorkflowEngine:
    def __init__(self):
        self.steps = []

    def run(self, workflow):
        # Generic workflow run (not LangChain)
        return f"Executed workflow: {workflow}"

    def invoke(self, step_name):
        # Generic workflow invoke (not LangChain)
        return f"Step {step_name} completed"

# Generic imports with similar names
from some_library import BaseTool as GenericBaseTool
from another_lib import Agent as GenericAgent

# Generic variable assignments (should not be detected)
tools = ["tool1", "tool2", "tool3"]
agent = "string agent"
base_tool = "not a class"

# Generic function parameters
def process_data(data, tools=None, agent=None):
    """Generic function with tools/agent parameters (not LangChain)"""
    if tools:
        print(f"Processing with tools: {tools}")
    return data

def create_service(name, tools=None):
    """Generic service creation (not LangChain)"""
    return {"name": name, "tools": tools or []}

# Generic class methods
class GenericProcessor:
    def __init__(self):
        self.tools = []

    def run(self, input_data):
        # Generic run
        return f"Processed: {input_data}"

    def invoke(self, command):
        # Generic invoke
        return f"Invoked: {command}"

    def __call__(self, data):
        # Generic callable
        return self.run(data)

# Generic HTTP client
class HTTPClient:
    def run(self, request):
        # Generic HTTP run (not LangChain)
        return requests.get(request.url)

    def invoke(self, method, url, data=None):
        # Generic HTTP invoke (not LangChain)
        return {"status": 200, "data": "response"}

# Generic task runner
class TaskRunner:
    def __init__(self):
        self.tasks = []

    def run(self, task):
        # Generic task run (not LangChain)
        return f"Task {task} completed"

    def _run(self, task_id):
        # Generic task _run (not LangChain)
        return {"task_id": task_id, "status": "done"}

    async def _arun(self, async_task):
        # Generic async task (not LangChain)
        return f"Async task {async_task} completed"

# Generic configuration
CONFIG = {
    "tools": ["tool1", "tool2"],
    "agent_type": "simple",
    "timeout": 30
}

# Generic utilities
def run_command(command):
    """Generic command runner (not LangChain)"""
    return os.system(command)

def invoke_service(service_name, params):
    """Generic service invoker (not LangChain)"""
    return f"Service {service_name} invoked with {params}"

# Generic plugin system
class PluginManager:
    def __init__(self):
        self.plugins = []

    def run(self, plugin_name):
        # Generic plugin run (not LangChain)
        return f"Plugin {plugin_name} executed"

    def register_function(self, func):
        # Generic plugin registration (not LangChain)
        self.plugins.append(func)

    def call_function(self, name):
        # Generic plugin call (not LangChain)
        return f"Plugin function {name} called"

# Generic machine learning
class MLModel:
    def run(self, features):
        # Generic ML run (not LangChain)
        return [0.1, 0.9]  # Mock prediction

    def _run(self, input_data):
        # Generic ML _run (not LangChain)
        return {"prediction": 0.8, "confidence": 0.95}

# Generic data structures
tool_list = []
agent_config = {}
function_registry = {}

# Generic functions that might have similar names
def run_analysis(data, tools=None):
    """Generic analysis function (not LangChain)"""
    return {"result": "analysis complete"}

def invoke_pipeline(steps, agent=None):
    """Generic pipeline invoker (not LangChain)"""
    return "pipeline complete"

# Generic chain pattern (not LangChain)
class ProcessingChain:
    def __init__(self, steps):
        self.steps = steps

    def run(self, input_data):
        result = input_data
        for step in self.steps:
            result = step(result)
        return result

    def invoke(self, data):
        return self.run(data)

# Generic logging
import logging

logger = logging.getLogger(__name__)

def log_tool_usage(tool_name, result):
    """Generic logging (not LangChain)"""
    logger.info(f"Tool {tool_name} returned: {result}")

# Generic testing utilities
class TestRunner:
    def run(self, test_case):
        # Generic test run (not LangChain)
        return {"passed": True, "test": test_case}

    def invoke(self, test_suite):
        # Generic test invoke (not LangChain)
        return "all tests passed"

# Generic business logic
def process_order(order_data, tools=None):
    """Generic business process (not LangChain)"""
    return {"order_id": 123, "status": "processed"}

def handle_request(request, agent=None):
    """Generic request handler (not LangChain)"""
    return {"response": "handled", "agent": agent}