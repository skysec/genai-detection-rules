# POSITIVE TEST CASES - Should be detected by the rule

# =============================================================================
# MCP Core Imports - SHOULD MATCH (HIGH CONFIDENCE)
# =============================================================================

import mcp
from mcp import Server
from mcp import Context
from mcp.server import BaseServer
from mcp.server import RequestHandler
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp import Context
from mcp.server.session import ServerSession
from mcp.server.session import Session

# =============================================================================
# FastMCP Class Instantiation - SHOULD MATCH (HIGH CONFIDENCE)
# =============================================================================

# Basic FastMCP instantiation
mcp_server = FastMCP("weather-server")
weather_app = FastMCP("weather")
server = FastMCP("my-server")

# FastMCP with configuration
mcp = FastMCP(
    "advanced-server",
    description="Advanced MCP server with tools"
)

# =============================================================================
# MCP Tool Decorators - SHOULD MATCH (HIGH CONFIDENCE)
# =============================================================================

# Standard @server.tool decorator
@server.tool
def get_weather(location: str) -> str:
    """Get weather for a location."""
    return f"Weather for {location}"

@server.tool
async def fetch_data(query: str) -> dict:
    """Fetch data asynchronously."""
    return {"data": query}

# @mcp.tool decorator variants
@mcp.tool
def calculate(expression: str) -> str:
    """Calculate mathematical expression."""
    return str(eval(expression))

@mcp.tool
async def process_request(data: dict) -> dict:
    """Process request data."""
    return {"processed": data}

@mcp.tool()
def simple_tool() -> str:
    """Simple tool with no parameters."""
    return "Hello, World!"

@mcp.tool()
async def async_tool(value: int) -> int:
    """Async tool with parameters."""
    return value * 2

# Tool with structured parameters
@mcp.tool
def complex_tool(name: str, age: int, active: bool = True) -> dict:
    """Complex tool with multiple parameters."""
    return {"name": name, "age": age, "active": active}

# =============================================================================
# MCP Resource Decorators - SHOULD MATCH (HIGH CONFIDENCE)
# =============================================================================

@server.resource("file://{path}")
def read_file(path: str) -> str:
    """Read file content."""
    return f"Content of {path}"

@mcp.resource("documents://{doc_id}")
def get_document(doc_id: str) -> dict:
    """Get document by ID."""
    return {"id": doc_id, "content": "Document content"}

@mcp.resource("config://settings")
async def get_settings() -> dict:
    """Get application settings."""
    return {"setting1": "value1", "setting2": "value2"}

# =============================================================================
# MCP Prompt Decorators - SHOULD MATCH (HIGH CONFIDENCE)
# =============================================================================

@server.prompt
def code_review_prompt(code: str) -> str:
    """Generate code review prompt."""
    return f"Please review this code: {code}"

@mcp.prompt
def data_analysis_prompt(dataset: str) -> str:
    """Generate data analysis prompt."""
    return f"Analyze this dataset: {dataset}"

@mcp.prompt()
def simple_prompt() -> str:
    """Simple prompt template."""
    return "Please help me with this task."

# =============================================================================
# MCP Server Running Patterns - SHOULD MATCH (HIGH CONFIDENCE)
# =============================================================================

# Basic server run
mcp.run()
server.run()
weather_server.run()

# Main block patterns
if __name__ == "__main__":
    mcp.run()

if __name__ == "__main__":
    server.run()

if __name__ == "__main__":
    weather_app = FastMCP("weather")
    weather_app.run()

# =============================================================================
# MCP Context Usage - SHOULD MATCH (MEDIUM CONFIDENCE)
# =============================================================================

from mcp.server.fastmcp import Context
from mcp import Context

# Functions with Context parameter
def progress_tool(task: str, context: Context) -> str:
    """Tool that reports progress."""
    context.progress_token = "task_progress"
    return f"Processing {task}"

async def advanced_tool(data: dict, context: Context) -> dict:
    """Advanced tool with context."""
    context.progress_token = "advanced_progress"
    return {"result": data}

@mcp.tool
def context_aware_tool(input_data: str, context: Context) -> str:
    """Tool that uses context for progress reporting."""
    # Use context for progress reporting
    return f"Processed: {input_data}"

# =============================================================================
# NEGATIVE TEST CASES - Should NOT be detected
# =============================================================================

# Regular Python imports - should NOT match
import json
import asyncio
import httpx
import os
from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass
from pydantic import BaseModel

# Regular class definitions - should NOT match
class DataProcessor:
    def __init__(self):
        self.data = []

    def process(self, input_data):
        return input_data.upper()

class WebServer:
    def __init__(self, port):
        self.port = port

    def start(self):
        print(f"Starting server on port {self.port}")

class APIHandler:
    def handle_request(self, request):
        return {"status": "success"}

# Regular function definitions - should NOT match
def process_data(data):
    return data * 2

async def fetch_data(url):
    return {"data": "example"}

def handle_http_request(request):
    return {"status": 200, "body": "OK"}

def calculate_result(x, y):
    return x + y

# Regular decorators - should NOT match
@dataclass
class User:
    name: str
    email: str
    age: int

@property
def full_name(self):
    return f"{self.first_name} {self.last_name}"

from functools import wraps

@wraps
def logged(func):
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__}")
        return func(*args, **kwargs)
    return wrapper

@logged
def some_function():
    return "result"

# Regular async patterns - should NOT match
async def main():
    data = await fetch_api_data()
    return process_results(data)

async def handle_request(request):
    return {"processed": request}

# Flask/FastAPI patterns - should NOT match
from flask import Flask
from fastapi import FastAPI

app = Flask(__name__)
api = FastAPI()

@app.route("/api/data")
def get_data():
    return {"data": "value"}

@api.get("/items/{item_id}")
async def read_item(item_id: int):
    return {"item_id": item_id}

# Regular server operations - should NOT match
server = WebServer(8080)
server.start()

http_server.run()
app.run()
api.run()

# Regular tool/utility functions - should NOT match
def tool_helper(data):
    return data.strip()

def resource_loader(path):
    with open(path) as f:
        return f.read()

def prompt_formatter(text):
    return f"Prompt: {text}"

# Regular context usage - should NOT match
def function_with_context(data, context=None):
    if context:
        print(f"Context: {context}")
    return data

class Context:
    def __init__(self):
        self.data = {}

# Regular run patterns - should NOT match
def run_process():
    print("Running process")

process.run()
application.run()
script.run()

if __name__ == "__main__":
    run_process()

if __name__ == "__main__":
    app = create_app()
    app.run()

# Database patterns - should NOT match
from sqlalchemy import create_engine
from django.db import models

class DatabaseModel(models.Model):
    name = models.CharField(max_length=100)

# Machine learning patterns that are not MCP - should NOT match
from sklearn.ensemble import RandomForestClassifier
from tensorflow import keras
import torch

model = RandomForestClassifier()
neural_net = keras.Sequential()
tensor = torch.tensor([1, 2, 3])

# API client patterns - should NOT match
import requests
import aiohttp

response = requests.get("https://api.example.com/data")
async with aiohttp.ClientSession() as session:
    async with session.get("https://api.example.com") as resp:
        data = await resp.json()