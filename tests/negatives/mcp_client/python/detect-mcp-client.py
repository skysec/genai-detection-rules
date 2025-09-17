# NEGATIVE TEST CASES - Should NOT be detected by the MCP client rule
# These are generic patterns that might be confused with MCP but are not MCP-specific

import asyncio
import json
import os
from typing import Dict, List, Any
from urllib.parse import parse_qs, urlparse

# STANDARD HTTP CLIENT LIBRARIES - should NOT match
import requests
import httpx
import urllib.request
import aiohttp
from requests import Session

# STANDARD ASYNC LIBRARIES - should NOT match
import websockets
import asyncio
import uvloop

# DATABASE CLIENTS - should NOT match
import sqlite3
import psycopg2
from sqlalchemy import create_engine
from databases import Database
import pymongo
from redis import Redis

# GENERIC API CLIENTS - should NOT match
class APIClient:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session = requests.Session()

    def list_tools(self):
        """This is NOT MCP - just a generic API method"""
        return self.session.get(f"{self.base_url}/tools")

    def call_tool(self, name: str, args: Dict):
        """This is NOT MCP - just a generic API method"""
        return self.session.post(f"{self.base_url}/tools/{name}", json=args)

    def list_resources(self):
        """This is NOT MCP - just a generic API method"""
        return self.session.get(f"{self.base_url}/resources")

    def read_resource(self, uri: str):
        """This is NOT MCP - just a generic API method"""
        return self.session.get(f"{self.base_url}/resource?uri={uri}")

# GENERIC CLIENT SESSION PATTERNS - should NOT match
class ClientSession:
    """This is NOT MCP ClientSession - just a generic class with same name"""
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port

    async def connect(self):
        # Generic connection, not MCP
        pass

    async def initialize(self):
        # Generic initialization, not MCP
        pass

    async def list_tools(self):
        # Generic tool listing, not MCP
        return []

# GENERIC ASYNC CONTEXT MANAGERS - should NOT match
async def generic_client_example():
    async with httpx.AsyncClient() as client:
        response = await client.get("https://api.example.com/data")
        return response.json()

async def websocket_client_example():
    async with websockets.connect("ws://localhost:8765") as websocket:
        await websocket.send("hello")
        response = await websocket.recv()

async def database_client_example():
    database = Database("postgresql://user:pass@localhost/db")
    async with database.transaction():
        await database.execute("SELECT * FROM users")

# REGULAR IMPORTS THAT COULD BE CONFUSED - should NOT match
from client import Session  # Not MCP
from api.client import HTTPClient  # Not MCP
import client_lib  # Not MCP
from myapp.clients.stdio import stdio_connection  # Not MCP, similar name

# GENERIC PARAMETER CLASSES - should NOT match
class ServerParameters:
    """Generic server params, not MCP StdioServerParameters"""
    def __init__(self, command: str, args: List[str]):
        self.command = command
        self.args = args

class ConnectionParams:
    """Generic connection params"""
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port

# GENERIC SERVICE CLIENTS - should NOT match
class ServiceClient:
    def __init__(self, endpoint: str):
        self.endpoint = endpoint

    async def call_service(self, method: str, params: Dict):
        # Generic service call, not MCP
        async with httpx.AsyncClient() as client:
            return await client.post(f"{self.endpoint}/{method}", json=params)

# RPC CLIENTS - should NOT match
import xmlrpc.client
import jsonrpc_requests

def rpc_client_example():
    # XML-RPC client
    proxy = xmlrpc.client.ServerProxy("http://localhost:8000/")
    result = proxy.list_tools()  # NOT MCP

    # JSON-RPC client
    client = jsonrpc_requests.HTTPClient("http://localhost:8001/rpc")
    response = client.call("list_resources")  # NOT MCP

# GRPC CLIENTS - should NOT match
import grpc

async def grpc_client_example():
    async with grpc.aio.insecure_channel("localhost:50051") as channel:
        # Generic gRPC usage, not MCP
        pass

# GENERIC TOOLS AND RESOURCES - should NOT match
def process_tools(tool_list: List[str]):
    """Process a list of tools - NOT MCP related"""
    for tool in tool_list:
        print(f"Processing tool: {tool}")

def read_resource_file(filepath: str):
    """Read a resource file - NOT MCP related"""
    with open(filepath, 'r') as f:
        return f.read()

class ResourceManager:
    """Generic resource manager - NOT MCP"""
    def list_resources(self):
        return os.listdir("./resources")

    def read_resource(self, name: str):
        with open(f"./resources/{name}", 'r') as f:
            return f.read()

# GENERIC ASYNC PATTERNS - should NOT match
async def async_operations():
    # Generic async operations that might look similar
    session = aiohttp.ClientSession()

    # List tools from a generic API
    async with session.get("http://api.example.com/tools") as response:
        tools = await response.json()

    # Call a generic API endpoint
    async with session.post("http://api.example.com/execute", json={"tool": "calculator"}) as response:
        result = await response.json()

    # Read from a generic resource API
    async with session.get("http://api.example.com/resources/readme") as response:
        content = await response.text()

    await session.close()

# GENERIC COMPLETION APIs - should NOT match
from openai import OpenAI
import anthropic

def ai_completion_examples():
    # OpenAI completion - NOT MCP
    openai_client = OpenAI()
    response = openai_client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt="Complete this text"
    )

    # Anthropic completion - NOT MCP
    anthropic_client = anthropic.Anthropic()
    message = anthropic_client.messages.create(
        model="claude-3-sonnet-20240229",
        messages=[{"role": "user", "content": "Hello"}]
    )

# GENERIC OAUTH PATTERNS - should NOT match
from requests_oauthlib import OAuth2Session
import oauth2

def oauth_examples():
    # Generic OAuth, not MCP-specific
    oauth = OAuth2Session(client_id="abc123")
    authorization_url, state = oauth.authorization_url("https://provider.com/oauth/authorize")

# CONFIGURATION PATTERNS - should NOT match
config = {
    "server_url": "https://api.example.com",
    "timeout": 30,
    "max_retries": 3,
    "api_key": "secret"
}

# Generic client with similar method names
class GenericClient:
    def __init__(self, config: Dict):
        self.config = config

    def initialize(self):
        # Generic initialization, not MCP
        pass

    def list_tools(self):
        # Generic method, not MCP
        return ["hammer", "screwdriver", "wrench"]

    def call_tool(self, name: str, args: Dict):
        # Generic method, not MCP
        return f"Called {name} with {args}"

# FILE OPERATIONS - should NOT match
def file_operations():
    # Regular file operations that might look like resource operations
    with open("resource.txt", "r") as f:
        content = f.read()

    # List files (not MCP resources)
    import glob
    files = glob.glob("*.txt")

    # Read configuration
    with open("config.json", "r") as f:
        config = json.load(f)

# GENERIC SESSION MANAGEMENT - should NOT match
class Session:
    """Generic session class, not MCP"""
    def __init__(self, connection):
        self.connection = connection

    async def start(self):
        await self.connection.connect()

    async def close(self):
        await self.connection.disconnect()

# TESTING FRAMEWORKS - should NOT match
import pytest
import unittest

class TestClient:
    """Test client - NOT MCP"""
    def list_tools(self):
        return ["test_tool1", "test_tool2"]

# SUBPROCESS USAGE - should NOT match
import subprocess

def run_command():
    # Running subprocesses, not MCP stdio
    result = subprocess.run(["python", "script.py"], capture_output=True, text=True)
    return result.stdout

# REGULAR ASYNC CONTEXT MANAGERS WITH SIMILAR SIGNATURES - should NOT match
async def similar_patterns():
    # Database transaction
    async with Database("sqlite:///db.sqlite").transaction() as transaction:
        await transaction.execute("SELECT 1")

    # HTTP session
    async with httpx.AsyncClient() as (client):
        response = await client.get("http://example.com")

# LOGGING AND MONITORING - should NOT match
import logging

logger = logging.getLogger(__name__)

def log_operations():
    logger.info("Starting operation")
    logger.debug("Debug information")
    logger.warning("Warning message")
    logger.error("Error occurred")

# GENERIC NOTIFICATION SYSTEMS - should NOT match
class NotificationManager:
    async def send_notification(self, message: str):
        # Generic notification, not MCP
        print(f"Notification: {message}")

    async def broadcast_update(self, resource_id: str):
        # Generic broadcast, not MCP
        print(f"Resource {resource_id} updated")

if __name__ == "__main__":
    # Run generic examples that should NOT trigger MCP detection
    client = APIClient("https://api.example.com")
    tools = client.list_tools()

    generic_client = GenericClient(config)
    generic_client.initialize()

    asyncio.run(generic_client_example())
    asyncio.run(async_operations())