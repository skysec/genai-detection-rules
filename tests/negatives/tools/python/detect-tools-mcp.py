"""
Negative test cases for MCP tool detection rule
These should NOT be detected by the detect-tools-mcp rule
"""

# Test 1: Regular function without decorator
def regular_function(a: int, b: int) -> int:
    """Just a regular function"""
    return a + b

# Test 2: Async function without MCP decorator
async def async_function(url: str) -> dict:
    """Regular async function"""
    return {"data": "example"}

# Test 3: Function with different decorator
from functools import lru_cache

@lru_cache(maxsize=128)
def cached_function(n: int) -> int:
    """Function with cache decorator"""
    return n * 2

# Test 4: Class method (not a tool)
class MyClass:
    def method(self, x: int) -> int:
        """Class method"""
        return x + 1

    @staticmethod
    def static_method(y: int) -> int:
        """Static method"""
        return y * 2

# Test 5: Function named 'tool' but not MCP
def tool(name: str) -> dict:
    """Function named tool"""
    return {"name": name}

# Test 6: Custom decorator that looks similar
def my_tool(func):
    """Custom tool decorator"""
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@my_tool
def custom_decorated(x: int) -> int:
    """Function with custom decorator"""
    return x

# Test 7: Comments mentioning tools
# This code uses MCP tools for processing
# The @mcp.tool decorator is useful

# Test 8: String containing tool patterns
description = "Use @mcp.tool decorator to define tools"
documentation = """
To create a tool:
@server.tool
def my_tool():
    pass
"""

# Test 9: Mock or test tool class
class MockTool:
    """Mock tool for testing"""
    def __init__(self, name: str):
        self.name = name

    def execute(self):
        pass

# Test 10: Tool configuration (not tool definition)
tool_config = {
    "name": "my_tool",
    "description": "Tool description",
    "parameters": {}
}

# Test 11: Import statements only (no usage)
# from mcp import tool  # commented out

# Test 12: Tool usage/calling (not definition)
def use_tool(tool_name: str, args: dict):
    """Function that uses tools"""
    result = execute_tool(tool_name, args)
    return result

def execute_tool(name, parameters):
    """Execute a tool by name"""
    pass

# Test 13: Dataclass with tool field
from dataclasses import dataclass

@dataclass
class ToolConfig:
    name: str
    description: str
    handler: callable

# Test 14: Type hints with Context (but not MCP tool)
def non_mcp_function(data: str, context: dict) -> str:
    """Function with context parameter but not MCP"""
    return data

# Test 15: Decorator on class (not function)
class ToolManager:
    """Manager for tools"""
    tools = []

# Test 16: Lambda functions
calculate = lambda x: x * 2
process = lambda data: {"result": data}

# Test 17: Generator function
def generate_data():
    """Generator function"""
    for i in range(10):
        yield i

# Test 18: Context manager
class ToolContext:
    """Context manager for tools"""
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

# Test 19: Decorator factory
def create_tool_decorator(name: str):
    """Factory for creating decorators"""
    def decorator(func):
        func.tool_name = name
        return func
    return decorator

@create_tool_decorator("my_tool")
def decorated_with_factory(x: int) -> int:
    return x

# Test 20: Abstract base class
from abc import ABC, abstractmethod

class BaseTool(ABC):
    """Abstract base tool class"""
    @abstractmethod
    def execute(self, params: dict) -> any:
        pass
