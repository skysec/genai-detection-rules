"""
Positive test cases for MCP tool detection rule
These should all be detected by the detect-tools-mcp rule
"""

from mcp.server.fastmcp import FastMCP, Context
from mcp import tool

# Test 1: Basic tool decorator
mcp = FastMCP("test-server")

# ruleid: detect-tools-mcp
@mcp.tool
def calculate_sum(a: int, b: int) -> int:
    """Calculate the sum of two numbers"""
    return a + b

# Test 2: Async tool with decorator
# ruleid: detect-tools-mcp
@mcp.tool
async def fetch_data(url: str) -> dict:
    """Fetch data from a URL"""
    # Implementation
    return {"data": "example"}

# Test 3: Tool with parameters
# ruleid: detect-tools-mcp
@mcp.tool(name="custom_search", description="Search for information")
def search_tool(query: str) -> list:
    """Search tool with custom parameters"""
    return []

# Test 4: Tool with Context parameter
# ruleid: detect-tools-mcp
@mcp.tool
def context_aware_tool(query: str, context: Context) -> str:
    """Tool that uses MCP context"""
    return f"Processing: {query}"

# Test 5: Async tool with Context
# ruleid: detect-tools-mcp
@mcp.tool
async def async_context_tool(data: dict, context: Context) -> dict:
    """Async tool with context"""
    return data

# Test 6: Tool with detailed schema
# ruleid: detect-tools-mcp
@mcp.tool(
    name="process_document",
    description="Process a document and extract information"
)
def process_document(file_path: str, extract_images: bool = False) -> dict:
    """
    Process a document file.

    Args:
        file_path: Path to the document
        extract_images: Whether to extract images

    Returns:
        Processed document data
    """
    return {"processed": True}

# Test 7: Database query tool
# ruleid: detect-tools-mcp
@mcp.tool
async def query_database(sql: str, ctx: Context) -> list:
    """Execute a database query"""
    return []

# Test 8: File operations tool
# ruleid: detect-tools-mcp
@mcp.tool
def read_file(path: str) -> str:
    """Read contents of a file"""
    return ""

# Test 9: API call tool
# ruleid: detect-tools-mcp
@mcp.tool
async def call_api(endpoint: str, method: str = "GET") -> dict:
    """Make an API call"""
    return {}

# Test 10: Data transformation tool
# ruleid: detect-tools-mcp
@mcp.tool(description="Transform data format")
def transform_data(data: dict, format: str) -> str:
    """Transform data to specified format"""
    return ""

# Test 11: Multiple tools in one file
server = FastMCP("multi-tool-server")

# ruleid: detect-tools-mcp
@server.tool
def tool_one(param: str) -> str:
    """First tool"""
    return param

# ruleid: detect-tools-mcp
@server.tool
async def tool_two(param: int) -> int:
    """Second tool"""
    return param * 2

# Test 12: Tool with complex return type
# ruleid: detect-tools-mcp
@mcp.tool
def get_weather(location: str) -> dict[str, any]:
    """Get weather for a location"""
    return {"temp": 72, "condition": "sunny"}

# Test 13: Tool decorator without parentheses
# ruleid: detect-tools-mcp
@server.tool
def simple_tool(x: int):
    """Simple tool implementation"""
    pass

# Test 14: Tool with optional Context
# ruleid: detect-tools-mcp
@mcp.tool
async def optional_context_tool(query: str, context: Context | None = None) -> str:
    """Tool with optional context parameter"""
    return query

# Test 15: Tool from imported decorator
# ruleid: detect-tools-mcp
@tool
def standalone_tool(input_data: str) -> str:
    """Tool using imported decorator"""
    return input_data
