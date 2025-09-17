# POSITIVE TEST CASES - Should be detected by the MCP client rule
# These are based on the official MCP Python SDK patterns

import asyncio
from pydantic import AnyUrl

# OFFICIAL MCP SDK IMPORTS - should match
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.client.auth import OAuthClientProvider
from mcp.types import PromptReference, ResourceTemplateReference
from mcp import types
from mcp.shared.auth import OAuthToken, OAuthClientInformationFull

# MCP-SPECIFIC PARAMETER CLASSES - should match
server_params = StdioServerParameters(
    command="uv",
    args=["run", "server", "fastmcp_quickstart", "stdio"],
    env={"UV_INDEX": ""}
)

# Alternative parameter initialization
stdio_params = StdioServerParameters(
    command="python",
    args=["-m", "my_mcp_server"]
)

# OFFICIAL STDIO CLIENT USAGE - should match
async def stdio_client_example():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            return tools

async def stdio_client_variant():
    # Direct usage without assignment
    async with stdio_client(StdioServerParameters(command="mcp-server", args=[])) as (read, write):
        session = ClientSession(read, write)
        await session.initialize()

# OFFICIAL STREAMABLE HTTP CLIENT USAGE - should match
async def streamable_http_example():
    async with streamablehttp_client("http://localhost:8000/mcp") as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            tools = await session.list_tools()

async def streamable_http_variant():
    client_connection = streamablehttp_client("http://mcp-server:3000/mcp")

# CLIENT SESSION PATTERNS - should match
async def client_session_examples():
    async with stdio_client(server_params) as (read, write):
        # Basic ClientSession usage
        session = ClientSession(read, write)
        await session.initialize()

        # Session with context manager
        async with ClientSession(read, write) as session:
            await session.initialize()

# MCP-SPECIFIC TOOL OPERATIONS - should match
async def tool_operations():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # List tools
            tools = await session.list_tools()
            available_tools = await session.list_tools()

            # Call tools
            result = await session.call_tool("weather", {"location": "Seattle"})
            weather_result = await session.call_tool("get_weather", {"city": "Tokyo"})

# MCP-SPECIFIC RESOURCE OPERATIONS - should match
async def resource_operations():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # List resources
            resources = await session.list_resources()
            available_resources = await session.list_resources()

            # Read resources
            content = await session.read_resource(AnyUrl("file:///greeting.txt"))
            document = await session.read_resource(AnyUrl("documents://readme.md"))

            # List resource templates
            templates = await session.list_resource_templates()

# MCP-SPECIFIC PROMPT OPERATIONS - should match
async def prompt_operations():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # List prompts
            prompts = await session.list_prompts()
            available_prompts = await session.list_prompts()

            # Get prompts
            prompt = await session.get_prompt("greet_user", arguments={"name": "Alice", "style": "friendly"})
            greeting = await session.get_prompt("generate_greeting", {"user": "Bob"})

# MCP COMPLETION API - should match (very specific to MCP)
async def completion_operations():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Resource template completion
            result = await session.complete(
                ref=ResourceTemplateReference(type="ref/resource", uri="github://owner/{repo}"),
                argument={"name": "repo", "value": ""}
            )

            # Prompt completion
            completion = await session.complete(
                ref=PromptReference(type="ref/prompt", name="greet_user"),
                argument={"name": "style", "value": ""}
            )

# MCP NOTIFICATION SENDING - should match
async def notification_operations():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Send resource notifications
            await session.send_resource_updated(AnyUrl("file:///config.json"))
            await session.send_resource_list_changed()

# MCP TYPES USAGE - should match
def mcp_types_usage():
    # Type imports and usage
    prompt_ref = PromptReference(type="ref/prompt", name="my_prompt")
    resource_ref = ResourceTemplateReference(type="ref/resource", uri="docs://{file}")
    file_url = AnyUrl("file:///documents/readme.md")

# OAUTH PATTERNS SPECIFIC TO MCP - should match
async def oauth_patterns():
    oauth_provider = OAuthClientProvider(
        server_url="http://localhost:8001",
        client_metadata=types.OAuthClientMetadata(
            client_name="Example MCP Client",
            redirect_uris=[AnyUrl("http://localhost:3000/callback")],
            grant_types=["authorization_code", "refresh_token"],
            response_types=["code"],
            scope="user"
        )
    )

# REAL-WORLD COMPREHENSIVE EXAMPLE - should match
async def comprehensive_mcp_client():
    """Complete MCP client implementation example"""
    server_params = StdioServerParameters(
        command="uv",
        args=["run", "my-mcp-server"],
        env={"MCP_ENV": "production"}
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize connection
            await session.initialize()

            # Discover capabilities
            tools = await session.list_tools()
            resources = await session.list_resources()
            prompts = await session.list_prompts()
            templates = await session.list_resource_templates()

            # Execute tool
            if tools.tools:
                result = await session.call_tool(
                    tools.tools[0].name,
                    {"input": "test data"}
                )

            # Read resource
            if resources.resources:
                content = await session.read_resource(resources.resources[0].uri)

            # Get prompt
            if prompts.prompts:
                prompt_result = await session.get_prompt(
                    prompts.prompts[0].name,
                    {"context": "example"}
                )

            # Completion example
            if templates.resourceTemplates:
                completion = await session.complete(
                    ref=ResourceTemplateReference(
                        type="ref/resource",
                        uri=templates.resourceTemplates[0].uriTemplate
                    ),
                    argument={"name": "param", "value": "partial"}
                )

# STREAMABLE HTTP WITH AUTH - should match
async def streamable_http_with_auth():
    oauth_auth = OAuthClientProvider(
        server_url="http://localhost:8001",
        client_metadata=types.OAuthClientMetadata(
            client_name="Auth MCP Client",
            redirect_uris=[AnyUrl("http://localhost:3000/callback")]
        )
    )

    async with streamablehttp_client("http://localhost:8001/mcp", auth=oauth_auth) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()

if __name__ == "__main__":
    # Run examples
    asyncio.run(stdio_client_example())
    asyncio.run(comprehensive_mcp_client())