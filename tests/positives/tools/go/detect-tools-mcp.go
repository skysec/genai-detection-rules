/**
 * Positive test cases for MCP tool detection rule
 * These should all be detected by the detect-tools-mcp rule
 */

package main

import (
	// ruleid: detect-tools-mcp
	"github.com/modelcontextprotocol/go-sdk/protocol"
	// ruleid: detect-tools-mcp
	"github.com/modelcontextprotocol/go-sdk/server"
)

// Test 1: Tool handler function
// ruleid: detect-tools-mcp
func HandleCalculate(srv *server.MCPServer, req *protocol.CallToolRequest) (*protocol.CallToolResult, error) {
	// Tool implementation
	return &protocol.CallToolResult{
		Content: []protocol.Content{
			{Type: "text", Text: "result"},
		},
	}, nil
}

// Test 2: Tool list handler
// ruleid: detect-tools-mcp
func ListAvailableTools(srv *server.MCPServer, req *protocol.ListToolsRequest) (*protocol.ListToolsResult, error) {
	tools := []*protocol.Tool{
		{
			Name:        "calculate",
			Description: "Calculate sum of two numbers",
			InputSchema: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"a": map[string]string{"type": "number"},
					"b": map[string]string{"type": "number"},
				},
			},
		},
	}
	return &protocol.ListToolsResult{Tools: tools}, nil
}

// Test 3: Method on custom type for tool handling
type ToolHandler struct {
	tools map[string]func(map[string]interface{}) (interface{}, error)
}

// ruleid: detect-tools-mcp
func (h *ToolHandler) HandleToolCall(req *protocol.CallToolRequest) (*protocol.CallToolResult, error) {
	// Implementation
	return &protocol.CallToolResult{}, nil
}

// Test 4: Tool struct definition
// ruleid: detect-tools-mcp
type CustomTool struct {
	Name        string
	Description string
	InputSchema map[string]interface{}
	Handler     func(map[string]interface{}) (interface{}, error)
}

// Test 5: Tool registration pattern
func SetupTools(srv *server.MCPServer) {
	// ruleid: detect-tools-mcp
	srv.RegisterTool("search", handleSearch)
	// ruleid: detect-tools-mcp
	srv.RegisterTool("fetch", handleFetch)
}

func handleSearch(params map[string]interface{}) (interface{}, error) {
	return nil, nil
}

func handleFetch(params map[string]interface{}) (interface{}, error) {
	return nil, nil
}

// Test 6: Tool with protocol.Tool struct
func CreateTool() *protocol.Tool {
	// ruleid: detect-tools-mcp
	return &protocol.Tool{
		Name:        "query_database",
		Description: "Execute a database query",
		InputSchema: map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"query": map[string]string{
					"type":        "string",
					"description": "SQL query to execute",
				},
			},
		},
	}
}

// Test 7: Tool switch statement
// ruleid: detect-tools-mcp
func ExecuteTool(req *protocol.CallToolRequest) (*protocol.CallToolResult, error) {
	switch req.Params.Name {
	case "read_file":
		return &protocol.CallToolResult{}, nil
	case "write_file":
		return &protocol.CallToolResult{}, nil
	default:
		return nil, fmt.Errorf("unknown tool: %s", req.Params.Name)
	}
}

// Test 8: Tool handler with switch
type MCPToolServer struct {
	server *server.MCPServer
}

// ruleid: detect-tools-mcp
func (s *MCPToolServer) HandleToolCall(req *protocol.CallToolRequest) (*protocol.CallToolResult, error) {
	// ruleid: detect-tools-mcp
	switch req.Params.Name {
	case "get_weather":
		return s.getWeather(req)
	case "get_time":
		return s.getTime(req)
	default:
		return nil, fmt.Errorf("tool not found")
	}
}

func (s *MCPToolServer) getWeather(req *protocol.CallToolRequest) (*protocol.CallToolResult, error) {
	return &protocol.CallToolResult{}, nil
}

func (s *MCPToolServer) getTime(req *protocol.CallToolRequest) (*protocol.CallToolResult, error) {
	return &protocol.CallToolResult{}, nil
}

// Test 9: Tool schema definition
// ruleid: detect-tools-mcp
var searchToolSchema = map[string]interface{}{
	"type": "object",
	"properties": map[string]interface{}{
		"query": map[string]interface{}{
			"type":        "string",
			"description": "Search query",
		},
		"limit": map[string]interface{}{
			"type":        "number",
			"description": "Maximum results",
		},
	},
	"required": []string{"query"},
}

// Test 10: Multiple tool definitions
func InitializeTools() []*protocol.Tool {
	return []*protocol.Tool{
		// ruleid: detect-tools-mcp
		{
			Name:        "tool1",
			Description: "First tool",
			InputSchema: map[string]interface{}{"type": "object"},
		},
		// ruleid: detect-tools-mcp
		{
			Name:        "tool2",
			Description: "Second tool",
			InputSchema: map[string]interface{}{"type": "object"},
		},
	}
}

// Test 11: Tool handler method
// ruleid: detect-tools-mcp
func (s *MCPToolServer) ListTools() ([]*protocol.Tool, error) {
	tools := []*protocol.Tool{
		{
			Name:        "calculate",
			Description: "Perform calculations",
		},
		{
			Name:        "search",
			Description: "Search data",
		},
	}
	return tools, nil
}

// Test 12: Tool registration helper
func RegisterCustomTools(mcp *server.MCPServer) error {
	// ruleid: detect-tools-mcp
	mcp.AddTool(&protocol.Tool{
		Name:        "custom_tool",
		Description: "Custom tool implementation",
		InputSchema: map[string]interface{}{
			"type": "object",
		},
	})
	return nil
}

// Test 13: Tool handler with error handling
// ruleid: detect-tools-mcp
func SafeToolHandler(srv *server.MCPServer, req *protocol.CallToolRequest) (*protocol.CallToolResult, error) {
	defer func() {
		if r := recover(); r != nil {
			// Handle panic
		}
	}()

	return &protocol.CallToolResult{}, nil
}

func main() {
	// Initialize server and tools
	_ = SetupTools
	_ = CreateTool
	_ = ExecuteTool
	_ = InitializeTools
	_ = RegisterCustomTools
}
