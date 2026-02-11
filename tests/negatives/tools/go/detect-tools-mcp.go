/**
 * Negative test cases for MCP tool detection rule
 * These should NOT be detected by the detect-tools-mcp rule
 */

package main

import (
	"fmt"
	"net/http"
)

// Test 1: Regular function
func regularFunction(a, b int) int {
	return a + b
}

// Test 2: HTTP handler (not MCP)
func httpHandler(w http.ResponseWriter, r *http.Request) {
	fmt.Fprintf(w, "Hello, World!")
}

// Test 3: Custom struct (not protocol.Tool)
type CustomTool struct {
	Name    string
	Handler func(map[string]interface{}) error
}

// Test 4: Interface definition
type ToolExecutor interface {
	Execute(params map[string]interface{}) (interface{}, error)
	GetName() string
}

// Test 5: Regular method
type Service struct {
	name string
}

func (s *Service) Process(data string) (string, error) {
	return data, nil
}

func (s *Service) Transform(input interface{}) (interface{}, error) {
	return input, nil
}

// Test 6: Comments mentioning tools
// This code uses MCP tools for processing
// The HandleToolCall pattern is useful
/* Multi-line comment:
   Tool handlers go here
   Use protocol.CallToolRequest for tool calls
*/

// Test 7: String containing tool patterns
const documentation = `
To define a tool:
func HandleTool(srv *server.MCPServer, req *protocol.CallToolRequest) (*protocol.CallToolResult, error) {
  // handler code
}
`

const example = "Use protocol.CallToolRequest for tool handling"

// Test 8: Mock structures
type MockServer struct {
	tools map[string]func(interface{}) error
}

func (m *MockServer) RegisterTool(name string, handler func(interface{}) error) {
	m.tools[name] = handler
}

func (m *MockServer) HandleRequest(req interface{}) error {
	return nil
}

// Test 9: Configuration structs
type ToolConfig struct {
	Name        string
	Description string
	Enabled     bool
}

type ServerConfig struct {
	Tools []ToolConfig
}

// Test 10: Regular switch statement (not tool-related)
func processCommand(command string) string {
	switch command {
	case "start":
		return "Starting..."
	case "stop":
		return "Stopping..."
	default:
		return "Unknown command"
	}
}

// Test 11: Function that uses tools (not defines them)
func invokeTool(toolName string, params map[string]interface{}) (interface{}, error) {
	// Code that invokes/calls a tool
	return "result", nil
}

// Test 12: Builder pattern
type ToolBuilder struct {
	name        string
	description string
}

func NewToolBuilder() *ToolBuilder {
	return &ToolBuilder{}
}

func (b *ToolBuilder) WithName(name string) *ToolBuilder {
	b.name = name
	return b
}

func (b *ToolBuilder) WithDescription(desc string) *ToolBuilder {
	b.description = desc
	return b
}

func (b *ToolBuilder) Build() interface{} {
	return map[string]string{
		"name":        b.name,
		"description": b.description,
	}
}

// Test 13: Generic handler function
func createHandler(processor func(interface{}) error) func(interface{}) error {
	return func(input interface{}) error {
		return processor(input)
	}
}

// Test 14: Event handlers
type EventEmitter struct {
	handlers map[string][]func(interface{})
}

func (e *EventEmitter) On(event string, handler func(interface{})) {
	e.handlers[event] = append(e.handlers[event], handler)
}

func (e *EventEmitter) Emit(event string, data interface{}) {
	for _, handler := range e.handlers[event] {
		handler(data)
	}
}

// Test 15: Factory pattern
func createToolExecutor(toolType string) ToolExecutor {
	// Factory implementation
	return nil
}

// Test 16: Method with similar signature but not MCP
type APIServer struct {
	handlers map[string]func(map[string]interface{}) (interface{}, error)
}

func (s *APIServer) HandleAPICall(endpoint string, params map[string]interface{}) (interface{}, error) {
	if handler, exists := s.handlers[endpoint]; exists {
		return handler(params)
	}
	return nil, fmt.Errorf("endpoint not found")
}

// Test 17: Type aliases
type ToolName string
type ToolParams map[string]interface{}
type ToolResult interface{}

// Test 18: Constants
const (
	ToolTypeQuery     = "query"
	ToolTypeTransform = "transform"
	ToolTypeFetch     = "fetch"
)

// Test 19: Slice operations
var toolNames = []string{"tool1", "tool2", "tool3"}

func getTools() []string {
	return toolNames
}

// Test 20: Context pattern (not MCP Context)
type Context struct {
	values map[string]interface{}
}

func (c *Context) Set(key string, value interface{}) {
	c.values[key] = value
}

func (c *Context) Get(key string) interface{} {
	return c.values[key]
}

func processWithContext(ctx *Context, data string) (string, error) {
	return data, nil
}

// Test 21: Generic function
func process[T any](input T) (T, error) {
	return input, nil
}

// Test 22: Closure
func createProcessor() func(string) string {
	counter := 0
	return func(data string) string {
		counter++
		return fmt.Sprintf("%s-%d", data, counter)
	}
}

func main() {
	// Use some functions to avoid unused warnings
	_ = regularFunction(1, 2)
	_ = httpHandler
	_ = processCommand("start")
	_ = invokeTool("test", nil)
	_ = createHandler(nil)
	_ = createToolExecutor("type")
	_ = getTools()
	_ = processWithContext(nil, "data")
	_ = process[string]("data")
	_ = createProcessor()
}
