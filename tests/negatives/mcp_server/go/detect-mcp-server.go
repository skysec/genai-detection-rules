// Negative test cases for Go MCP server detection - these should NOT trigger the rule
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"

	// Test case 1: Regular HTTP server imports
	"github.com/gin-gonic/gin"
	"github.com/gorilla/mux"
	"github.com/labstack/echo/v4"

	// Test case 2: Regular RPC imports
	"net/rpc"
	"net/rpc/jsonrpc"

	// Test case 3: gRPC imports
	"google.golang.org/grpc"

	// Test case 4: Other protocol imports
	"github.com/golang/protobuf/proto"
)

// Test case 5: Regular HTTP server setup (not MCP)
func createHTTPServer() {
	router := mux.NewRouter()
	router.HandleFunc("/api/tools", handleTools).Methods("GET")
	router.HandleFunc("/api/resources", handleResources).Methods("GET")

	server := &http.Server{
		Addr:    ":8080",
		Handler: router,
	}

	log.Fatal(server.ListenAndServe())
}

// Test case 6: Gin HTTP server (not MCP)
func createGinServer() {
	r := gin.Default()
	r.GET("/tools/list", func(c *gin.Context) {
		c.JSON(200, gin.H{"tools": []string{"tool1", "tool2"}})
	})
	r.POST("/tools/call", func(c *gin.Context) {
		c.JSON(200, gin.H{"result": "success"})
	})
	r.Run(":8080")
}

// Test case 7: Echo HTTP server (not MCP)
func createEchoServer() {
	e := echo.New()
	e.GET("/resources/list", listResources)
	e.GET("/resources/read", readResource)
	e.Start(":1323")
}

// Test case 8: Regular JSON-RPC server (not MCP-specific)
type RPCService struct{}

func (s *RPCService) Calculate(args *CalculateArgs, reply *int) error {
	*reply = args.A + args.B
	return nil
}

func createJSONRPCServer() {
	service := new(RPCService)
	rpc.Register(service)

	listener, err := net.Listen("tcp", ":1234")
	if err != nil {
		log.Fatal(err)
	}

	for {
		conn, err := listener.Accept()
		if err != nil {
			continue
		}
		go jsonrpc.ServeConn(conn)
	}
}

// Test case 9: Regular gRPC server
func createGRPCServer() {
	server := grpc.NewServer()
	// Register services here
	log.Fatal(server.Serve(nil))
}

// Test case 10: Regular struct types (not MCP-specific)
type Tool struct {
	Name        string `json:"name"`
	Description string `json:"description"`
	Handler     func() string
}

type Server struct {
	Tools     []Tool
	Resources []Resource
}

type Resource struct {
	URI      string `json:"uri"`
	MimeType string `json:"mimeType"`
	Content  string `json:"content"`
}

// Test case 11: Regular function signatures (not MCP-specific)
func handleToolRequest(ctx context.Context, req *ToolRequest) (*ToolResult, error) {
	return &ToolResult{
		Status: "success",
		Data:   req.Input + " processed",
	}, nil
}

func handleResourceRequest(ctx context.Context, req ResourceRequest) ([]ResourceContent, error) {
	return []ResourceContent{
		{
			Path:    req.Path,
			Content: "file content",
		},
	}, nil
}

// Test case 12: Regular method switch cases (not MCP protocol methods)
func handleAPIRequests(request APIRequest) {
	switch request.Method {
	case "GET":
		handleGetRequest()
	case "POST":
		handlePostRequest()
	case "PUT":
		handlePutRequest()
	case "DELETE":
		handleDeleteRequest()
	case "api/status":
		handleStatusRequest()
	case "api/health":
		handleHealthRequest()
	}
}

// Test case 13: Custom server implementation (not MCP)
type CustomServer struct {
	name    string
	version string
	tools   map[string]func() string
}

func NewCustomServer(name, version string) *CustomServer {
	return &CustomServer{
		name:    name,
		version: version,
		tools:   make(map[string]func() string),
	}
}

func (s *CustomServer) AddTool(name string, handler func() string) {
	s.tools[name] = handler
}

func (s *CustomServer) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	// Regular HTTP handling
}

// Test case 14: Database server or cache server patterns
type DatabaseServer struct {
	connection string
}

func NewDatabaseServer(conn string) *DatabaseServer {
	return &DatabaseServer{connection: conn}
}

func (db *DatabaseServer) Run(ctx context.Context) error {
	// Database server logic
	return nil
}

// Test case 15: Message queue server patterns
type MessageServer struct {
	queue chan Message
}

type Message struct {
	ID      string      `json:"id"`
	Method  string      `json:"method"`
	Payload interface{} `json:"payload"`
}

func NewMessageServer() *MessageServer {
	return &MessageServer{
		queue: make(chan Message, 100),
	}
}

func (m *MessageServer) ServeMessages() {
	for msg := range m.queue {
		m.handleMessage(msg)
	}
}

func (m *MessageServer) handleMessage(msg Message) {
	switch msg.Method {
	case "process":
		// Handle process message
	case "notify":
		// Handle notification
	default:
		// Handle unknown message
	}
}

// Test case 16: Regular client implementations (not MCP client)
type HTTPClient struct {
	baseURL string
	client  *http.Client
}

func NewHTTPClient(url string) *HTTPClient {
	return &HTTPClient{
		baseURL: url,
		client:  &http.Client{},
	}
}

// Test case 17: Configuration and utility structs
type Config struct {
	ServerName string `json:"server_name"`
	Version    string `json:"version"`
	Tools      []ToolConfig
}

type ToolConfig struct {
	Name        string `json:"name"`
	Description string `json:"description"`
	Enabled     bool   `json:"enabled"`
}

// Test case 18: Regular context-based handlers (not MCP-specific)
func processRequest(ctx context.Context, req GenericRequest) (GenericResult, error) {
	return GenericResult{
		Message: "Request processed",
		Status:  200,
	}, nil
}

func handleAsync(ctx context.Context, input string) (string, error) {
	// Async processing
	return "processed: " + input, nil
}

// Supporting types for negative tests
type CalculateArgs struct {
	A int `json:"a"`
	B int `json:"b"`
}

type ToolRequest struct {
	Name  string `json:"name"`
	Input string `json:"input"`
}

type ToolResult struct {
	Status string `json:"status"`
	Data   string `json:"data"`
}

type ResourceRequest struct {
	Path string `json:"path"`
}

type ResourceContent struct {
	Path    string `json:"path"`
	Content string `json:"content"`
}

type APIRequest struct {
	Method string      `json:"method"`
	Params interface{} `json:"params"`
}

type GenericRequest struct {
	ID     string      `json:"id"`
	Action string      `json:"action"`
	Data   interface{} `json:"data"`
}

type GenericResult struct {
	Message string `json:"message"`
	Status  int    `json:"status"`
}

// Handler functions for HTTP endpoints
func handleTools(w http.ResponseWriter, r *http.Request) {
	tools := []string{"calculator", "text-processor"}
	json.NewEncoder(w).Encode(tools)
}

func handleResources(w http.ResponseWriter, r *http.Request) {
	resources := []string{"file1.txt", "file2.txt"}
	json.NewEncoder(w).Encode(resources)
}

func listResources(c echo.Context) error {
	return c.JSON(200, []string{"resource1", "resource2"})
}

func readResource(c echo.Context) error {
	return c.String(200, "Resource content")
}

// Regular request handlers (not MCP)
func handleGetRequest()    {}
func handlePostRequest()   {}
func handlePutRequest()    {}
func handleDeleteRequest() {}
func handleStatusRequest() {}
func handleHealthRequest() {}

func main() {
	fmt.Println("Regular Go server - not MCP")
	createHTTPServer()
}