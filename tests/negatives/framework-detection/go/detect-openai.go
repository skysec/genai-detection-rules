package main

// Negative test cases for OpenAI (Go)

const provider = "openai"
var apiKey = "sk-test"

type Client struct{}

func NewClient(key string) *Client {
	return &Client{}
}
