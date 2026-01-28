package main

// Negative test cases for Gemini (Go)

const provider = "gemini"
var modelName = "gemini-pro"

type Client struct{}

func NewClient() *Client {
	return &Client{}
}
