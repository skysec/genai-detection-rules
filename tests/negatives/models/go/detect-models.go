/**
 * Negative test cases for AI models detection rule
 * These should NOT be detected by the detect-ai-models rule
 */

package main

import (
	"fmt"
	"log"
	"os"
)

func main() {
	// Test 1: Comments mentioning model names
	// This code uses gpt-4 for processing
	// We prefer claude-3 over gemini-pro
	/* Multi-line comment about models:
	   - gpt-4 is good
	   - claude-3 is better
	*/

	// Test 2: String variables not assigned to model parameter
	modelDescription := "We use gpt-4 in production"
	documentation := "Claude-3-opus is the best model"
	notes := "Consider gemini-pro for embeddings"

	// Test 3: Struct fields that don't match model patterns
	type Config struct {
		ModelName    string
		ModelType    string
		ModelVersion string
		Engine       string
	}
	config := Config{
		ModelName:    "custom-model",
		ModelType:    "transformer",
		ModelVersion: "v1.0",
		Engine:       "neural-network",
	}

	// Test 4: Function names with model-like strings
	func trainGptModel() {
		// Train a custom GPT-like model
	}

	type ClaudeWrapper struct {
		// Wrapper for claude-like functionality
	}

	func processWithGeminiLogic() {
		// Process data using gemini-inspired logic
	}

	// Test 5: Non-AI model references
	databaseModel := "postgresql"
	mlModel := "random-forest"
	predictionModel := "linear-regression"

	// Test 6: URL and endpoint references
	apiURL := "https://api.openai.com/v1/chat/completions"
	endpoint := "/models/gpt-4/info"
	docsURL := "https://platform.openai.com/docs/models/gpt-4"

	// Test 7: Custom model names that don't match patterns
	type CustomRequest struct {
		Model string
	}
	req1 := CustomRequest{Model: "my-custom-model"}
	req2 := CustomRequest{Model: "company-llm-v2"}
	req3 := CustomRequest{Model: "internal-ai-engine"}

	// Test 8: Variable assignments without AI model patterns
	model := "custom-transformer"
	aiModel := "proprietary-llm"
	llmName := "in-house-model"

	// Test 9: Model names as part of larger strings (not exact matches)
	description := "The gpt-4-like model performs well"
	note := "Based on claude-3 architecture"
	comment := "Similar to gemini-pro functionality"

	// Test 10: Struct with model field but non-AI model value
	type Settings struct {
		Model   string
		Version string
	}
	settings := Settings{Model: "custom-engine", Version: "1.0"}

	type Params struct {
		Model       string
		Temperature float64
	}
	params := Params{Model: "proprietary", Temperature: 0.7}

	// Test 11: Model in log messages
	log.Println("Using model: custom-model-v1")
	fmt.Printf("Model: %s\n", "custom-model")

	// Test 12: Test data and fixtures
	const TEST_MODEL = "test-model-fixture"
	const MOCK_MODEL = "mock-ai-model"
	const SAMPLE_MODEL = "sample-model-data"

	// Test 13: Configuration file content (as strings)
	configYAML := `
model:
  name: custom-model
  version: 1.0
`

	configJSON := `{"model": "custom-engine", "api": "v1"}`

	// Test 14: Environment variables
	MODEL_NAME := os.Getenv("MODEL_NAME")
	if MODEL_NAME == "" {
		MODEL_NAME = "default-model"
	}
	AI_MODEL := os.Getenv("AI_MODEL")

	// Test 15: Type definitions
	type AIConfig struct {
		ModelType         string
		ModelArchitecture string
		ModelFramework    string
	}

	type ModelConfig struct {
		Name    string
		Version string
	}

	// Test 16: Function parameters named model (but with custom values)
	func initializeModel(model string) {
		if model == "" {
			model = "custom-model"
		}
		// Initialize with custom model
	}

	// Test 17: Model comparison or conditional logic
	modelFamily := []string{"gpt", "claude", "gemini"} // Generic names, not specific models
	for _, fam := range modelFamily {
		if fam == model {
			// process
		}
	}

	// Test 18: Model name fragments (incomplete)
	modelPrefix := "gpt"
	modelFamilyName := "claude"
	modelSeries := "gemini"

	// Test 19: Open source model names that don't match our patterns
	oss1 := CustomRequest{Model: "falcon-40b"}
	oss2 := CustomRequest{Model: "vicuna-13b"}
	oss3 := CustomRequest{Model: "alpaca-7b"}
	oss4 := CustomRequest{Model: "bloom-176b"}

	// Test 20: Version numbers alone
	version := "3.5"
	modelVer := "4.0"
	apiVersion := "2.1"

	// Test 21: Constants with generic names
	const (
		DefaultModel = "default"
		CustomModel  = "custom"
		TestModel    = "test"
	)

	// Test 22: Map with model keys
	models := map[string]string{
		"model-a": "config-a",
		"model-b": "config-b",
		"model-c": "config-c",
	}

	// Test 23: Slice of generic model names
	supportedModels := []string{"custom", "internal", "proprietary"}

	// Test 24: Interface definitions
	type ModelManager interface {
		GetModel() string
		SetModel(string)
	}

	// Test 25: Method receivers
	type Service struct {
		modelType string
	}

	func (s *Service) GetModelType() string {
		return "custom-model"
	}

	// Use variables to avoid unused warnings
	_, _, _, _, _, _, _, _, _, _ = modelDescription, documentation, notes, config, databaseModel, mlModel, predictionModel, apiURL, endpoint, docsURL
	_, _, _, _, _, _, _, _, _, _ = req1, req2, req3, model, aiModel, llmName, description, note, comment, settings
	_, _, _, _, _, _, _, _ = params, configYAML, configJSON, MODEL_NAME, AI_MODEL, modelPrefix, modelFamilyName, modelSeries
	_, _, _, _, _, _, _, _, _ = oss1, oss2, oss3, oss4, version, modelVer, apiVersion, models, supportedModels
	_, _ = trainGptModel, processWithGeminiLogic
	var _ ModelManager
	var _ Service
	_ = initializeModel
}
