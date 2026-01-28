/**
 * Negative test cases for OpenAI detection rule (TypeScript)
 * These should NOT be detected by the detect-openai rule
 */

// Test 1: Comment mentioning openai
// This code mentions openai but doesn't use it

// Test 2: String containing openai
const apiName = "openai";
const description = "Using OpenAI API";

// Test 3: Dictionary with openai
const config = {
  provider: "openai",
  model: "gpt-4"
};

// Test 4: Custom class
class OpenAI {
  constructor(config: any) {}
}

// Test 5: HTTP calls (not using SDK)
const response = await fetch("https://api.openai.com/v1/chat/completions", {
  method: "POST",
  headers: { "Authorization": "Bearer sk-test" }
});

// Test 6: Variable names
const openaiEnabled = true;
const useOpenAI = false;
