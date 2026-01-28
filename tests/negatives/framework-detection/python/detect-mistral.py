"""
Negative test cases for Mistral AI detection rule
These should NOT be detected by the detect-mistral rule
"""

# Test 1: Comments mentioning mistral
# This code uses mistral API but doesn't import it

# Test 2: Strings containing mistral
api_name = "mistral"
description = "This uses the Mistral AI API"
model_name = "mistral-large-latest"

# Test 3: Dictionary with mistral keys
config = {
    "provider": "mistral",
    "model": "mistral-medium",
    "api_key": "your_api_key"
}

# Test 4: Custom classes with similar names
class Mistral:
    """Custom class - not the actual Mistral client"""
    def __init__(self, api_key):
        self.api_key = api_key

    def chat(self, model, messages):
        return "mock response"

class MistralClient:
    """Custom client class"""
    def __init__(self, api_key):
        self.api_key = api_key

# Test 5: Functions with similar names
def chat(model, messages):
    """Custom chat function - not using Mistral"""
    return {"response": "mock"}

# Test 6: HTTP client calls (not using SDK)
import requests

response = requests.post(
    "https://api.mistral.ai/v1/chat/completions",
    headers={"Authorization": "Bearer your_api_key"},
    json={
        "model": "mistral-large-latest",
        "messages": [{"role": "user", "content": "Hello"}]
    }
)

# Test 7: Import unrelated packages
import mistral  # hypothetical package - not mistralai
import ai_client  # hypothetical package

# Test 8: Custom implementation with similar API
class MyMistralClient:
    def __init__(self):
        self.chat = self.Chat()
        self.models = self.Models()

    class Chat:
        def complete(self, model, messages):
            return "mock"

        def stream(self, model, messages):
            return ["chunk1", "chunk2"]

    class Models:
        def list(self):
            return ["model1", "model2"]

client = MyMistralClient()
response = client.chat.complete("mistral-medium", [])
models = client.models.list()

# Test 9: Variable names containing mistral
mistral_enabled = True
use_mistral_api = False
mistral_model_name = "mistral-large"

# Test 10: URLs or documentation references
api_endpoint = "https://api.mistral.ai"
docs_url = "https://docs.mistral.ai"

# Test 11: JSON or data structures
settings = {
    "mistral_settings": {
        "api_key": "placeholder",
        "model": "mistral-medium"
    }
}

# Test 12: Environment variables
import os
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL")

# Test 13: Log messages
def log_api_usage():
    """
    Logs API usage for mistral services
    This function tracks Mistral AI API calls
    """
    print("Mistral API called")
    log_message = "Using Mistral Large"
    return log_message

# Test 14: Test data or fixtures
TEST_MISTRAL_RESPONSE = {
    "choices": [{"message": {"content": "Hello"}}]
}

MOCK_MISTRAL_STREAM = ["chunk1", "chunk2", "chunk3"]

# Test 15: Configuration parsing
def parse_mistral_config():
    config_text = """
    [mistral]
    api_key = your_api_key
    model = mistral-large-latest
    """
    return config_text

# Test 16: Documentation strings
"""
This module integrates with Mistral AI
to provide LLM capabilities.
"""

class AIProvider:
    """
    AI provider using Mistral patterns.
    Note: This is a custom implementation, not using Mistral directly.
    """
    pass

# Test 17: Comments about implementation
# TODO: Add support for Mistral AI
# FIXME: Mistral integration not working
# NOTE: Consider using Mistral for this task

# Test 18: Mock classes for testing
class MockMistral:
    """Mock for testing - not real Mistral"""
    def __init__(self, api_key):
        self.api_key = api_key

    def chat(self, model, messages):
        return type('Response', (), {
            'choices': [type('Choice', (), {
                'message': type('Message', (), {'content': 'mock'})()
            })()]
        })()

mock = MockMistral(api_key="test")
response = mock.chat("mistral-medium", [])

# Test 19: Custom embeddings class
class Embeddings:
    """Custom embeddings class - not Mistral"""
    def create(self, model, input):
        return {"embeddings": [[0.1, 0.2, 0.3]]}

embeddings = Embeddings()
result = embeddings.create("mistral-embed", ["text"])

# Test 20: Custom file operations
class Files:
    """Custom files class - not Mistral"""
    def upload(self, file, purpose):
        return {"id": "file-123"}

    def list(self):
        return [{"id": "file-1"}, {"id": "file-2"}]

    def delete(self, file_id):
        print(f"Deleting {file_id}")

files = Files()
files.upload("data.jsonl", "fine-tune")
files.delete("file-123")
