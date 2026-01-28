"""
Negative test cases for OpenAI detection rule
These should NOT be detected by the detect-openai rule
"""

# Test 1: Comment with "openai" should not trigger
# This code mentions openai in a comment but doesn't use it

# Test 2: String containing "openai" should not trigger
api_name = "openai"
description = "This uses the openai API"

# Test 3: Dictionary key with openai
config = {
    "provider": "openai",
    "model": "gpt-4"
}

# Test 4: Custom class or variable named similar to OpenAI
class OpenAIWrapper:
    """Custom wrapper class - not the actual OpenAI client"""
    pass

# Test 5: Function named similar to OpenAI methods
def create_chat_completion(messages):
    """Custom function - not using OpenAI"""
    return {"response": "mock response"}

# Test 6: HTTP client calls to OpenAI (not using the SDK)
import requests

response = requests.post(
    "https://api.openai.com/v1/chat/completions",
    headers={"Authorization": "Bearer sk-test"},
    json={"model": "gpt-4", "messages": []}
)

# Test 7: Custom client with similar API
class MyClient:
    def __init__(self):
        self.chat = self.Chat()

    class Chat:
        def completions(self):
            return self

        def create(self, **kwargs):
            return "mock"

my_client = MyClient()
result = my_client.chat.completions().create(model="test")

# Test 8: Variable names containing openai
openai_compatible = True
use_openai_format = False

# Test 9: URL or domain references
api_endpoint = "https://api.openai.com"
docs_url = "https://platform.openai.com/docs"

# Test 10: JSON or data structures
settings = {
    "api_settings": {
        "openai_key": "placeholder",
        "openai_model": "gpt-4"
    }
}

# Test 11: Environment variable names
import os
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_ORG_ID = os.getenv("OPENAI_ORG_ID")

# Test 12: Log messages or documentation
def log_api_usage():
    """
    Logs API usage for openai services
    This function tracks OpenAI API calls
    """
    print("OpenAI API called")
    log_message = "Using OpenAI GPT-4"
    return log_message

# Test 13: Test data or fixtures
TEST_OPENAI_RESPONSE = {
    "choices": [{"message": {"content": "Hello"}}]
}

# Test 14: Configuration file parsing
def parse_config():
    config_text = """
    [openai]
    api_key = sk-test
    model = gpt-4
    """
    return config_text

# Test 15: Import of unrelated packages
import open  # hypothetical 'open' package
import ai  # hypothetical 'ai' package

# These imports are NOT 'import openai' or 'from openai import'
# so they should not match
