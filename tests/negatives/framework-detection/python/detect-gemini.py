"""
Negative test cases for Google Gemini detection rule
These should NOT be detected by the detect-gemini rule
"""

# Test 1: Comment mentioning gemini
# This code uses gemini API but doesn't actually import it

# Test 2: String containing gemini
api_name = "gemini"
description = "This uses the Google Gemini API"
model_name = "gemini-1.5-flash"

# Test 3: Dictionary with gemini keys
config = {
    "provider": "gemini",
    "model": "gemini-pro",
    "api_key": "AIzaSy..."
}

# Test 4: Custom class named similar to Gemini classes
class GenerativeModel:
    """Custom class - not the actual Gemini model"""
    def __init__(self, model_name):
        self.model = model_name

    def generate_content(self, prompt):
        return "mock response"

# Test 5: Function with similar naming
def generate_content(prompt):
    """Custom function - not using Gemini"""
    return {"text": "mock response"}

# Test 6: HTTP client calls to Gemini API (not using SDK)
import requests

response = requests.post(
    "https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateContent",
    headers={"Authorization": "Bearer token"},
    json={"contents": [{"parts": [{"text": "Hello"}]}]}
)

# Test 7: Import other Google packages
import google.auth
import google.cloud.storage
from google.oauth2 import service_account

# Test 8: Custom client with similar API
class MyGenerativeClient:
    def __init__(self):
        self.model = "custom-model"

    def start_chat(self):
        return self.ChatSession()

    class ChatSession:
        def send_message(self, text):
            return "mock"

client = MyGenerativeClient()
chat = client.start_chat()
response = chat.send_message("Hello")

# Test 9: Variable names containing gemini
gemini_enabled = True
use_gemini_api = False
gemini_model_name = "gemini-pro"

# Test 10: URL or domain references
api_endpoint = "https://generativelanguage.googleapis.com"
docs_url = "https://ai.google.dev/docs"

# Test 11: JSON or data structures
settings = {
    "gemini_settings": {
        "api_key": "placeholder",
        "model": "gemini-1.5-flash"
    }
}

# Test 12: Environment variable names
import os
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Test 13: Log messages or documentation
def log_api_usage():
    """
    Logs API usage for gemini services
    This function tracks Google Gemini API calls
    """
    print("Gemini API called")
    log_message = "Using Google Gemini Pro"
    return log_message

# Test 14: Test data or fixtures
TEST_GEMINI_RESPONSE = {
    "candidates": [{"content": {"parts": [{"text": "Hello"}]}}]
}

# Test 15: Configuration file parsing
def parse_gemini_config():
    config_text = """
    [gemini]
    api_key = AIzaSy...
    model = gemini-pro
    """
    return config_text

# Test 16: Import unrelated packages with similar names
import generative  # hypothetical package
import ai_models  # hypothetical package

# Test 17: Class method with similar name but different package
class AIProvider:
    def configure(self, api_key):
        self.api_key = api_key

    def list_models(self):
        return ["model1", "model2"]

provider = AIProvider()
provider.configure(api_key="test")
models = provider.list_models()

# Test 18: Comments about using Gemini
# TODO: Add support for Google Gemini API
# FIXME: Gemini integration not working

# Test 19: Documentation strings
"""
This module would integrate with Google Gemini
if we decide to use generative AI in the future.
"""

# Test 20: Mock implementations for testing
class MockGenerativeModel:
    """Mock for testing - not real Gemini"""
    def __init__(self, model_name):
        self.model = model_name

    def generate_content(self, prompt):
        return type('Response', (), {'text': 'mock'})()

    def start_chat(self):
        return type('Chat', (), {
            'send_message': lambda x: type('Response', (), {'text': 'mock'})()
        })()
