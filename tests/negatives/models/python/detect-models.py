"""
Negative test cases for AI models detection rule
These should NOT be detected by the detect-ai-models rule
"""

# Test 1: Comments mentioning model names
# This code uses gpt-4 for processing
# We prefer claude-3 over gemini-pro

# Test 2: String variables not assigned to model parameter
model_description = "We use gpt-4 in production"
documentation = "Claude-3-opus is the best model"
notes = "Consider gemini-pro for embeddings"

# Test 3: Dictionary keys that don't match model patterns
config = {
    "model_name": "custom-model",
    "model_type": "transformer",
    "model_version": "v1.0",
    "engine": "neural-network"
}

# Test 4: Function/class names with model-like strings
def train_gpt_model():
    """Train a custom GPT-like model"""
    pass

class ClaudeWrapper:
    """Wrapper for claude-like functionality"""
    pass

def process_with_gemini_logic():
    """Process data using gemini-inspired logic"""
    pass

# Test 5: Non-AI model references
database_model = "postgresql"
ml_model = "random-forest"
prediction_model = "linear-regression"

# Test 6: URL and endpoint references
api_url = "https://api.openai.com/v1/chat/completions"
endpoint = "/models/gpt-4/info"
docs_url = "https://platform.openai.com/docs/models/gpt-4"

# Test 7: Custom model names that don't match patterns
response = client.chat.completions.create(model="my-custom-model")
response = client.chat.completions.create(model="company-llm-v2")
response = client.chat.completions.create(model="internal-ai-engine")

# Test 8: Variable assignments without AI model patterns
model = "custom-transformer"
ai_model = "proprietary-llm"
llm_name = "in-house-model"

# Test 9: Model names as part of larger strings (not exact matches)
description = "The gpt-4-like model performs well"
note = "Based on claude-3 architecture"
comment = "Similar to gemini-pro functionality"

# Test 10: Dictionary with model key but non-AI model value
settings = {"model": "custom-engine", "version": "1.0"}
params = {"model": "proprietary", "temperature": 0.7}

# Test 11: Model in log messages
import logging
logging.info("Using model: custom-model-v1")
print(f"Model: {custom_model_name}")

# Test 12: Test data and fixtures
TEST_MODEL = "test-model-fixture"
MOCK_MODEL = "mock-ai-model"
SAMPLE_MODEL = "sample-model-data"

# Test 13: Configuration file content (as strings)
config_yaml = """
model:
  name: custom-model
  version: 1.0
"""

config_json = '{"model": "custom-engine", "api": "v1"}'

# Test 14: Environment variables
import os
MODEL_NAME = os.getenv("MODEL_NAME", "default-model")
AI_MODEL = os.getenv("AI_MODEL")

# Test 15: Class attributes with model-like names
class AIConfig:
    model_type = "neural-network"
    model_architecture = "transformer"
    model_framework = "pytorch"

# Test 16: Function parameters named model (but with custom values)
def initialize_model(model="custom-model", config=None):
    """Initialize with custom model"""
    pass

# Test 17: Model comparison or conditional logic
if model_name in ["gpt", "claude", "gemini"]:  # Generic names, not specific models
    process()

# Test 18: Model name fragments (incomplete)
model_prefix = "gpt"
model_family = "claude"
model_series = "gemini"

# Test 19: Open source model names that don't match our patterns
response = client.chat(model="falcon-40b")
response = client.chat(model="vicuna-13b")
response = client.chat(model="alpaca-7b")
response = client.chat(model="bloom-176b")

# Test 20: Version numbers alone
version = "3.5"
model_ver = "4.0"
api_version = "2.1"
