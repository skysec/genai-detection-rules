"""
Negative test cases for Hugging Face detection rule
These should NOT be detected by the detect-huggingface rule
"""

# Test 1: Comments mentioning huggingface
# This code uses huggingface transformers but doesn't import it

# Test 2: Strings containing huggingface
library_name = "huggingface"
description = "This uses the Hugging Face transformers library"
model_source = "huggingface hub"

# Test 3: Dictionary with huggingface keys
config = {
    "library": "huggingface",
    "model": "bert-base-uncased",
    "source": "transformers"
}

# Test 4: Custom classes with similar names
class AutoModel:
    """Custom class - not HuggingFace AutoModel"""
    @classmethod
    def from_pretrained(cls, model_name):
        return cls(model_name)

    def __init__(self, model_name):
        self.model = model_name

class AutoTokenizer:
    """Custom tokenizer class"""
    @classmethod
    def from_pretrained(cls, model_name):
        return cls(model_name)

    def __init__(self, model_name):
        self.model = model_name

# Test 5: Functions with similar names
def pipeline(task, model=None):
    """Custom pipeline function - not HuggingFace"""
    return {"task": task, "model": model}

def load_dataset(name):
    """Custom dataset loader"""
    return {"name": name, "data": []}

# Test 6: HTTP client calls (not using SDK)
import requests

response = requests.get(
    "https://huggingface.co/models/bert-base-uncased",
    headers={"Authorization": "Bearer hf_token"}
)

# Test 7: Import unrelated packages
import transformer  # hypothetical package - not transformers
import dataset  # hypothetical package - not datasets
import hub  # hypothetical package

# Test 8: Custom implementation with similar API
class MyModel:
    def __init__(self, model_name):
        self.model = model_name

    @classmethod
    def from_pretrained(cls, model_name):
        return cls(model_name)

    def generate(self, input_text):
        return "mock output"

    def push_to_hub(self, repo_name):
        print(f"Pushing to {repo_name}")

model = MyModel("my-model")
output = model.generate("test")
model.push_to_hub("my-repo")

# Test 9: Variable names containing huggingface
huggingface_model = "bert-base-uncased"
use_huggingface = False
transformers_enabled = True

# Test 10: URLs or documentation references
docs_url = "https://huggingface.co/docs"
hub_url = "https://huggingface.co/models"
model_card_url = "https://huggingface.co/bert-base-uncased"

# Test 11: JSON or data structures
settings = {
    "huggingface_settings": {
        "token": "hf_...",
        "cache_dir": "./cache"
    },
    "transformers_config": {
        "model": "bert-base-uncased"
    }
}

# Test 12: Environment variables
import os
HF_TOKEN = os.getenv("HF_TOKEN")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
TRANSFORMERS_CACHE = os.getenv("TRANSFORMERS_CACHE")

# Test 13: Log messages
def log_model_usage():
    """
    Logs model usage for huggingface models
    This function tracks Hugging Face Transformers operations
    """
    print("HuggingFace model loaded")
    log_message = "Using transformers library for NLP"
    return log_message

# Test 14: Test data or mock responses
TEST_HF_RESPONSE = {
    "model": "bert-base-uncased",
    "output": [0.1, 0.2, 0.3]
}

MOCK_TOKENIZER_OUTPUT = {
    "input_ids": [101, 2003, 102],
    "attention_mask": [1, 1, 1]
}

# Test 15: Configuration parsing
def parse_hf_config():
    config_text = """
    [huggingface]
    token = hf_...
    model = bert-base-uncased
    cache_dir = ./cache
    """
    return config_text

# Test 16: Documentation strings
"""
This module integrates with Hugging Face models
to provide NLP capabilities using transformers.
"""

class NLPProcessor:
    """
    NLP processor using Hugging Face patterns.
    Note: This is a custom implementation, not using HuggingFace directly.
    """
    pass

# Test 17: Comments about implementation
# TODO: Migrate to Hugging Face transformers
# FIXME: HuggingFace integration pending
# NOTE: Consider using transformers library for this

# Test 18: Mock classes for testing
class MockBertModel:
    """Mock BERT model for testing - not real HuggingFace"""
    @classmethod
    def from_pretrained(cls, model_name):
        return cls(model_name)

    def __init__(self, model_name):
        self.model = model_name

    def forward(self, input_ids):
        return {"logits": [0.1, 0.2, 0.3]}

class MockTrainer:
    """Mock trainer for testing"""
    def __init__(self, model, args):
        self.model = model
        self.args = args

    def train(self):
        print("Training...")

# Test 19: Custom dataset class
class Dataset:
    """Custom dataset class - not HuggingFace"""
    @classmethod
    def from_dict(cls, data_dict):
        return cls(data_dict)

    @classmethod
    def from_pandas(cls, df):
        return cls(df.to_dict())

    def __init__(self, data):
        self.data = data

    def push_to_hub(self, repo_name):
        print(f"Pushing to {repo_name}")

dataset = Dataset.from_dict({"text": ["hello", "world"]})
dataset.push_to_hub("my-dataset")

# Test 20: Custom hub API
class HfApi:
    """Custom Hub API - not HuggingFace"""
    def upload_file(self, path, repo_id):
        print(f"Uploading {path} to {repo_id}")

    def create_repo(self, repo_id):
        print(f"Creating repo {repo_id}")

api = HfApi()
api.upload_file("model.bin", "my-model")
api.create_repo("my-new-model")

# Test 21: Custom training arguments
class TrainingArguments:
    """Custom training arguments - not HuggingFace"""
    def __init__(self, output_dir):
        self.output_dir = output_dir

args = TrainingArguments(output_dir="./results")

# Test 22: Custom inference client
class InferenceClient:
    """Custom inference client - not HuggingFace"""
    def __init__(self, model=None):
        self.model = model

    def predict(self, text):
        return {"output": "mock prediction"}

client = InferenceClient(model="my-model")
result = client.predict("test input")
