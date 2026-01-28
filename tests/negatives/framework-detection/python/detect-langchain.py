"""
Negative test cases for LangChain detection rule
These should NOT be detected by the detect-langchain rule
"""

# Test 1: Comments mentioning langchain
# This code uses langchain framework but doesn't import it

# Test 2: Strings containing langchain
framework_name = "langchain"
description = "This uses the LangChain framework"

# Test 3: Dictionary with langchain keys
config = {
    "framework": "langchain",
    "llm_provider": "openai",
    "chain_type": "stuff"
}

# Test 4: Custom classes with similar names
class ChatOpenAI:
    """Custom class - not the actual LangChain ChatOpenAI"""
    def __init__(self, model):
        self.model = model

class LLMChain:
    """Custom chain class"""
    def __init__(self, llm, prompt):
        self.llm = llm
        self.prompt = prompt

# Test 5: Functions with similar names
def initialize_agent(tools, llm, agent_type):
    """Custom function - not LangChain's"""
    return {"tools": tools, "llm": llm}

def create_react_agent(llm, tools, prompt):
    """Custom agent creator"""
    return "mock_agent"

# Test 6: HTTP client calls (not using SDK)
import requests

response = requests.post(
    "https://api.langchain.com/v1/chain",
    headers={"Authorization": "Bearer token"},
    json={"chain": "my_chain"}
)

# Test 7: Import unrelated packages
import lang  # hypothetical package
import chain  # hypothetical package
import language_chain  # not langchain

# Test 8: Custom implementation with similar API
class MyLLM:
    def __init__(self, model_name):
        self.model = model_name

    def invoke(self, prompt):
        return "mock response"

class MyChain:
    def __init__(self, llm):
        self.llm = llm

    def run(self, input_text):
        return self.llm.invoke(input_text)

my_llm = MyLLM("gpt-4")
my_chain = MyChain(my_llm)
result = my_chain.run("Hello")

# Test 9: Variable names containing langchain
langchain_enabled = True
use_langchain = False
langchain_config = {}

# Test 10: URLs or documentation references
docs_url = "https://python.langchain.com/docs"
github_url = "https://github.com/langchain-ai/langchain"

# Test 11: JSON or data structures
settings = {
    "langchain_settings": {
        "model": "gpt-4",
        "temperature": 0.7
    }
}

# Test 12: Environment variables
import os
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_TRACING = os.getenv("LANGCHAIN_TRACING")

# Test 13: Log messages
def log_framework_usage():
    """
    Logs framework usage for langchain
    This function tracks LangChain operations
    """
    print("LangChain chain executed")
    log_message = "Using LangChain with OpenAI"
    return log_message

# Test 14: Test data or mock responses
TEST_LANGCHAIN_RESPONSE = {
    "output": "This is a test response",
    "metadata": {"model": "gpt-4"}
}

MOCK_CHAIN_OUTPUT = "Mock LangChain output"

# Test 15: Configuration parsing
def parse_langchain_config():
    config_text = """
    [langchain]
    provider = openai
    model = gpt-4
    temperature = 0.7
    """
    return config_text

# Test 16: Documentation strings
"""
This module integrates with LangChain framework
to provide chain-based LLM operations.
"""

class AIOrchestrator:
    """
    Orchestrator for AI operations using LangChain patterns.
    Note: This is a custom implementation, not using LangChain directly.
    """
    pass

# Test 17: Comments about implementation
# TODO: Migrate to LangChain framework
# FIXME: LangChain integration pending
# NOTE: Consider using LangChain for this

# Test 18: Mock classes for testing
class MockChatOpenAI:
    """Mock ChatOpenAI for testing - not real LangChain"""
    def __init__(self, model="gpt-4", temperature=0):
        self.model = model
        self.temperature = temperature

    def invoke(self, prompt):
        return {"output": "mock response"}

class MockLLMChain:
    """Mock LLM chain for testing"""
    def __init__(self, llm, prompt):
        self.llm = llm
        self.prompt = prompt

    def run(self, input_text):
        return f"Mock output for: {input_text}"

# Test 19: Custom vector store implementation
class VectorStore:
    """Custom vector store - not LangChain's"""
    @classmethod
    def from_documents(cls, documents, embeddings):
        return cls(documents)

    def __init__(self, documents):
        self.documents = documents

# Test 20: Custom text splitter
class TextSplitter:
    """Custom text splitter - not LangChain's"""
    def __init__(self, chunk_size=1000):
        self.chunk_size = chunk_size

    def split_text(self, text):
        return [text[i:i+self.chunk_size]
                for i in range(0, len(text), self.chunk_size)]

splitter = TextSplitter(chunk_size=500)
chunks = splitter.split_text("Some long text")

# Test 21: Custom prompt template
class PromptTemplate:
    """Custom prompt template - not LangChain's"""
    def __init__(self, template):
        self.template = template

    def format(self, **kwargs):
        return self.template.format(**kwargs)

prompt = PromptTemplate(template="Tell me about {topic}")
formatted = prompt.format(topic="AI")

# Test 22: Custom output parser
class StrOutputParser:
    """Custom string parser - not LangChain's"""
    def parse(self, output):
        return str(output)

parser = StrOutputParser()
result = parser.parse({"key": "value"})
