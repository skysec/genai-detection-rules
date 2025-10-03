#!/usr/bin/env python3
"""
Negative test cases for LlamaIndex memory detection.
These patterns should NOT be detected by the detect-memory-llamaindex.yaml rule.
"""

import os
import json
import time
from typing import Dict, List, Any

# Generic memory/storage operations (not LlamaIndex)
import psutil
import gc

# Generic classes with similar method names (but not LlamaIndex)
class GenericMemory:
    def __init__(self):
        self.data = {}

    def reset(self):
        # Generic reset method (not LlamaIndex)
        self.data.clear()

    def get_all(self):
        # Generic get_all method (not LlamaIndex)
        return list(self.data.values())

    def get(self, key):
        # Generic get method (not LlamaIndex)
        return self.data.get(key)

    def put(self, key, value):
        # Generic put method (not LlamaIndex)
        self.data[key] = value

# Generic chat/conversation classes (not LlamaIndex)
class ChatBot:
    def __init__(self):
        self.history = []
        self.memory = {}

    def reset(self):
        # Not LlamaIndex reset
        self.history.clear()

    def get_all(self):
        # Not LlamaIndex get_all
        return self.history

    def put(self, key, value):
        # Not LlamaIndex put
        self.memory[key] = value

    def get(self, key):
        # Not LlamaIndex get
        return self.memory.get(key)

# Generic buffer classes (not LlamaIndex)
class BufferManager:
    def __init__(self, limit=1000):
        self.buffer = []
        self.limit = limit

    def reset(self):
        # Generic buffer reset
        self.buffer.clear()

    def get_all(self):
        # Generic buffer get all
        return self.buffer.copy()

    def put(self, item):
        # Generic buffer put
        if len(self.buffer) >= self.limit:
            self.buffer.pop(0)
        self.buffer.append(item)

    def get(self, index):
        # Generic buffer get
        return self.buffer[index] if 0 <= index < len(self.buffer) else None

# Generic factory methods (not LlamaIndex)
class ConfigManager:
    @classmethod
    def from_defaults(cls, **kwargs):
        # Generic from_defaults (not LlamaIndex)
        return cls()

    def reset(self):
        pass

# Generic cache implementations
class CacheSystem:
    def __init__(self):
        self.cache = {}

    def reset(self):
        # Generic cache reset
        self.cache.clear()

    def get_all(self):
        # Generic cache get all
        return list(self.cache.values())

    def put(self, key, value):
        # Generic cache put
        self.cache[key] = value

    def get(self, key):
        # Generic cache get
        return self.cache.get(key)

# Generic database operations
class DatabaseManager:
    def __init__(self, connection_string):
        self.connection_string = connection_string
        self.data = {}

    def reset(self):
        # Generic database reset
        self.data.clear()

    def get_all(self):
        # Generic database get all
        return list(self.data.values())

    def put(self, key, value):
        # Generic database put
        self.data[key] = value

    def get(self, key):
        # Generic database get
        return self.data.get(key)

# System memory operations
def check_memory_usage():
    """System memory monitoring (not LlamaIndex)"""
    memory_info = psutil.virtual_memory()
    return {
        "total": memory_info.total,
        "available": memory_info.available,
        "percent": memory_info.percent
    }

def garbage_collection():
    """Python garbage collection (not LlamaIndex)"""
    gc.collect()

# Generic AI/ML operations (not LlamaIndex memory)
import numpy as np

class TextProcessor:
    def __init__(self):
        self.memory = []

    def reset(self):
        # Generic reset (not LlamaIndex)
        self.memory.clear()

    def get_all(self):
        # Generic get_all (not LlamaIndex)
        return self.memory.copy()

    def put(self, text):
        # Generic put (not LlamaIndex)
        self.memory.append(text)

    def get(self, index):
        # Generic get (not LlamaIndex)
        return self.memory[index] if 0 <= index < len(self.memory) else None

# Generic conversation tracking
class ConversationTracker:
    def __init__(self):
        self.conversations = []

    def reset(self):
        # Generic conversation reset
        self.conversations.clear()

    def get_all(self):
        # Generic get all conversations
        return self.conversations.copy()

    def put(self, conversation):
        # Generic put conversation
        self.conversations.append(conversation)

    def get(self, index):
        # Generic get conversation
        return self.conversations[index] if 0 <= index < len(self.conversations) else None

# Generic file operations
def save_to_file(data, filename):
    """Generic file save (not LlamaIndex)"""
    with open(filename, 'w') as f:
        json.dump(data, f)

def load_from_file(filename):
    """Generic file load (not LlamaIndex)"""
    with open(filename, 'r') as f:
        return json.load(f)

# Generic variable assignments (should not be detected)
memory = "just a string variable"
chat_memory = []
buffer_memory = {}
conversation_memory = ["msg1", "msg2"]

# Generic function parameters
def process_data(data, memory=None):
    """Generic function with memory parameter (not LlamaIndex)"""
    if memory:
        print(f"Processing with memory: {memory}")
    return data

def create_session(session_id, memory=None):
    """Generic session creation (not LlamaIndex)"""
    return {"id": session_id, "memory": memory}

# Generic class methods with similar names
class GenericProcessor:
    def __init__(self):
        self.memory = []

    def reset(self):
        # Generic reset
        self.memory.clear()

    def get_all(self):
        # Generic get_all
        return self.memory.copy()

    def put(self, item):
        # Generic put
        self.memory.append(item)

    def get(self, index):
        # Generic get
        return self.memory[index] if 0 <= index < len(self.memory) else None

# Generic factory patterns
class ComponentFactory:
    @staticmethod
    def from_defaults(**kwargs):
        # Generic factory method (not LlamaIndex)
        return GenericProcessor()

# Generic API client
class APIClient:
    def __init__(self, endpoint):
        self.endpoint = endpoint
        self.session_data = {}

    def reset(self):
        # Generic API reset
        self.session_data.clear()

    def get_all(self):
        # Generic API get all
        return list(self.session_data.values())

    def put(self, key, value):
        # Generic API put
        self.session_data[key] = value

    def get(self, key):
        # Generic API get
        return self.session_data.get(key)

# Generic configuration
CONFIG = {
    "memory_limit": 1000,
    "buffer_size": 500,
    "cache_ttl": 3600
}

# Generic utilities
def reset_system():
    """Generic system reset (not LlamaIndex)"""
    pass

def get_all_processes():
    """Generic process listing (not LlamaIndex)"""
    return []

def put_data_in_cache(key, value):
    """Generic cache put (not LlamaIndex)"""
    pass

def get_data_from_cache(key):
    """Generic cache get (not LlamaIndex)"""
    return None

# Generic data structures
data_buffer = []
message_queue = []
session_store = {}

# Generic engine patterns (not LlamaIndex)
class ProcessingEngine:
    def __init__(self, config):
        self.config = config
        self.memory = []

    def process(self, input_data):
        return f"Processed: {input_data}"

    def reset(self):
        self.memory.clear()

# Generic chat interfaces
class SimpleChatInterface:
    def __init__(self):
        self.history = []

    def chat(self, message):
        # Generic chat method (not LlamaIndex)
        response = f"Response to: {message}"
        self.history.append({"user": message, "bot": response})
        return response

    def reset(self):
        self.history.clear()

# Generic imports with similar names
from some_library import ChatMemory
from another_lib import MemoryBuffer

# Generic method calls that might look similar
def create_engine(mode="simple", memory=None):
    """Generic engine creation (not LlamaIndex)"""
    return ProcessingEngine({"mode": mode, "memory": memory})

# Generic token/limit patterns
class TokenManager:
    def __init__(self, limit=1000):
        self.limit = limit
        self.tokens = []

    @classmethod
    def from_defaults(cls, token_limit=500):
        # Generic from_defaults with token_limit (not LlamaIndex)
        return cls(limit=token_limit)

    def reset(self):
        self.tokens.clear()

# Generic context managers
class ContextManager:
    def __init__(self):
        self.context = {}

    def reset(self):
        self.context.clear()

    def get_all(self):
        return self.context.copy()

    def put(self, key, value):
        self.context[key] = value

    def get(self, key):
        return self.context.get(key)