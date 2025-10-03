#!/usr/bin/env python3
"""
Negative test cases for LangChain memory detection.
These patterns should NOT be detected by the detect-memory-langchain.yaml rule.
"""

import os
import json
import redis
import sqlite3
from typing import Dict, List, Any

# Generic memory/storage operations (not LangChain)
import psutil
import gc

# Generic classes with similar method names (but not LangChain)
class GenericMemory:
    def __init__(self):
        self.data = {}

    def save_context(self, key, value):
        # Not LangChain - generic save
        self.data[key] = value

    def load_memory_variables(self, filters=None):
        # Not LangChain - generic load
        return self.data

    def clear(self):
        # Generic clear method
        self.data.clear()

# Generic conversation classes (not LangChain)
class ChatBot:
    def __init__(self):
        self.history = []

    def save_context(self, input_data, output_data):
        # Not LangChain save_context
        self.history.append((input_data, output_data))

    def load_memory_variables(self, query):
        # Not LangChain load_memory_variables
        return {"history": self.history}

# Generic database operations
class DatabaseConnection:
    def __init__(self, connection_string):
        self.connection_string = connection_string

    def save_context(self, table, data):
        # Not LangChain - database save
        pass

    def load_memory_variables(self, table):
        # Not LangChain - database load
        return {}

# Generic file operations
def save_context_to_file(filename, context):
    """Generic file save (not LangChain)"""
    with open(filename, 'w') as f:
        json.dump(context, f)

def load_memory_variables_from_file(filename):
    """Generic file load (not LangChain)"""
    with open(filename, 'r') as f:
        return json.load(f)

# Generic cache/memory management
class CacheManager:
    def __init__(self):
        self.cache = {}

    def save_context(self, key, value, ttl=3600):
        # Not LangChain - cache save
        self.cache[key] = {"value": value, "ttl": ttl}

    def load_memory_variables(self, key):
        # Not LangChain - cache load
        return self.cache.get(key, {})

# System memory operations
def check_memory_usage():
    """System memory monitoring (not LangChain)"""
    memory_info = psutil.virtual_memory()
    return {
        "total": memory_info.total,
        "available": memory_info.available,
        "percent": memory_info.percent
    }

def garbage_collection():
    """Python garbage collection (not LangChain)"""
    gc.collect()

# Generic AI/ML operations (not LangChain memory)
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

class TextProcessor:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.memory = {}  # Generic memory variable

    def process_text(self, text):
        # Not LangChain processing
        return self.vectorizer.fit_transform([text])

    def save_context(self, text, result):
        # Not LangChain - generic save
        self.memory[text] = result

# Generic conversation tracking
class ConversationTracker:
    def __init__(self):
        self.conversations = []

    def add_exchange(self, user_input, bot_response):
        # Not LangChain exchange
        self.conversations.append({
            "user": user_input,
            "bot": bot_response,
            "timestamp": time.time()
        })

    def get_history(self):
        return self.conversations

# Generic Redis operations
def redis_operations():
    r = redis.Redis(host='localhost', port=6379, db=0)

    # Generic Redis operations (not LangChain)
    r.set("session_123", "some_data")
    r.get("session_123")
    r.delete("session_123")

# Generic PostgreSQL operations
def postgres_operations():
    import psycopg2

    conn = psycopg2.connect(
        host="localhost",
        database="testdb",
        user="user",
        password="password"
    )

    # Generic database operations (not LangChain)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM conversations")
    results = cursor.fetchall()

    return results

# Generic variable assignments (should not be detected)
memory = "just a string variable"
chat_memory = []
message_history = {}
conversation_history = ["msg1", "msg2"]

# Generic function parameters
def process_data(data, memory=None, chat_memory=None):
    """Generic function with memory parameters (not LangChain)"""
    if memory:
        print(f"Processing with memory: {memory}")
    return data

def create_session(session_id, memory_store=None):
    """Generic session creation (not LangChain)"""
    return {"id": session_id, "store": memory_store}

# Generic class methods
class GenericProcessor:
    def __init__(self):
        self.memory = []

    def save_context(self, item):
        # Not LangChain - just appending to list
        self.memory.append(item)

    def load_memory_variables(self):
        # Not LangChain - just returning list
        return self.memory

    def clear(self):
        # Generic clear
        self.memory.clear()

# Generic imports with similar names
from some_library import ConversationBuffer
from another_lib import MemoryManager

# Generic conversation patterns
def simple_chat():
    history = []

    while True:
        user_input = input("User: ")
        if user_input.lower() == 'quit':
            break

        # Generic response (not LangChain)
        response = f"Response to: {user_input}"

        # Generic history tracking (not LangChain)
        history.append({"user": user_input, "bot": response})

        print(f"Bot: {response}")

# Generic configuration
CONFIG = {
    "memory_limit": 1000,
    "session_timeout": 3600,
    "persistence": True
}

# Generic utilities
def save_to_file(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f)

def load_from_file(filename):
    with open(filename, 'r') as f:
        return json.load(f)

# Generic API client
class APIClient:
    def __init__(self, endpoint):
        self.endpoint = endpoint
        self.session_data = {}

    def save_context(self, session_id, data):
        # Generic API save (not LangChain)
        self.session_data[session_id] = data

    def load_memory_variables(self, session_id):
        # Generic API load (not LangChain)
        return self.session_data.get(session_id, {})

# File system operations
import tempfile

def create_temp_memory():
    """Generic temporary file operations (not LangChain)"""
    temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
    temp_file.write("temporary data")
    temp_file.close()
    return temp_file.name

# Generic logging
import logging

logger = logging.getLogger(__name__)

def log_conversation(user_input, bot_response):
    """Generic logging (not LangChain)"""
    logger.info(f"User: {user_input}, Bot: {bot_response}")

# Generic data structures
conversation_buffer = []
message_queue = []
session_store = {}

# Generic functions that might have similar names
def predict(input_data, context=None):
    """Generic prediction function (not LangChain)"""
    return f"Prediction for: {input_data}"

def process_input(text, memory_context=None):
    """Generic text processing (not LangChain)"""
    return text.upper()

# Generic chain pattern (not LangChain)
class ProcessingChain:
    def __init__(self, steps):
        self.steps = steps

    def run(self, input_data):
        result = input_data
        for step in self.steps:
            result = step(result)
        return result