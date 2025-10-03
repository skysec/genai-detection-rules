#!/usr/bin/env python3
"""
Positive test cases for LlamaIndex memory detection.
These patterns should be detected by the detect-memory-llamaindex.yaml rule.
"""

# Core LlamaIndex memory imports - VERY HIGH CONFIDENCE
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.memory import SimpleComposableMemory
from llama_index.memory import ChatMemoryBuffer as LegacyMemory

# Memory instantiation - VERY HIGH CONFIDENCE
memory = ChatMemoryBuffer.from_defaults()
chat_memory = ChatMemoryBuffer.from_defaults(token_limit=2000)
buffer_memory = ChatMemoryBuffer.from_defaults(
    token_limit=1500,
    tokenizer_fn=tokenizer
)

# Memory with custom configuration
memory_with_config = ChatMemoryBuffer.from_defaults(
    token_limit=3000,
    tokenizer_fn=custom_tokenizer
)

# Chat engine with memory - HIGH CONFIDENCE
chat_engine = index.as_chat_engine(
    chat_mode="context",
    memory=memory,
    system_prompt=system_prompt
)

query_engine = index.as_chat_engine(
    chat_mode="condense_question",
    memory=chat_memory,
    verbose=True
)

# Memory-specific method calls - VERY HIGH CONFIDENCE
memory.reset()
chat_memory.reset()
all_messages = memory.get_all()
recent_messages = memory.get("recent")
memory.put("key", "value")
memory.put("context", conversation_context)

# Real-world usage examples
def setup_chat_with_memory():
    from llama_index.core import VectorStoreIndex

    # Create memory buffer
    memory = ChatMemoryBuffer.from_defaults(token_limit=2000)

    # Create chat engine with memory
    chat_engine = index.as_chat_engine(
        chat_mode="context",
        memory=memory,
        system_prompt="You are a helpful assistant."
    )

    return chat_engine

def conversation_with_memory():
    # Initialize memory
    memory = ChatMemoryBuffer.from_defaults(token_limit=1500)

    # Create chat engine
    chat_engine = index.as_chat_engine(
        chat_mode="condense_question",
        memory=memory
    )

    # Multiple chat exchanges
    response1 = chat_engine.chat("What is machine learning?")
    response2 = chat_engine.chat("Can you give me examples?")
    response3 = chat_engine.chat("How does it relate to AI?")

    # Access memory
    all_messages = memory.get_all()
    memory.put("summary", "Discussed ML and AI concepts")

    return chat_engine, all_messages

def custom_memory_implementation():
    # Custom token limit
    memory = ChatMemoryBuffer.from_defaults(
        token_limit=5000,
        tokenizer_fn=lambda x: len(x.split())
    )

    # Reset memory when needed
    memory.reset()

    # Store custom data
    memory.put("user_preferences", {"language": "python", "level": "advanced"})
    memory.put("session_start", "2024-01-01T10:00:00Z")

    return memory

def multi_modal_chat_with_memory():
    # Memory for multi-modal conversations
    memory = ChatMemoryBuffer.from_defaults(token_limit=3000)

    # Multi-modal chat engine
    chat_engine = index.as_chat_engine(
        chat_mode="context",
        memory=memory,
        similarity_top_k=5
    )

    # Memory operations
    memory.put("image_context", "User uploaded an image of a cat")
    context = memory.get("image_context")

    return chat_engine

# Advanced memory patterns
class ChatBot:
    def __init__(self, index):
        self.memory = ChatMemoryBuffer.from_defaults(token_limit=2000)
        self.chat_engine = index.as_chat_engine(
            chat_mode="context",
            memory=self.memory
        )

    def chat(self, message):
        response = self.chat_engine.chat(message)
        return response

    def reset_conversation(self):
        self.memory.reset()

    def get_conversation_history(self):
        return self.memory.get_all()

    def save_context(self, key, value):
        self.memory.put(key, value)

    def load_context(self, key):
        return self.memory.get(key)

# Memory with different chat modes
def create_chat_engines():
    memory1 = ChatMemoryBuffer.from_defaults(token_limit=1000)
    memory2 = ChatMemoryBuffer.from_defaults(token_limit=2000)
    memory3 = ChatMemoryBuffer.from_defaults(token_limit=3000)

    engines = {
        'context': index.as_chat_engine(
            chat_mode="context",
            memory=memory1
        ),
        'condense': index.as_chat_engine(
            chat_mode="condense_question",
            memory=memory2
        ),
        'simple': index.as_chat_engine(
            chat_mode="simple",
            memory=memory3
        )
    }

    return engines

# Memory state management
def manage_memory_state():
    memory = ChatMemoryBuffer.from_defaults(token_limit=2000)

    # Store various types of data
    memory.put("user_id", "user_123")
    memory.put("session_id", "session_456")
    memory.put("preferences", {"theme": "dark", "language": "en"})
    memory.put("last_query", "What is the weather like?")

    # Retrieve data
    user_id = memory.get("user_id")
    preferences = memory.get("preferences")
    all_data = memory.get_all()

    # Reset when needed
    if len(all_data) > 100:
        memory.reset()

    return memory

# Function calls with memory parameter - HIGH CONFIDENCE
def create_agent_with_memory():
    memory = ChatMemoryBuffer.from_defaults(token_limit=1500)

    agent = create_chat_agent(
        tools=tools,
        memory=memory,
        verbose=True
    )

    return agent

def setup_retrieval_chat():
    memory = ChatMemoryBuffer.from_defaults(token_limit=2500)

    chat_engine = RetrievalChatEngine.from_defaults(
        retriever=retriever,
        memory=memory,
        chat_mode="context"
    )

    return chat_engine

# Memory buffer with custom tokenizer
def custom_tokenizer_memory():
    def custom_tokenizer(text):
        # Custom tokenization logic
        return len(text.split())

    memory = ChatMemoryBuffer.from_defaults(
        token_limit=2000,
        tokenizer_fn=custom_tokenizer
    )

    return memory

# Multiple memory instances
memories = [
    ChatMemoryBuffer.from_defaults(token_limit=1000),
    ChatMemoryBuffer.from_defaults(token_limit=2000),
    ChatMemoryBuffer.from_defaults(token_limit=3000)
]

# Memory operations in loops
for i, memory in enumerate(memories):
    memory.put(f"instance_{i}", f"Memory instance {i}")
    memory.reset()
    data = memory.get_all()

# Memory with index operations
def index_with_memory():
    from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

    # Load documents
    documents = SimpleDirectoryReader("data").load_data()
    index = VectorStoreIndex.from_documents(documents)

    # Create memory
    memory = ChatMemoryBuffer.from_defaults(token_limit=2000)

    # Chat engine with memory
    chat_engine = index.as_chat_engine(
        chat_mode="context",
        memory=memory,
        system_prompt="You are a helpful assistant with access to documents."
    )

    return chat_engine, memory