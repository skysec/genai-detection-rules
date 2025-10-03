#!/usr/bin/env python3
"""
Positive test cases for LangChain memory detection.
These patterns should be detected by the detect-memory-langchain.yaml rule.
"""

# Core LangChain memory imports - VERY HIGH CONFIDENCE
from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationSummaryMemory
from langchain.memory import ConversationBufferWindowMemory
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationChain

# Additional memory imports
from langchain.memory import ConversationTokenBufferMemory
from langchain.memory import ConversationEntityMemory
from langchain.memory.chat_message_histories import RedisChatMessageHistory
from langchain.memory.chat_message_histories import PostgresChatMessageHistory
from langchain.memory.chat_message_histories import FileChatMessageHistory

# Memory instantiation - VERY HIGH CONFIDENCE
memory = ConversationBufferMemory()
buffer_memory = ConversationBufferMemory(return_messages=True)
summary_memory = ConversationSummaryMemory(llm=llm)
window_memory = ConversationBufferWindowMemory(k=5)
token_memory = ConversationTokenBufferMemory(llm=llm, max_token_limit=100)

# Memory with configuration
memory_with_key = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

entity_memory = ConversationEntityMemory(
    llm=llm,
    entity_extraction_prompt=entity_prompt
)

# Conversation chain with memory - HIGH CONFIDENCE
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

chat_chain = ConversationChain(
    llm=llm,
    memory=buffer_memory
)

# Memory usage patterns - VERY HIGH CONFIDENCE
memory.save_context({"input": "Hello"}, {"output": "Hi there!"})
memory.save_context(inputs, outputs)
variables = memory.load_memory_variables({})
context = memory.load_memory_variables({"input": "test"})

# Chat message history - HIGH CONFIDENCE
redis_history = RedisChatMessageHistory(
    session_id="session_123",
    url="redis://localhost:6379"
)

postgres_history = PostgresChatMessageHistory(
    connection_string="postgresql://user:pass@localhost/dbname",
    session_id="session_456"
)

file_history = FileChatMessageHistory(file_path="chat_history.json")

# Real-world usage examples
def setup_conversation_memory():
    # Initialize memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    # Create conversation chain
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=True
    )

    return conversation

def conversation_with_summary():
    # Summary memory for long conversations
    memory = ConversationSummaryMemory(
        llm=llm,
        memory_key="chat_history",
        return_messages=True
    )

    # Save conversation context
    memory.save_context(
        {"input": "What is machine learning?"},
        {"output": "Machine learning is a subset of AI..."}
    )

    # Load memory variables
    history = memory.load_memory_variables({})

    return memory

def windowed_conversation():
    # Window memory keeps only recent messages
    memory = ConversationBufferWindowMemory(
        k=3,
        memory_key="chat_history",
        return_messages=True
    )

    # Multiple exchanges
    exchanges = [
        ("Hello", "Hi there!"),
        ("How are you?", "I'm doing well, thank you!"),
        ("What can you help with?", "I can help with various tasks...")
    ]

    for user_input, ai_output in exchanges:
        memory.save_context(
            {"input": user_input},
            {"output": ai_output}
        )

    return memory

def persistent_memory_example():
    # Redis-backed persistent memory
    redis_history = RedisChatMessageHistory(
        session_id="user_123",
        url="redis://localhost:6379/0"
    )

    memory = ConversationBufferMemory(
        chat_memory=redis_history,
        memory_key="chat_history",
        return_messages=True
    )

    # Create conversation with persistent memory
    conversation = ConversationChain(
        llm=llm,
        memory=memory
    )

    return conversation

def entity_memory_example():
    # Entity memory tracks entities in conversation
    memory = ConversationEntityMemory(
        llm=llm,
        memory_key="chat_history",
        entity_extraction_prompt=entity_prompt
    )

    # Save context with entities
    memory.save_context(
        {"input": "John works at Anthropic"},
        {"output": "That's interesting! Anthropic is an AI safety company."}
    )

    # Load variables including entities
    variables = memory.load_memory_variables({"input": "What does John do?"})

    return memory

# Memory in different chain types
class ChatBot:
    def __init__(self):
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        self.conversation = ConversationChain(
            llm=llm,
            memory=self.memory
        )

    def chat(self, message):
        response = self.conversation.predict(input=message)
        return response

    def get_history(self):
        return self.memory.load_memory_variables({})

    def clear_memory(self):
        self.memory.clear()

# Advanced memory configurations
def token_limited_memory():
    # Token buffer memory with limits
    memory = ConversationTokenBufferMemory(
        llm=llm,
        max_token_limit=1000,
        return_messages=True
    )

    return memory

def summary_buffer_memory():
    # Combined summary and buffer memory
    memory = ConversationSummaryBufferMemory(
        llm=llm,
        max_token_limit=500,
        return_messages=True
    )

    # Save multiple contexts
    memory.save_context(
        {"input": "Explain quantum computing"},
        {"output": "Quantum computing uses quantum mechanics..."}
    )

    return memory

# Memory parameters in function calls - HIGH CONFIDENCE
def create_agent_with_memory():
    agent = create_conversational_agent(
        tools=tools,
        llm=llm,
        memory=memory,
        verbose=True
    )

    return agent

def setup_retrieval_qa():
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True
    )

    return qa_chain

# Multiple memory types
memory_types = {
    'buffer': ConversationBufferMemory(),
    'summary': ConversationSummaryMemory(llm=llm),
    'window': ConversationBufferWindowMemory(k=5),
    'token': ConversationTokenBufferMemory(llm=llm, max_token_limit=200)
}

# Memory with external storage
for session_id in ["user_1", "user_2", "user_3"]:
    redis_memory = ConversationBufferMemory(
        chat_memory=RedisChatMessageHistory(
            session_id=session_id,
            url="redis://localhost:6379"
        )
    )

# Function calls with memory parameter
chat_chain = ConversationChain(
    llm=llm,
    memory=buffer_memory,
    verbose=True
)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True
)