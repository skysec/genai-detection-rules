# GenAI Application Code Samples

This comprehensive guide covers code samples for popular GenAI frameworks across Python and TypeScript, demonstrating key functionalities including embeddings, tools, memory, chat, LLM integration, vector databases, and MCP (Model Context Protocol) implementation.

## Table of Contents
1. [LangChain](#langchain)
2. [LlamaIndex](#llamaindex) 
3. [Haystack](#haystack)
4. [Semantic Kernel](#semantic-kernel)
5. [AutoGen](#autogen)
6. [CrewAI](#crewai)
7. [Hugging Face Transformers](#hugging-face-transformers)
8. [Rasa](#rasa)

---

## LangChain

### Python Implementation

#### Basic LLM and Chat
```python
from langchain_openai import ChatOpenAI, OpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Initialize LLM
llm = OpenAI(
    temperature=0.7,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)

# Initialize Chat Model
chat = ChatOpenAI(
    model="gpt-4",
    temperature=0.3
)

# Chat conversation
messages = [
    SystemMessage(content="You are a helpful AI assistant."),
    HumanMessage(content="What is machine learning?")
]

response = chat(messages)
print(response.content)
```

#### Embeddings and Vector Store
```python
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document

# Initialize embeddings
embeddings = OpenAIEmbeddings()

# Create documents
documents = [
    Document(page_content="Python is a programming language.", metadata={"source": "doc1"}),
    Document(page_content="Machine learning uses algorithms to find patterns.", metadata={"source": "doc2"}),
]

# Split documents
text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# Create vector store
vectorstore = Chroma.from_documents(texts, embeddings)

# Similarity search
results = vectorstore.similarity_search("programming", k=2)
for doc in results:
    print(doc.page_content)
```

#### Tools and Agents
```python
from langchain.tools import BaseTool
from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI
import requests

class WeatherTool(BaseTool):
    name = "weather"
    description = "Get weather information for a location"
    
    def _run(self, location: str) -> str:
        # Simplified weather API call
        return f"The weather in {location} is sunny, 75°F"
    
    async def _arun(self, location: str) -> str:
        return self._run(location)

# Initialize agent with tools
llm = ChatOpenAI(temperature=0)
tools = [WeatherTool()]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

response = agent.run("What's the weather in New York?")
print(response)
```

#### Memory Implementation
```python
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(temperature=0)

# Buffer Memory
buffer_memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm=llm,
    memory=buffer_memory,
    verbose=True
)

# First interaction
response1 = conversation.predict(input="Hi, my name is Alice")
print(response1)

# Second interaction - memory retained
response2 = conversation.predict(input="What's my name?")
print(response2)

# Summary Memory for longer conversations
summary_memory = ConversationSummaryMemory(llm=llm)
summary_conversation = ConversationChain(
    llm=llm,
    memory=summary_memory,
    verbose=True
)
```

#### Retrieval-Augmented Generation (RAG)
```python
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader

# Load and process documents
loader = TextLoader("document.txt")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
texts = text_splitter.split_documents(documents)

# Create vector store
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(texts, embeddings)

# Create RAG chain
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(),
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
)

response = qa.run("What is the main topic of the document?")
print(response)
```

### TypeScript Implementation

```typescript
import { ChatOpenAI } from "@langchain/openai";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { OpenAIEmbeddings } from "@langchain/openai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { Document } from "@langchain/core/documents";
import { BufferMemory } from "langchain/memory";
import { ConversationChain } from "langchain/chains";

// Basic Chat
const chat = new ChatOpenAI({
  modelName: "gpt-4",
  temperature: 0.3,
});

const messages = [
  new SystemMessage("You are a helpful AI assistant."),
  new HumanMessage("What is TypeScript?"),
];

const response = await chat.invoke(messages);
console.log(response.content);

// Embeddings and Vector Store
const embeddings = new OpenAIEmbeddings();
const documents = [
  new Document({
    pageContent: "TypeScript is a superset of JavaScript.",
    metadata: { source: "doc1" },
  }),
  new Document({
    pageContent: "React is a JavaScript library for building UIs.",
    metadata: { source: "doc2" },
  }),
];

const vectorStore = await MemoryVectorStore.fromDocuments(
  documents,
  embeddings
);

const results = await vectorStore.similaritySearch("JavaScript", 2);
console.log(results);

// Memory
const memory = new BufferMemory();
const conversation = new ConversationChain({
  llm: chat,
  memory: memory,
});

const result1 = await conversation.call({
  input: "Hi, I'm working on a TypeScript project",
});
console.log(result1);

const result2 = await conversation.call({
  input: "What programming language am I using?",
});
console.log(result2);
```

---

## LlamaIndex

### Python Implementation

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.tools import FunctionTool
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

# Configure settings
Settings.llm = OpenAI(model="gpt-4", temperature=0.1)
Settings.embed_model = OpenAIEmbedding()

# Load and index documents
documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)

# Basic Query Engine
query_engine = index.as_query_engine()
response = query_engine.query("What is the main topic?")
print(response)

# Chat Engine with Memory
memory = ChatMemoryBuffer.from_defaults(token_limit=3900)
chat_engine = index.as_chat_engine(
    chat_mode="context",
    memory=memory,
    system_prompt="You are a helpful assistant."
)

response = chat_engine.chat("Hello, I'm interested in learning about AI")
print(response)

response = chat_engine.chat("What did I just say I was interested in?")
print(response)

# Custom Tools
def weather_function(location: str) -> str:
    """Get weather information for a location."""
    return f"Weather in {location}: Sunny, 72°F"

weather_tool = FunctionTool.from_defaults(fn=weather_function)

# Agent with Tools
from llama_index.core.agent import ReActAgent

agent = ReActAgent.from_tools([weather_tool], llm=Settings.llm, verbose=True)
response = agent.chat("What's the weather in San Francisco?")
print(response)

# Vector Store with External Database
chroma_client = chromadb.EphemeralClient()
chroma_collection = chroma_client.create_collection("documents")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

# Custom Retriever
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle

class CustomRetriever(BaseRetriever):
    def __init__(self, vector_index):
        self._vector_index = vector_index
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        nodes = self._vector_index.as_retriever(similarity_top_k=5).retrieve(query_bundle)
        # Custom filtering logic
        filtered_nodes = [node for node in nodes if node.score > 0.7]
        return filtered_nodes

custom_retriever = CustomRetriever(index)
query_engine = RetrieverQueryEngine.from_args(custom_retriever)
```

### TypeScript Implementation

```typescript
import {
  VectorStoreIndex,
  SimpleDirectoryReader,
  OpenAI,
  Settings,
  ChatMessage,
} from "llamaindex";

// Configure settings
Settings.llm = new OpenAI({ model: "gpt-4", temperature: 0.1 });

// Load documents and create index
const reader = new SimpleDirectoryReader();
const documents = await reader.loadData("./data");
const index = await VectorStoreIndex.fromDocuments(documents);

// Query Engine
const queryEngine = index.asQueryEngine();
const response = await queryEngine.query("What is machine learning?");
console.log(response.toString());

// Chat Engine
const chatEngine = index.asChatEngine();
const chatResponse = await chatEngine.chat({
  message: "Explain the key concepts",
  chatHistory: [],
});
console.log(chatResponse.response);

// Streaming responses
const stream = await chatEngine.chat({
  message: "Tell me more about this topic",
  stream: true,
});

for await (const chunk of stream) {
  process.stdout.write(chunk.response.delta);
}
```

---

## Haystack

### Python Implementation

```python
from haystack import Pipeline, Document
from haystack.components.generators import OpenAIGenerator
from haystack.components.embedders import OpenAITextEmbedder, OpenAIDocumentEmbedder
from haystack.components.retrievers import InMemoryEmbeddingRetriever
from haystack.document_stores import InMemoryDocumentStore
from haystack.components.builders import PromptBuilder
from haystack.components.tools import Tool

# Initialize components
document_store = InMemoryDocumentStore()
doc_embedder = OpenAIDocumentEmbedder()
text_embedder = OpenAITextEmbedder()
retriever = InMemoryEmbeddingRetriever(document_store=document_store)
generator = OpenAIGenerator(model="gpt-4")

# Create documents
documents = [
    Document(content="Python is a versatile programming language."),
    Document(content="Machine learning algorithms learn from data."),
    Document(content="Neural networks are inspired by the human brain."),
]

# Embed and store documents
embedded_docs = doc_embedder.run(documents=documents)
document_store.write_documents(embedded_docs["documents"])

# RAG Pipeline
template = """
Given the following information, answer the question.

Context:
{% for document in documents %}
  {{ document.content }}
{% endfor %}

Question: {{ question }}
Answer:
"""

prompt_builder = PromptBuilder(template=template)

rag_pipeline = Pipeline()
rag_pipeline.add_component("text_embedder", text_embedder)
rag_pipeline.add_component("retriever", retriever)
rag_pipeline.add_component("prompt_builder", prompt_builder)
rag_pipeline.add_component("generator", generator)

# Connect components
rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
rag_pipeline.connect("retriever", "prompt_builder.documents")
rag_pipeline.connect("prompt_builder", "generator")

# Run pipeline
result = rag_pipeline.run({
    "text_embedder": {"text": "What is Python?"},
    "prompt_builder": {"question": "What is Python?"}
})

print(result["generator"]["replies"][0])

# Custom Tool
class WeatherTool(Tool):
    def run(self, location: str) -> str:
        """Get weather information for a location."""
        return f"Weather in {location}: Sunny, 75°F"

# Chat Pipeline with Tools
chat_template = """
You are a helpful assistant with access to tools.

{% if tools_output %}
Tool Output: {{ tools_output }}
{% endif %}

User: {{ query }}
Assistant:
"""

chat_prompt_builder = PromptBuilder(template=chat_template)
weather_tool = WeatherTool()

chat_pipeline = Pipeline()
chat_pipeline.add_component("prompt_builder", chat_prompt_builder)
chat_pipeline.add_component("generator", generator)
chat_pipeline.add_component("weather_tool", weather_tool)

# Memory implementation using custom component
class ConversationMemory:
    def __init__(self):
        self.history = []
    
    def add_exchange(self, user_input: str, assistant_output: str):
        self.history.append({"user": user_input, "assistant": assistant_output})
    
    def get_context(self, max_turns: int = 5) -> str:
        recent_history = self.history[-max_turns:]
        context = ""
        for exchange in recent_history:
            context += f"User: {exchange['user']}\nAssistant: {exchange['assistant']}\n"
        return context

memory = ConversationMemory()
```

---

## Semantic Kernel

### Python Implementation

```python
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.core_plugins import TextPlugin, TimePlugin
from semantic_kernel.functions import kernel_function
import asyncio

# Initialize kernel
kernel = sk.Kernel()

# Add OpenAI service
kernel.add_service(OpenAIChatCompletion(
    service_id="chat",
    ai_model_id="gpt-4",
    api_key="your-api-key"
))

# Add built-in plugins
kernel.add_plugin(TextPlugin(), plugin_name="text")
kernel.add_plugin(TimePlugin(), plugin_name="time")

# Custom plugin
class WeatherPlugin:
    @kernel_function(
        description="Get weather information",
        name="get_weather"
    )
    def get_weather(self, location: str) -> str:
        """Get weather for a location."""
        return f"Weather in {location}: Sunny, 72°F"

kernel.add_plugin(WeatherPlugin(), plugin_name="weather")

# Semantic function (prompt)
summarize_function = kernel.create_function_from_prompt(
    prompt="""
    {{$input}}
    
    Summarize the above text in 2-3 sentences.
    """,
    function_name="summarize",
    plugin_name="text_summary"
)

async def main():
    # Use semantic function
    summary_result = await kernel.invoke(
        summarize_function,
        input="Artificial Intelligence is transforming industries..."
    )
    print(f"Summary: {summary_result}")
    
    # Use native function
    weather_result = await kernel.invoke(
        kernel.plugins["weather"]["get_weather"],
        location="Seattle"
    )
    print(f"Weather: {weather_result}")
    
    # Chain functions
    text = "Today is a beautiful day in Seattle."
    time_result = await kernel.invoke(kernel.plugins["time"]["now"])
    summary_result = await kernel.invoke(
        summarize_function,
        input=f"{text} Current time: {time_result}"
    )
    print(f"Chained result: {summary_result}")

# Memory and Planning
class ConversationMemory:
    def __init__(self):
        self.messages = []
    
    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
    
    def get_context(self) -> str:
        return "\n".join([f"{msg['role']}: {msg['content']}" for msg in self.messages[-10:]])

memory = ConversationMemory()

# Planner for multi-step tasks
from semantic_kernel.planners import BasicPlanner

planner = BasicPlanner()

async def plan_and_execute():
    ask = "Check the weather in Seattle and summarize what I should wear"
    
    plan = await planner.create_plan(ask, kernel)
    result = await plan.invoke(kernel)
    
    print(f"Plan result: {result}")

if __name__ == "__main__":
    asyncio.run(main())
    asyncio.run(plan_and_execute())
```

### C# Implementation

```csharp
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Connectors.OpenAI;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.SemanticKernel.Embeddings;

// Initialize kernel
var builder = Kernel.CreateBuilder();
builder.AddOpenAIChatCompletion("gpt-4", "your-api-key");
builder.AddOpenAITextEmbeddingGeneration("text-embedding-ada-002", "your-api-key");

var kernel = builder.Build();

// Custom plugin
public class WeatherPlugin
{
    [KernelFunction, Description("Get weather information")]
    public string GetWeather([Description("Location")] string location)
    {
        return $"Weather in {location}: Sunny, 72°F";
    }
}

// Add plugin
kernel.ImportPluginFromType<WeatherPlugin>();

// Create semantic function
var summarizeFunction = kernel.CreateFunctionFromPrompt(
    "{{$input}}\n\nSummarize the above in 2-3 sentences.",
    functionName: "Summarize"
);

// Execute functions
var weatherResult = await kernel.InvokeAsync("WeatherPlugin", "GetWeather", 
    new() { ["location"] = "Seattle" });

var summaryResult = await kernel.InvokeAsync(summarizeFunction, 
    new() { ["input"] = "Long text to summarize..." });

// Chat completion with memory
var chatService = kernel.GetRequiredService<IChatCompletionService>();
var chatHistory = new ChatHistory("You are a helpful assistant");

chatHistory.AddUserMessage("What's the weather like?");
var response = await chatService.GetChatMessageContentAsync(chatHistory);
chatHistory.Add(response);

Console.WriteLine(response.Content);
```

---

## AutoGen

### Python Implementation

```python
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
from autogen.coding import LocalCommandLineCodeExecutor
import tempfile
import os

# Configuration
config_list = [
    {
        "model": "gpt-4",
        "api_key": "your-api-key",
    }
]

# Basic two-agent conversation
assistant = AssistantAgent(
    name="assistant",
    llm_config={"config_list": config_list},
    system_message="You are a helpful AI assistant."
)

user_proxy = UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=3,
    code_execution_config={
        "executor": LocalCommandLineCodeExecutor(timeout=60, work_dir=tempfile.mkdtemp())
    }
)

# Start conversation
user_proxy.initiate_chat(
    assistant,
    message="Write a Python function to calculate fibonacci numbers and test it."
)

# Multi-agent group chat
planner = AssistantAgent(
    name="planner",
    llm_config={"config_list": config_list},
    system_message="""You are a planner. Create step-by-step plans for tasks."""
)

coder = AssistantAgent(
    name="coder", 
    llm_config={"config_list": config_list},
    system_message="""You are a programmer. Write clean, efficient code."""
)

tester = AssistantAgent(
    name="tester",
    llm_config={"config_list": config_list},
    system_message="""You are a tester. Create comprehensive tests for code."""
)

user_proxy_group = UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=3,
    code_execution_config={
        "executor": LocalCommandLineCodeExecutor(timeout=60, work_dir=tempfile.mkdtemp())
    }
)

# Group chat
group_chat = GroupChat(
    agents=[planner, coder, tester, user_proxy_group],
    messages=[],
    max_round=10
)

manager = GroupChatManager(
    groupchat=group_chat,
    llm_config={"config_list": config_list}
)

# Start group conversation
user_proxy_group.initiate_chat(
    manager,
    message="Create a web scraper for extracting product prices from an e-commerce site."
)

# Custom agent with tools
class DataAnalystAgent(AssistantAgent):
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        self.register_function(
            function_map={
                "analyze_data": self._analyze_data,
                "create_visualization": self._create_visualization,
            }
        )
    
    def _analyze_data(self, data_path: str) -> str:
        """Analyze data from a CSV file."""
        # Simplified analysis
        return f"Data analysis complete for {data_path}: Found trends and patterns."
    
    def _create_visualization(self, chart_type: str, data_columns: list) -> str:
        """Create a data visualization."""
        return f"Created {chart_type} chart with columns: {', '.join(data_columns)}"

data_analyst = DataAnalystAgent(
    name="data_analyst",
    llm_config={"config_list": config_list},
    system_message="You are a data analyst with access to analysis and visualization tools."
)

# Memory implementation with conversation history
class ConversationMemory:
    def __init__(self):
        self.history = []
    
    def add_message(self, sender: str, message: str):
        self.history.append({"sender": sender, "message": message, "timestamp": time.time()})
    
    def get_recent_context(self, n_messages: int = 10) -> str:
        recent = self.history[-n_messages:]
        return "\n".join([f"{msg['sender']}: {msg['message']}" for msg in recent])

memory = ConversationMemory()

# Agent with memory
class MemoryAgent(AssistantAgent):
    def __init__(self, name, memory, **kwargs):
        super().__init__(name, **kwargs)
        self.memory = memory
    
    def generate_reply(self, messages=None, sender=None, exclude=None):
        # Add context from memory
        context = self.memory.get_recent_context()
        if context:
            system_message = f"{self.system_message}\n\nRecent conversation context:\n{context}"
            # Update system message temporarily
            original_system_message = self.system_message
            self.system_message = system_message
            
        reply = super().generate_reply(messages, sender, exclude)
        
        # Store the exchange
        if messages:
            self.memory.add_message(sender.name if sender else "unknown", messages[-1]["content"])
        if reply:
            self.memory.add_message(self.name, reply)
            
        # Restore original system message
        if context:
            self.system_message = original_system_message
            
        return reply

memory_assistant = MemoryAgent(
    name="memory_assistant",
    memory=memory,
    llm_config={"config_list": config_list},
    system_message="You are an assistant with memory of previous conversations."
)
```

---

## CrewAI

### Python Implementation

```python
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain.tools import DuckDuckGoSearchRun
import os

# Initialize LLM
llm = ChatOpenAI(
    model="gpt-4",
    temperature=0.1
)

# Custom tools
class WebSearchTool(BaseTool):
    name: str = "web_search"
    description: str = "Search the web for information"
    
    def _run(self, query: str) -> str:
        search = DuckDuckGoSearchRun()
        return search.run(query)

class DataAnalysisTool(BaseTool):
    name: str = "data_analysis"
    description: str = "Analyze data and provide insights"
    
    def _run(self, data: str) -> str:
        return f"Analysis of {data}: Identified key trends and patterns."

# Define agents
researcher = Agent(
    role='Senior Research Analyst',
    goal='Uncover cutting-edge developments in AI and machine learning',
    backstory="""You work at a leading tech think tank.
    Your expertise lies in identifying emerging trends.
    You have a knack for dissecting complex data and presenting actionable insights.""",
    verbose=True,
    allow_delegation=False,
    tools=[WebSearchTool()],
    llm=llm
)

writer = Agent(
    role='Tech Content Strategist',
    goal='Craft compelling content on tech advancements',
    backstory="""You are a renowned Content Strategist, known for your insightful
    and engaging articles. You transform complex concepts into compelling narratives.""",
    verbose=True,
    allow_delegation=True,
    llm=llm
)

data_analyst = Agent(
    role='Data Analyst',
    goal='Analyze data to provide actionable insights',
    backstory="""You are an expert data analyst with years of experience in
    extracting meaningful insights from complex datasets.""",
    verbose=True,
    allow_delegation=False,
    tools=[DataAnalysisTool()],
    llm=llm
)

# Define tasks
research_task = Task(
    description="""Conduct a comprehensive analysis of the latest advancements
    in AI in 2024. Identify key trends, breakthrough technologies, and the
    top 5 AI companies leading the way.""",
    expected_output="A comprehensive 3 paragraph report on the latest AI advancements in 2024",
    agent=researcher
)

analysis_task = Task(
    description="""Using the research provided, analyze the market impact
    and future implications of these AI advancements.""",
    expected_output="A detailed analysis report with market implications",
    agent=data_analyst,
    context=[research_task]
)

write_task = Task(
    description="""Using the research and analysis provided, write a compelling
    blog post about the future of AI. Make it engaging and accessible.""",
    expected_output="A 4 paragraph blog post about AI advancements",
    agent=writer,
    context=[research_task, analysis_task]
)

# Assemble crew
crew = Crew(
    agents=[researcher, data_analyst, writer],
    tasks=[research_task, analysis_task, write_task],
    verbose=2,
    process=Process.sequential
)

# Execute the crew
result = crew.kickoff()
print(result)

# Hierarchical process with manager
manager = Agent(
    role='Project Manager',
    goal='Manage the team and ensure project success',
    backstory="""You're a seasoned project manager with a talent for getting
    the best out of your team and ensuring projects are delivered on time.""",
    verbose=True,
    allow_delegation=True,
    llm=llm
)

hierarchical_crew = Crew(
    agents=[researcher, writer, data_analyst],
    tasks=[research_task, write_task],
    verbose=2,
    process=Process.hierarchical,
    manager_agent=manager
)

# Memory implementation
class CrewMemory:
    def __init__(self):
        self.task_history = []
        self.agent_interactions = []
    
    def log_task_completion(self, task_name: str, result: str):
        self.task_history.append({
            "task": task_name,
            "result": result,
            "timestamp": datetime.now()
        })
    
    def log_agent_interaction(self, agent_name: str, action: str, result: str):
        self.agent_interactions.append({
            "agent": agent_name,
            "action": action,
            "result": result,
            "timestamp": datetime.now()
        })
    
    def get_context(self) -> str:
        context = "Previous tasks:\n"
        for task in self.task_history[-3:]:  # Last 3 tasks
            context += f"- {task['task']}: {task['result'][:100]}...\n"
        return context

crew_memory = CrewMemory()

# Agent with memory
class MemoryAwareAgent(Agent):
    def __init__(self, memory, **kwargs):
        super().__init__(**kwargs)
        self.memory = memory
    
    def execute_task(self, task):
        # Add memory context to task
        context = self.memory.get_context()
        enhanced_description = f"{task.description}\n\nContext from previous work:\n{context}"
        
        # Execute task with enhanced context
        result = super().execute_task(task)
        
        # Log the interaction
        self.memory.log_task_completion(task.description, str(result))
        
        return result

# Custom callback for monitoring
class CrewCallback:
    def on_task_start(self, task, agent):
        print(f"Task '{task.description[:50]}...' started by {agent.role}")
    
    def on_task_complete(self, task, agent, result):
        print(f"Task completed by {agent.role}. Result: {str(result)[:100]}...")

callback = CrewCallback()

# Crew with callbacks and memory
crew_with_memory = Crew(
    agents=[
        MemoryAwareAgent(memory=crew_memory, **researcher.dict()),
        MemoryAwareAgent(memory=crew_memory, **writer.dict())
    ],
    tasks=[research_task, write_task],
    verbose=2,
    process=Process.sequential,
    callbacks=[callback]
)
```

---

## Hugging Face Transformers

### Python Implementation

```python
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForCausalLM, 
    pipeline, Trainer, TrainingArguments
)
from sentence_transformers import SentenceTransformer
import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Basic LLM Usage
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Add padding token
tokenizer.pad_token = tokenizer.eos_token

def generate_response(input_text, chat_history_ids=None):
    # Encode input
    new_user_input_ids = tokenizer.encode(
        input_text + tokenizer.eos_token, 
        return_tensors='pt'
    )
    
    # Append to chat history
    if chat_history_ids is not None:
        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)
    else:
        bot_input_ids = new_user_input_ids
    
    # Generate response
    chat_history_ids = model.generate(
        bot_input_ids, 
        max_length=1000,
        num_beams=5,
        no_repeat_ngram_size=3,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Decode response
    response = tokenizer.decode(
        chat_history_ids[:, bot_input_ids.shape[-1]:][0], 
        skip_special_tokens=True
    )
    
    return response, chat_history_ids

# Chat conversation with memory
chat_history = None
while True:
    user_input = input("You: ")
    if user_input.lower() == 'quit':
        break
    
    response, chat_history = generate_response(user_input, chat_history)
    print(f"Bot: {response}")

# Embeddings with Sentence Transformers
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

documents = [
    "Python is a programming language",
    "Machine learning uses algorithms",
    "Neural networks mimic the brain"
]

# Generate embeddings
embeddings = embedding_model.encode(documents)

# Similarity search
query = "What is Python?"
query_embedding = embedding_model.encode([query])

similarities = cosine_similarity(query_embedding, embeddings)
best_match_idx = np.argmax(similarities)

print(f"Query: {query}")
print(f"Best match: {documents[best_match_idx]}")
print(f"Similarity: {similarities[0][best_match_idx]:.4f}")

# Vector Database Implementation
class SimpleVectorDB:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.documents = []
        self.embeddings = None
    
    def add_documents(self, docs):
        self.documents.extend(docs)
        new_embeddings = self.embedding_model.encode(docs)
        
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
    
    def search(self, query, top_k=3):
        query_embedding = self.embedding_model.encode([query])
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                'document': self.documents[idx],
                'score': similarities[idx]
            })
        
        return results

# Initialize vector database
vector_db = SimpleVectorDB(embedding_model)
vector_db.add_documents(documents)

# Search
results = vector_db.search("programming language")
for result in results:
    print(f"Document: {result['document']}")
    print(f"Score: {result['score']:.4f}")
    print("---")

# Pipeline Usage
# Text Generation
generator = pipeline("text-generation", model="gpt2")
generated = generator("The future of AI is", max_length=100, num_return_sequences=2)
for text in generated:
    print(text['generated_text'])

# Question Answering
qa_pipeline = pipeline("question-answering")
context = "Python is a high-level programming language known for its simplicity."
question = "What is Python known for?"
answer = qa_pipeline(question=question, context=context)
print(f"Answer: {answer['answer']}")

# Sentiment Analysis
sentiment_pipeline = pipeline("sentiment-analysis")
sentiment = sentiment_pipeline("I love using transformers!")
print(f"Sentiment: {sentiment[0]['label']} ({sentiment[0]['score']:.4f})")

# Custom Dataset for Fine-tuning
class ChatDataset(Dataset):
    def __init__(self, conversations, tokenizer, max_length=512):
        self.conversations = conversations
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.conversations)
    
    def __getitem__(self, idx):
        conversation = self.conversations[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            conversation,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': encoding['input_ids'].flatten()
        }

# Fine-tuning setup
conversations = [
    "Hello! How can I help you today?",
    "I'm looking for information about machine learning.",
    "Machine learning is a subset of AI that enables computers to learn."
]

dataset = ChatDataset(conversations, tokenizer)

training_args = TrainingArguments(
    output_dir="./fine-tuned-model",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

# Train model (uncomment to run)
# trainer.train()

# Custom Tool Integration
class HuggingFaceToolkit:
    def __init__(self):
        self.qa_pipeline = pipeline("question-answering")
        self.summarizer = pipeline("summarization")
        self.sentiment = pipeline("sentiment-analysis")
        self.embeddings = SentenceTransformer('all-MiniLM-L6-v2')
    
    def answer_question(self, question: str, context: str) -> str:
        result = self.qa_pipeline(question=question, context=context)
        return result['answer']
    
    def summarize_text(self, text: str) -> str:
        summary = self.summarizer(text, max_length=150, min_length=30)
        return summary[0]['summary_text']
    
    def analyze_sentiment(self, text: str) -> dict:
        result = self.sentiment(text)[0]
        return {"label": result['label'], "confidence": result['score']}
    
    def get_embeddings(self, texts: list) -> np.ndarray:
        return self.embeddings.encode(texts)
    
    def semantic_search(self, query: str, documents: list, top_k: int = 3) -> list:
        doc_embeddings = self.embeddings.encode(documents)
        query_embedding = self.embeddings.encode([query])
        
        similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                'document': documents[idx],
                'score': float(similarities[idx])
            })
        
        return results

toolkit = HuggingFaceToolkit()

# Example usage
context = "Transformers are a type of neural network architecture."
question = "What are transformers?"
answer = toolkit.answer_question(question, context)
print(f"Answer: {answer}")

# Memory implementation for conversations
class ConversationMemory:
    def __init__(self, max_history=10):
        self.history = []
        self.max_history = max_history
        self.embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def add_exchange(self, user_input: str, bot_response: str):
        exchange = {
            'user': user_input,
            'bot': bot_response,
            'timestamp': torch.tensor(len(self.history))
        }
        self.history.append(exchange)
        
        # Keep only recent history
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
    
    def get_relevant_context(self, current_input: str, top_k: int = 3) -> str:
        if not self.history:
            return ""
        
        # Get embeddings for current input and historical exchanges
        current_embedding = self.embeddings_model.encode([current_input])
        
        historical_texts = [f"{ex['user']} {ex['bot']}" for ex in self.history]
        historical_embeddings = self.embeddings_model.encode(historical_texts)
        
        # Find most relevant exchanges
        similarities = cosine_similarity(current_embedding, historical_embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        context = ""
        for idx in top_indices:
            exchange = self.history[idx]
            context += f"Previous: User: {exchange['user']} Bot: {exchange['bot']}\n"
        
        return context

memory = ConversationMemory()
```

### TypeScript Implementation

```typescript
import { pipeline, env } from '@xenova/transformers';

// Disable local models for web deployment
env.allowLocalModels = false;

class HuggingFaceToolkit {
  private generator: any;
  private classifier: any;
  private qaModel: any;
  
  async initialize() {
    // Initialize models
    this.generator = await pipeline('text-generation', 'Xenova/gpt2');
    this.classifier = await pipeline('sentiment-analysis', 'Xenova/distilbert-base-uncased-finetuned-sst-2-english');
    this.qaModel = await pipeline('question-answering', 'Xenova/distilbert-base-cased-distilled-squad');
  }
  
  async generateText(prompt: string, maxLength: number = 100): Promise<string> {
    const result = await this.generator(prompt, {
      max_new_tokens: maxLength,
      temperature: 0.7,
      do_sample: true,
    });
    return result[0].generated_text;
  }
  
  async analyzeSentiment(text: string): Promise<{label: string, score: number}> {
    const result = await this.classifier(text);
    return {
      label: result[0].label,
      score: result[0].score
    };
  }
  
  async answerQuestion(question: string, context: string): Promise<string> {
    const result = await this.qaModel(question, context);
    return result.answer;
  }
}

// Chat implementation with memory
class ChatBot {
  private toolkit: HuggingFaceToolkit;
  private conversationHistory: Array<{user: string, bot: string}> = [];
  
  constructor(toolkit: HuggingFaceToolkit) {
    this.toolkit = toolkit;
  }
  
  async chat(userInput: string): Promise<string> {
    // Get context from previous conversations
    const context = this.getContext();
    
    // Generate response
    const prompt = `${context}\nUser: ${userInput}\nBot:`;
    const response = await this.toolkit.generateText(prompt, 50);
    
    // Extract bot response (remove the prompt part)
    const botResponse = response.split('Bot:').pop()?.trim() || 'I understand.';
    
    // Store in memory
    this.conversationHistory.push({
      user: userInput,
      bot: botResponse
    });
    
    // Keep only recent history
    if (this.conversationHistory.length > 5) {
      this.conversationHistory = this.conversationHistory.slice(-5);
    }
    
    return botResponse;
  }
  
  private getContext(): string {
    return this.conversationHistory
      .map(exchange => `User: ${exchange.user}\nBot: ${exchange.bot}`)
      .join('\n');
  }
}

// Vector database implementation
class SimpleVectorDB {
  private documents: string[] = [];
  private embeddings: number[][] = [];
  
  addDocument(text: string, embedding: number[]) {
    this.documents.push(text);
    this.embeddings.push(embedding);
  }
  
  search(queryEmbedding: number[], topK: number = 3): Array<{document: string, score: number}> {
    const similarities = this.embeddings.map((embedding, index) => ({
      index,
      score: this.cosineSimilarity(queryEmbedding, embedding)
    }));
    
    similarities.sort((a, b) => b.score - a.score);
    
    return similarities.slice(0, topK).map(result => ({
      document: this.documents[result.index],
      score: result.score
    }));
  }
  
  private cosineSimilarity(a: number[], b: number[]): number {
    const dotProduct = a.reduce((sum, val, i) => sum + val * b[i], 0);
    const magnitudeA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0));
    const magnitudeB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0));
    return dotProduct / (magnitudeA * magnitudeB);
  }
}

// Usage example
async function main() {
  const toolkit = new HuggingFaceToolkit();
  await toolkit.initialize();
  
  const chatBot = new ChatBot(toolkit);
  
  // Example conversation
  const response1 = await chatBot.chat("Hello, how are you?");
  console.log("Bot:", response1);
  
  const response2 = await chatBot.chat("What did I just ask you?");
  console.log("Bot:", response2);
  
  // Sentiment analysis
  const sentiment = await toolkit.analyzeSentiment("I love this new AI model!");
  console.log("Sentiment:", sentiment);
  
  // Question answering
  const answer = await toolkit.answerQuestion(
    "What is TypeScript?",
    "TypeScript is a strongly typed programming language that builds on JavaScript."
  );
  console.log("Answer:", answer);
}

main().catch(console.error);
```

---

## Rasa

### Python Implementation

```python
# domain.yml
"""
version: "3.1"

intents:
  - greet
  - goodbye
  - ask_weather
  - ask_time
  - ask_name
  - provide_name

entities:
  - location
  - person_name

slots:
  user_name:
    type: text
    mappings:
      - type: from_entity
        entity: person_name
  location:
    type: text
    mappings:
      - type: from_entity
        entity: location

responses:
  utter_greet:
    - text: "Hello! How can I help you today?"
    - text: "Hi there! What can I do for you?"

  utter_goodbye:
    - text: "Goodbye! Have a great day!"
    - text: "See you later!"

  utter_ask_name:
    - text: "What's your name?"

actions:
  - action_get_weather
  - action_get_time
  - action_remember_name
"""

# Custom Actions
from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet
import requests
from datetime import datetime

class ActionGetWeather(Action):
    def name(self) -> Text:
        return "action_get_weather"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        location = tracker.get_slot("location")
        
        if not location:
            dispatcher.utter_message(text="Please provide a location for weather information.")
            return []
        
        # Simplified weather API call
        weather_info = self.get_weather(location)
        
        dispatcher.utter_message(text=f"The weather in {location} is {weather_info}")
        
        return []
    
    def get_weather(self, location: str) -> str:
        # Simplified weather service
        return f"sunny and 75°F"

class ActionGetTime(Action):
    def name(self) -> Text:
        return "action_get_time"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        current_time = datetime.now().strftime("%H:%M")
        dispatcher.utter_message(text=f"The current time is {current_time}")
        
        return []

class ActionRememberName(Action):
    def name(self) -> Text:
        return "action_remember_name"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        user_name = tracker.get_slot("user_name")
        
        if user_name:
            dispatcher.utter_message(text=f"Nice to meet you, {user_name}! I'll remember your name.")
            return [SlotSet("user_name", user_name)]
        else:
            dispatcher.utter_message(text="I didn't catch your name. Could you tell me again?")
            return []

# Custom NLU Components
from rasa.engine.graph import ExecutionContext
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.nlu.training_data.message import Message
from rasa.nlu.components.dense_featurizer import DenseFeaturizer
from rasa.nlu.featurizers._sklearn_featurizer import SklearnFeaturizer
import numpy as np
from sentence_transformers import SentenceTransformer

@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.MESSAGE_FEATURIZER, is_trainable=False
)
class SentenceTransformerFeaturizer(DenseFeaturizer):
    
    def __init__(self, config: Dict[Text, Any]) -> None:
        super().__init__(config)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def process(self, messages: List[Message]) -> List[Message]:
        for message in messages:
            text = message.get("text")
            if text:
                embedding = self.model.encode([text])[0]
                features = np.array([embedding])
                message.set("dense_features", features)
        
        return messages

# Memory Implementation
class ConversationMemory:
    def __init__(self):
        self.sessions = {}
    
    def get_session_data(self, session_id: str) -> Dict:
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                'conversation_history': [],
                'user_preferences': {},
                'context': {}
            }
        return self.sessions[session_id]
    
    def add_message(self, session_id: str, message: str, intent: str, entities: List[Dict]):
        session_data = self.get_session_data(session_id)
        session_data['conversation_history'].append({
            'message': message,
            'intent': intent,
            'entities': entities,
            'timestamp': datetime.now()
        })
    
    def update_preferences(self, session_id: str, preferences: Dict):
        session_data = self.get_session_data(session_id)
        session_data['user_preferences'].update(preferences)

# Custom Action with Memory
class ActionWithMemory(Action):
    def __init__(self):
        self.memory = ConversationMemory()
    
    def name(self) -> Text:
        return "action_with_memory"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        session_id = tracker.sender_id
        latest_message = tracker.latest_message
        
        # Store conversation in memory
        self.memory.add_message(
            session_id,
            latest_message.get('text', ''),
            latest_message.get('intent', {}).get('name', ''),
            latest_message.get('entities', [])
        )
        
        # Get conversation context
        session_data = self.memory.get_session_data(session_id)
        history_count = len(session_data['conversation_history'])
        
        dispatcher.utter_message(
            text=f"I remember our conversation. We've exchanged {history_count} messages."
        )
        
        return []

# Training Data Generator
class TrainingDataGenerator:
    def __init__(self):
        self.intents = {
            'greet': [
                "hello", "hi", "hey", "good morning", "good afternoon",
                "how are you", "what's up", "greetings"
            ],
            'goodbye': [
                "bye", "goodbye", "see you later", "farewell", "take care",
                "until next time", "catch you later"
            ],
            'ask_weather': [
                "what's the weather like", "how's the weather", "weather forecast",
                "is it raining", "will it be sunny", "weather in [location]"
            ],
            'ask_time': [
                "what time is it", "current time", "tell me the time",
                "what's the time", "time please"
            ]
        }
    
    def generate_nlu_data(self) -> str:
        nlu_data = "version: \"3.1\"\nnlu:\n"
        
        for intent, examples in self.intents.items():
            nlu_data += f"- intent: {intent}\n  examples: |\n"
            for example in examples:
                nlu_data += f"    - {example}\n"
            nlu_data += "\n"
        
        return nlu_data
    
    def generate_stories(self) -> str:
        stories = """
version: "3.1"

stories:
- story: greet and ask weather
  steps:
  - intent: greet
  - action: utter_greet
  - intent: ask_weather
  - action: action_get_weather

- story: greet and goodbye
  steps:
  - intent: greet
  - action: utter_greet
  - intent: goodbye
  - action: utter_goodbye

- story: ask time
  steps:
  - intent: ask_time
  - action: action_get_time
"""
        return stories

# Configuration
config_yml = """
language: en

pipeline:
  - name: WhitespaceTokenizer
  - name: RegexFeaturizer
  - name: LexicalSyntacticFeaturizer
  - name: CountVectorsFeaturizer
  - name: CountVectorsFeaturizer
    analyzer: char_wb
    min_ngram: 1
    max_ngram: 4
  - name: DIETClassifier
    epochs: 100
  - name: EntitySynonymMapper
  - name: ResponseSelector
    epochs: 100

policies:
  - name: MemoizationPolicy
  - name: TEDPolicy
    max_history: 5
    epochs: 100
  - name: RulePolicy
"""

# Vector Store Integration
class RasaVectorStore:
    def __init__(self):
        self.embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.documents = []
        self.embeddings = []
    
    def add_documents(self, docs: List[str]):
        embeddings = self.embeddings_model.encode(docs)
        self.documents.extend(docs)
        self.embeddings.extend(embeddings)
    
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        query_embedding = self.embeddings_model.encode([query])[0]
        
        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            similarity = np.dot(query_embedding, doc_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
            )
            similarities.append({'document': self.documents[i], 'score': similarity, 'index': i})
        
        similarities.sort(key=lambda x: x['score'], reverse=True)
        return similarities[:top_k]

# RAG Action
class ActionRAG(Action):
    def __init__(self):
        self.vector_store = RasaVectorStore()
        # Add some sample documents
        docs = [
            "Python is a programming language",
            "Machine learning is a subset of AI",
            "Rasa is an open source conversational AI framework"
        ]
        self.vector_store.add_documents(docs)
    
    def name(self) -> Text:
        return "action_rag"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        user_message = tracker.latest_message.get('text', '')
        
        # Search for relevant documents
        results = self.vector_store.search(user_message)
        
        if results and results[0]['score'] > 0.5:
            response = f"Based on my knowledge: {results[0]['document']}"
        else:
            response = "I don't have specific information about that topic."
        
        dispatcher.utter_message(text=response)
        
        return []
```

---

## MCP (Model Context Protocol) Implementation

### MCP Server Implementation (Python)

```python
import asyncio
import json
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import sqlite3

@dataclass
class Resource:
    uri: str
    name: str
    description: str
    mime_type: str

@dataclass
class Tool:
    name: str
    description: str
    input_schema: Dict[str, Any]

class MCPServer:
    def __init__(self, name: str, version: str):
        self.name = name
        self.version = version
        self.tools: Dict[str, Tool] = {}
        self.resources: Dict[str, Resource] = {}
        self.db = sqlite3.connect('mcp_data.db')
        self._init_db()
    
    def _init_db(self):
        """Initialize database for storing data"""
        cursor = self.db.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY,
                title TEXT,
                content TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY,
                document_id INTEGER,
                embedding BLOB,
                FOREIGN KEY (document_id) REFERENCES documents (id)
            )
        ''')
        self.db.commit()
    
    def register_tool(self, tool: Tool, handler):
        """Register a tool with its handler"""
        self.tools[tool.name] = tool
        setattr(self, f"handle_{tool.name}", handler)
    
    def register_resource(self, resource: Resource, handler):
        """Register a resource with its handler"""
        self.resources[resource.uri] = resource
        setattr(self, f"get_resource_{resource.uri.replace('/', '_')}", handler)
    
    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Main request handler"""
        method = request.get('method')
        params = request.get('params', {})
        
        if method == 'initialize':
            return await self.handle_initialize(params)
        elif method == 'tools/list':
            return await self.handle_list_tools()
        elif method == 'tools/call':
            return await self.handle_call_tool(params)
        elif method == 'resources/list':
            return await self.handle_list_resources()
        elif method == 'resources/read':
            return await self.handle_read_resource(params)
        else:
            return {'error': {'code': -32601, 'message': 'Method not found'}}
    
    async def handle_initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle initialization request"""
        return {
            'result': {
                'serverInfo': {
                    'name': self.name,
                    'version': self.version
                },
                'capabilities': {
                    'tools': {},
                    'resources': {}
                }
            }
        }
    
    async def handle_list_tools(self) -> Dict[str, Any]:
        """List available tools"""
        tools_list = []
        for tool in self.tools.values():
            tools_list.append({
                'name': tool.name,
                'description': tool.description,
                'inputSchema': tool.input_schema
            })
        
        return {'result': {'tools': tools_list}}
    
    async def handle_call_tool(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tool execution"""
        tool_name = params.get('name')
        arguments = params.get('arguments', {})
        
        if tool_name not in self.tools:
            return {'error': {'code': -32602, 'message': 'Tool not found'}}
        
        handler = getattr(self, f"handle_{tool_name}", None)
        if not handler:
            return {'error': {'code': -32603, 'message': 'Tool handler not found'}}
        
        try:
            result = await handler(arguments)
            return {'result': {'content': [{'type': 'text', 'text': str(result)}]}}
        except Exception as e