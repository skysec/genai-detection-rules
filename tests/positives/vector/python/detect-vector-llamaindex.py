#!/usr/bin/env python3
"""
Positive test cases for LlamaIndex vector/retrieval detection.
These patterns should be detected by the detect-vector-llamaindex.yaml rule.
"""

# Core vector index
from llama_index.core import VectorStoreIndex
index = VectorStoreIndex.from_documents(documents)
vector_index = VectorStoreIndex.from_documents(docs, service_context=service_context)

# Vector store imports
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.vector_stores import PineconeVectorStore
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.vector_stores.weaviate import WeaviateVectorStore

# Vector store creation
vector_store = ChromaVectorStore(chroma_collection=collection)
pinecone_store = PineconeVectorStore(pinecone_index=index)

# Storage context with vector store
from llama_index.core.storage.storage_context import StorageContext
storage_context = StorageContext.from_defaults(vector_store=vector_store)
context = StorageContext.from_defaults(vector_store=chroma_store)

# Index with storage context
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
vector_idx = VectorStoreIndex.from_documents(docs, storage_context=context)

# Query engine (RAG)
query_engine = index.as_query_engine()
engine = vector_index.as_query_engine()
response = query_engine.query("What is machine learning?")
answer = engine.query("Tell me about AI")

# Retriever patterns
from llama_index.core.retrievers import BaseRetriever
retriever = index.as_retriever(similarity_top_k=5)
base_retriever = vector_index.as_retriever(top_k=10)

# Custom retriever implementation
class CustomRetriever(BaseRetriever):
    def __init__(self, vector_index):
        self._vector_index = vector_index
        super().__init__()

def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
    nodes = self._vector_index.as_retriever().retrieve(query_bundle)
    return nodes

# Query bundle and retrieval
from llama_index.core.schema import QueryBundle
from llama_index.core.schema import NodeWithScore
nodes = retriever.retrieve(query_bundle)

# Directory reader for documents
from llama_index.core import SimpleDirectoryReader
documents = SimpleDirectoryReader("./data").load_data()
reader = SimpleDirectoryReader("./docs")

# Query engine creation patterns
from llama_index.core.query_engine import RetrieverQueryEngine
query_engine = RetrieverQueryEngine.from_args(retriever)
engine = RetrieverQueryEngine.from_args(custom_retriever)

# Vector search patterns - HIGH CONFIDENCE
filtered_nodes = [node for node in nodes if node.score > 0.5]
high_score_nodes = [node for node in nodes if node.score > threshold]

# External vector database integration with LlamaIndex - HIGH CONFIDENCE
chroma_store = ChromaVectorStore(chroma_collection=collection)
pinecone_vector_store = PineconeVectorStore(index_name="test")
weaviate_store = WeaviateVectorStore(weaviate_client=client)

# Real-world LlamaIndex usage
def setup_llamaindex_rag():
    from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
    from llama_index.core.storage.storage_context import StorageContext
    from llama_index.vector_stores.chroma import ChromaVectorStore
    import chromadb

    # Load documents
    documents = SimpleDirectoryReader("./data").load_data()

    # Create vector store
    chroma_client = chromadb.EphemeralClient()
    chroma_collection = chroma_client.create_collection("quickstart")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # Create storage context
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Create index
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context
    )

    # Create query engine
    query_engine = index.as_query_engine()

    return query_engine

# Multiple vector stores
def multi_vectorstore_llamaindex():
    from llama_index.vector_stores.chroma import ChromaVectorStore
    from llama_index.vector_stores.pinecone import PineconeVectorStore
    from llama_index.vector_stores.weaviate import WeaviateVectorStore

    # Different vector stores
    chroma_store = ChromaVectorStore(chroma_collection=chroma_collection)
    pinecone_store = PineconeVectorStore(pinecone_index=pinecone_index)
    weaviate_store = WeaviateVectorStore(weaviate_client=weaviate_client)

    # Create indices with different stores
    chroma_context = StorageContext.from_defaults(vector_store=chroma_store)
    pinecone_context = StorageContext.from_defaults(vector_store=pinecone_store)

    chroma_index = VectorStoreIndex.from_documents(docs, storage_context=chroma_context)
    pinecone_index = VectorStoreIndex.from_documents(docs, storage_context=pinecone_context)

    return chroma_index, pinecone_index

# Advanced retrieval patterns
class AdvancedLlamaIndexRetriever:
    def __init__(self, index):
        self.index = index
        self.retriever = index.as_retriever(similarity_top_k=10)

    def hybrid_retrieve(self, query):
        # Get initial results
        nodes = self.retriever.retrieve(query)

        # Filter by score
        high_quality_nodes = [node for node in nodes if node.score > 0.7]

        return high_quality_nodes

    def get_query_engine(self):
        return self.index.as_query_engine()

# Document processing with LlamaIndex
def process_documents_llamaindex():
    from llama_index.core import SimpleDirectoryReader, Document

    # Load documents
    reader = SimpleDirectoryReader("./documents")
    documents = reader.load_data()

    # Create custom documents
    custom_docs = [
        Document(text="Custom document 1"),
        Document(text="Custom document 2")
    ]

    # Combine documents
    all_docs = documents + custom_docs

    # Create index
    index = VectorStoreIndex.from_documents(all_docs)

    return index

# Query engines with different configurations
def configure_query_engines():
    # Basic query engine
    basic_engine = index.as_query_engine()

    # Retriever with custom top-k
    custom_retriever = index.as_retriever(similarity_top_k=3)
    custom_engine = RetrieverQueryEngine.from_args(custom_retriever)

    # Query with different parameters
    response1 = basic_engine.query("What is AI?")
    response2 = custom_engine.query("Explain machine learning")

    return response1, response2

# Embedding and storage patterns
def setup_embeddings():
    from llama_index.core import ServiceContext
    from llama_index.embeddings.openai import OpenAIEmbedding

    # Custom embedding model
    embed_model = OpenAIEmbedding()
    service_context = ServiceContext.from_defaults(embed_model=embed_model)

    # Create index with custom embeddings
    index = VectorStoreIndex.from_documents(
        documents,
        service_context=service_context
    )

    return index

# Memory and chat patterns
def setup_chat_engine():
    from llama_index.core.memory import ChatMemoryBuffer

    # Create chat memory
    memory = ChatMemoryBuffer.from_defaults(token_limit=3000)

    # Create chat engine
    chat_engine = index.as_chat_engine(
        chat_mode="condense_question",
        memory=memory
    )

    # Chat interaction
    response = chat_engine.chat("Hello, what can you tell me about AI?")

    return response

# Async patterns
async def async_llamaindex():
    # Async query
    response = await query_engine.aquery("What is the future of AI?")

    # Async retrieval
    nodes = await retriever.aretrieve("machine learning concepts")

    return response, nodes

# Evaluation and metrics
def evaluate_llamaindex():
    from llama_index.core.evaluation import RelevancyEvaluator

    # Create evaluator
    evaluator = RelevancyEvaluator()

    # Evaluate query
    eval_result = evaluator.evaluate_response(
        query="What is AI?",
        response=response
    )

    return eval_result

# Complex RAG pipeline
class LlamaIndexRAGPipeline:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.index = None
        self.query_engine = None

    def setup(self):
        # Load documents
        documents = SimpleDirectoryReader(self.data_dir).load_data()

        # Create vector store
        vector_store = ChromaVectorStore(chroma_collection=collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Create index
        self.index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context
        )

        # Create query engine
        self.query_engine = self.index.as_query_engine()

    def query(self, question):
        return self.query_engine.query(question)

    def get_retriever(self, top_k=5):
        return self.index.as_retriever(similarity_top_k=top_k)