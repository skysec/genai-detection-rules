#!/usr/bin/env python3
"""
Positive test cases for LlamaIndex embeddings detection.
These patterns should be detected by the detect-embeddings-llamaindex.yaml rule.
"""

# Core LlamaIndex embedding imports - VERY HIGH CONFIDENCE
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings import OpenAIEmbedding as EmbeddingModel
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.embeddings import BaseEmbedding

# Additional embedding imports
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.embeddings.mistralai import MistralAIEmbedding

# Settings and core imports - HIGH CONFIDENCE
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex
from llama_index.core.storage.storage_context import StorageContext

# Vector store imports - HIGH CONFIDENCE
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.vector_stores.weaviate import WeaviateVectorStore

# Embedding instantiation - VERY HIGH CONFIDENCE
embed_model = OpenAIEmbedding()
openai_embedding = OpenAIEmbedding(model="text-embedding-ada-002")
hf_embedding = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Settings configuration with embeddings - HIGH CONFIDENCE
Settings.embed_model = OpenAIEmbedding()
Settings.embed_model = HuggingFaceEmbedding()
Settings.embed_model = embed_model

# Explicit embedding model usage - VERY HIGH CONFIDENCE
text_embedding = embed_model.get_text_embedding("Hello world")
query_embedding = embed_model.get_query_embedding("What is machine learning?")
embeddings = openai_embedding.get_text_embedding("Sample text")
query_emb = hf_embedding.get_query_embedding("Search query")

# Vector index creation with embeddings - HIGH CONFIDENCE
index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
vector_index = VectorStoreIndex.from_documents(docs, embed_model=openai_embedding)
service_index = VectorStoreIndex.from_documents(documents, service_context=service_context)

# Storage context with vector store - HIGH CONFIDENCE
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

# Vector store instantiation - HIGH CONFIDENCE
chroma_store = ChromaVectorStore(chroma_collection=collection)
pinecone_store = PineconeVectorStore(pinecone_index=index)
weaviate_store = WeaviateVectorStore(weaviate_client=client)

# Real-world usage examples
def setup_llamaindex_embeddings():
    from llama_index.core import Document

    # Initialize embedding model
    embed_model = OpenAIEmbedding(
        model="text-embedding-ada-002",
        api_key="sk-..."
    )

    # Configure settings
    Settings.embed_model = embed_model

    # Create documents
    documents = [
        Document(text="LlamaIndex is a framework for building LLM applications"),
        Document(text="Embeddings enable semantic search capabilities"),
        Document(text="Vector stores provide efficient similarity search")
    ]

    # Create vector index
    index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)

    return index

def huggingface_llamaindex_example():
    # HuggingFace embeddings via LlamaIndex
    hf_embed = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-mpnet-base-v2",
        device="cuda"
    )

    # Set as default embedding model
    Settings.embed_model = hf_embed

    # Generate embeddings
    text = "Machine learning and artificial intelligence"
    text_embedding = hf_embed.get_text_embedding(text)

    # Query embedding
    query = "What is AI?"
    query_embedding = hf_embed.get_query_embedding(query)

    return text_embedding, query_embedding

def vector_store_with_embeddings():
    import chromadb
    from llama_index.core import Document

    # Initialize Chroma client
    chroma_client = chromadb.Client()
    chroma_collection = chroma_client.create_collection("llamaindex_demo")

    # Create vector store
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # Storage context
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Embedding model
    embed_model = OpenAIEmbedding()

    # Create documents and index
    documents = [Document(text="Sample document for embedding")]
    index = VectorStoreIndex.from_documents(
        documents=documents,
        storage_context=storage_context,
        embed_model=embed_model
    )

    return index

# Advanced patterns with multiple embedding models
class LlamaIndexEmbeddingService:
    def __init__(self):
        self.openai_embed = OpenAIEmbedding()
        self.hf_embed = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

    def get_openai_embedding(self, text):
        return self.openai_embed.get_text_embedding(text)

    def get_hf_embedding(self, text):
        return self.hf_embed.get_text_embedding(text)

    def setup_index(self, documents, embedding_type="openai"):
        if embedding_type == "openai":
            embed_model = self.openai_embed
        else:
            embed_model = self.hf_embed

        return VectorStoreIndex.from_documents(
            documents=documents,
            embed_model=embed_model
        )

# Pinecone integration with embeddings
def pinecone_llamaindex_setup():
    import pinecone

    # Initialize Pinecone
    pinecone.init(api_key="your-api-key", environment="us-west1-gcp")

    # Create Pinecone index
    pinecone_index = pinecone.Index("llamaindex-demo")

    # Create vector store
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

    # Storage context
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Embedding model
    embed_model = OpenAIEmbedding()

    # Create index
    index = VectorStoreIndex.from_documents(
        documents=documents,
        storage_context=storage_context,
        embed_model=embed_model
    )

    return index

# Multiple embedding providers
def multi_embedding_setup():
    from llama_index.core import Document

    # Different embedding models
    embedding_models = {
        'openai': OpenAIEmbedding(),
        'huggingface': HuggingFaceEmbedding(),
        'cohere': CohereEmbedding(),
        'azure': AzureOpenAIEmbedding()
    }

    documents = [Document(text="Sample document")]
    indices = {}

    for name, embed_model in embedding_models.items():
        # Set embedding model in settings
        Settings.embed_model = embed_model

        # Create index with specific embedding model
        index = VectorStoreIndex.from_documents(
            documents=documents,
            embed_model=embed_model
        )

        indices[name] = index

    return indices

# Custom embedding model configuration
custom_openai_embed = OpenAIEmbedding(
    model="text-embedding-3-large",
    dimensions=1024,
    api_key="sk-custom"
)

custom_hf_embed = HuggingFaceEmbedding(
    model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    device="cuda:0",
    normalize=True
)

# Embedding usage in query engines
def create_query_engine():
    from llama_index.core import Document

    # Setup embedding model
    embed_model = OpenAIEmbedding()
    Settings.embed_model = embed_model

    # Create documents
    documents = [
        Document(text="LlamaIndex supports various embedding models"),
        Document(text="Vector databases enable efficient similarity search")
    ]

    # Create index
    index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)

    # Create query engine
    query_engine = index.as_query_engine()

    return query_engine

# Batch embedding processing
def batch_embedding_example():
    from llama_index.core import Document

    # Large batch of documents
    documents = [
        Document(text=f"Document {i} about LlamaIndex and embeddings")
        for i in range(50)
    ]

    # Embedding model
    embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Batch process embeddings
    text_embeddings = []
    for doc in documents:
        embedding = embed_model.get_text_embedding(doc.text)
        text_embeddings.append(embedding)

    # Create index with all documents
    index = VectorStoreIndex.from_documents(
        documents=documents,
        embed_model=embed_model
    )

    return index, text_embeddings

# Direct embedding model methods
direct_embedding = embed_model.get_text_embedding("Direct embedding call")
direct_query = embed_model.get_query_embedding("Direct query embedding")

# Service context patterns (legacy but still detected)
from llama_index.core import ServiceContext

service_context = ServiceContext.from_defaults(embed_model=embed_model)
legacy_index = VectorStoreIndex.from_documents(documents, service_context=service_context)