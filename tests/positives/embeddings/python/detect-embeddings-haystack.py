#!/usr/bin/env python3
"""
Positive test cases for Haystack embeddings detection.
These patterns should be detected by the detect-embeddings-haystack.yaml rule.
"""

# Core Haystack embedding component imports - VERY HIGH CONFIDENCE
from haystack.components.embedders import OpenAITextEmbedder
from haystack.components.embedders import OpenAIDocumentEmbedder
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.embedders import SentenceTransformersDocumentEmbedder

# Additional embedder imports
from haystack.components.embedders import CohereTextEmbedder
from haystack.components.embedders import CohereDocumentEmbedder
from haystack.components.embedders import HuggingFaceAPITextEmbedder
from haystack.components.embedders import HuggingFaceAPIDocumentEmbedder

# Retriever imports - HIGH CONFIDENCE
from haystack.components.retrievers import InMemoryEmbeddingRetriever
from haystack.components.retrievers import PineconeEmbeddingRetriever
from haystack.components.retrievers import WeaviateEmbeddingRetriever

# Pipeline import for embeddings
from haystack import Pipeline
from haystack.components.writers import DocumentWriter

# Embedder instantiation - VERY HIGH CONFIDENCE
text_embedder = OpenAITextEmbedder()
doc_embedder = OpenAIDocumentEmbedder()
sentence_text_embedder = SentenceTransformersTextEmbedder()
sentence_doc_embedder = SentenceTransformersDocumentEmbedder()

# Embedders with parameters
openai_text_embedder = OpenAITextEmbedder(
    model="text-embedding-ada-002",
    api_key="sk-..."
)

openai_doc_embedder = OpenAIDocumentEmbedder(
    model="text-embedding-3-small",
    api_key="sk-..."
)

hf_text_embedder = SentenceTransformersTextEmbedder(
    model="sentence-transformers/all-MiniLM-L6-v2"
)

hf_doc_embedder = SentenceTransformersDocumentEmbedder(
    model="sentence-transformers/all-mpnet-base-v2",
    device="cuda"
)

# Embedding process - VERY HIGH CONFIDENCE
text_result = text_embedder.run(text="Hello world")
doc_result = doc_embedder.run(documents=documents)
embedded_docs = doc_embedder.run(documents=my_documents)
text_embedding_result = sentence_text_embedder.run(text="Machine learning")

# Retriever with document store - VERY HIGH CONFIDENCE
from haystack.document_stores.in_memory import InMemoryDocumentStore

document_store = InMemoryDocumentStore()
retriever = InMemoryEmbeddingRetriever(document_store=document_store)
pinecone_retriever = PineconeEmbeddingRetriever(document_store=pinecone_store)

# Real-world usage examples
def setup_haystack_embedding_pipeline():
    # Initialize components
    text_embedder = OpenAITextEmbedder(model="text-embedding-ada-002")
    doc_embedder = OpenAIDocumentEmbedder(model="text-embedding-ada-002")

    # Create document store and writer
    document_store = InMemoryDocumentStore()
    writer = DocumentWriter(document_store=document_store)

    # Create retriever
    retriever = InMemoryEmbeddingRetriever(document_store=document_store)

    # Create pipeline
    indexing_pipeline = Pipeline()
    indexing_pipeline.add_component("doc_embedder", doc_embedder)
    indexing_pipeline.add_component("writer", writer)

    # Connect components
    indexing_pipeline.connect("doc_embedder.documents", "writer.documents")

    return indexing_pipeline, retriever

def create_query_pipeline():
    # Text embedder for queries
    text_embedder = SentenceTransformersTextEmbedder(
        model="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Document store and retriever
    document_store = InMemoryDocumentStore()
    retriever = InMemoryEmbeddingRetriever(document_store=document_store)

    # Query pipeline
    query_pipeline = Pipeline()
    query_pipeline.add_component("text_embedder", text_embedder)
    query_pipeline.add_component("retriever", retriever)

    # Pipeline connections for embeddings - HIGH CONFIDENCE
    query_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")

    return query_pipeline

def document_embedding_workflow():
    from haystack import Document

    # Create documents
    documents = [
        Document(content="Haystack is a framework for building search systems"),
        Document(content="Embeddings capture semantic meaning of text"),
        Document(content="Vector databases enable similarity search")
    ]

    # Document embedder
    doc_embedder = SentenceTransformersDocumentEmbedder(
        model="sentence-transformers/all-mpnet-base-v2"
    )

    # Embed documents
    embedded_result = doc_embedder.run(documents=documents)
    embedded_docs = embedded_result["documents"]

    # Store embedded documents
    document_store = InMemoryDocumentStore()
    document_store.write_documents(embedded_docs)

    return embedded_docs

# Advanced pipeline setup
class HaystackEmbeddingService:
    def __init__(self):
        self.text_embedder = OpenAITextEmbedder()
        self.doc_embedder = OpenAIDocumentEmbedder()
        self.document_store = InMemoryDocumentStore()
        self.retriever = InMemoryEmbeddingRetriever(
            document_store=self.document_store
        )

    def index_documents(self, documents):
        # Embed documents
        embedding_result = self.doc_embedder.run(documents=documents)
        embedded_docs = embedding_result["documents"]

        # Write to document store
        self.document_store.write_documents(embedded_docs)

    def search(self, query_text, top_k=5):
        # Embed query text
        query_result = self.text_embedder.run(text=query_text)

        # Retrieve similar documents
        retrieval_result = self.retriever.run(
            query_embedding=query_result["embedding"],
            top_k=top_k
        )

        return retrieval_result["documents"]

# Multiple embedder types
def multi_embedder_setup():
    embedders = {
        'openai_text': OpenAITextEmbedder(),
        'openai_doc': OpenAIDocumentEmbedder(),
        'sentence_text': SentenceTransformersTextEmbedder(),
        'sentence_doc': SentenceTransformersDocumentEmbedder(),
        'cohere_text': CohereTextEmbedder(),
        'cohere_doc': CohereDocumentEmbedder()
    }

    # Test each embedder
    test_text = "Test embedding text"
    test_docs = [Document(content="Test document")]

    results = {}
    for name, embedder in embedders.items():
        if 'text' in name:
            result = embedder.run(text=test_text)
        else:
            result = embedder.run(documents=test_docs)
        results[name] = result

    return results

# Pipeline with embedding connections
def complex_embedding_pipeline():
    # Initialize components
    doc_embedder = OpenAIDocumentEmbedder()
    text_embedder = OpenAITextEmbedder()
    document_store = InMemoryDocumentStore()
    writer = DocumentWriter(document_store=document_store)
    retriever = InMemoryEmbeddingRetriever(document_store=document_store)

    # Indexing pipeline
    indexing_pipeline = Pipeline()
    indexing_pipeline.add_component("doc_embedder", doc_embedder)
    indexing_pipeline.add_component("writer", writer)
    indexing_pipeline.connect("doc_embedder.documents", "writer.documents")

    # Query pipeline
    query_pipeline = Pipeline()
    query_pipeline.add_component("text_embedder", text_embedder)
    query_pipeline.add_component("retriever", retriever)
    query_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")

    return indexing_pipeline, query_pipeline

# Document storage with embeddings - HIGH CONFIDENCE
def store_embedded_documents():
    from haystack import Document

    # Create and embed documents
    documents = [Document(content="Sample document")]
    doc_embedder = SentenceTransformersDocumentEmbedder()
    embedded_result = doc_embedder.run(documents=documents)
    embedded_docs = embedded_result["documents"]

    # Store in document store
    document_store = InMemoryDocumentStore()
    document_store.write_documents(embedded_docs)

    return document_store

# Batch processing with embeddings
def batch_document_embedding():
    from haystack import Document

    # Large batch of documents
    documents = [
        Document(content=f"Document {i} about embeddings and search")
        for i in range(100)
    ]

    # Batch embedder
    doc_embedder = SentenceTransformersDocumentEmbedder(
        model="sentence-transformers/all-MiniLM-L6-v2",
        batch_size=32
    )

    # Process documents in batches
    all_embedded_docs = []
    batch_size = 20

    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        result = doc_embedder.run(documents=batch)
        all_embedded_docs.extend(result["documents"])

    return all_embedded_docs

# Custom embedder configurations
custom_openai_text = OpenAITextEmbedder(
    model="text-embedding-3-large",
    dimensions=1024,
    api_key="sk-custom"
)

custom_sentence_doc = SentenceTransformersDocumentEmbedder(
    model="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    device="cuda:0",
    normalize_embeddings=True
)