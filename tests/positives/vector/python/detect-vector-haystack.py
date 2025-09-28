#!/usr/bin/env python3
"""
Positive test cases for Haystack vector/retrieval detection.
These patterns should be detected by the detect-vector-haystack.yaml rule.
"""

# Document store imports
from haystack.document_stores import InMemoryDocumentStore
from haystack.document_stores import ElasticsearchDocumentStore
document_store = InMemoryDocumentStore()
es_store = InMemoryDocumentStore(use_gpu=True)

# Retriever imports and creation
from haystack.components.retrievers import InMemoryEmbeddingRetriever
from haystack.components.retrievers import BM25Retriever
retriever = InMemoryEmbeddingRetriever(document_store=document_store)
bm25_retriever = InMemoryEmbeddingRetriever(document_store=es_store)

# RAG pipeline creation
from haystack import Pipeline
rag_pipeline = Pipeline()
qa_pipeline = Pipeline()

# Pipeline component addition
rag_pipeline.add_component("retriever", retriever)
rag_pipeline.add_component("text_embedder", text_embedder)
rag_pipeline.add_component("generator", generator)

# Pipeline connections for RAG - HIGH CONFIDENCE (specific to Haystack RAG)
rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
rag_pipeline.connect("retriever", "prompt_builder.documents")
rag_pipeline.connect("prompt_builder", "generator")
rag_pipeline.connect("text_embedder", "retriever")
rag_pipeline.connect("retriever.documents", "prompt_builder")

# RAG template patterns
template = """
Given the following information, answer the question.

Context:
{% for document in documents %}
  {{ document.content }}
{% endfor %}

Question: {{ question }}
Answer:
"""

from haystack.components.builders import PromptBuilder
prompt_builder = PromptBuilder(template=template)

# Pipeline execution for RAG
result = rag_pipeline.run({
    "text_embedder": {"text": query},
    "prompt_builder": {"question": question}
})

pipeline_result = rag_pipeline.run({})

# Document writing to store
document_store.write_documents(embedded_docs["documents"])
document_store.write_documents(documents)

# Generator component
from haystack.components.generators import OpenAIGenerator
generator = OpenAIGenerator(api_key="sk-...")
openai_gen = OpenAIGenerator(model="gpt-3.5-turbo")

# Document creation and processing
from haystack import Document
documents = [Document(content="Sample text"), Document(content="Another doc")]
doc = Document(content="This is a test document")

# Embedding and retrieval patterns - HIGH CONFIDENCE
embedded_docs = doc_embedder.run(documents=documents)
embeddings = text_embedder.run(documents=docs)
result = result["generator"]["replies"][0]
docs = result["retriever"]["documents"]

# Real-world Haystack RAG pipeline
def setup_haystack_rag():
    from haystack import Pipeline, Document
    from haystack.document_stores import InMemoryDocumentStore
    from haystack.components.retrievers import InMemoryEmbeddingRetriever
    from haystack.components.generators import OpenAIGenerator
    from haystack.components.builders import PromptBuilder
    from haystack.components.embedders import SentenceTransformersTextEmbedder

    # Create document store
    document_store = InMemoryDocumentStore()

    # Create documents
    documents = [
        Document(content="Paris is the capital of France."),
        Document(content="Berlin is the capital of Germany."),
        Document(content="Rome is the capital of Italy.")
    ]

    # Write documents to store
    document_store.write_documents(documents)

    # Create pipeline
    pipeline = Pipeline()

    # Add components
    text_embedder = SentenceTransformersTextEmbedder()
    retriever = InMemoryEmbeddingRetriever(document_store=document_store)
    prompt_builder = PromptBuilder(template=template)
    generator = OpenAIGenerator()

    pipeline.add_component("text_embedder", text_embedder)
    pipeline.add_component("retriever", retriever)
    pipeline.add_component("prompt_builder", prompt_builder)
    pipeline.add_component("generator", generator)

    # Connect components
    pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    pipeline.connect("retriever", "prompt_builder.documents")
    pipeline.connect("prompt_builder", "generator")

    return pipeline

# Multiple document stores
def multi_store_setup():
    from haystack.document_stores import InMemoryDocumentStore

    # Different stores
    store1 = InMemoryDocumentStore()
    store2 = InMemoryDocumentStore(use_bm25=True)

    # Multiple retrievers
    retriever1 = InMemoryEmbeddingRetriever(document_store=store1)
    retriever2 = InMemoryEmbeddingRetriever(document_store=store2)

    return store1, store2, retriever1, retriever2

# Advanced pipeline patterns
class CustomHaystackPipeline:
    def __init__(self):
        self.pipeline = Pipeline()
        self.document_store = InMemoryDocumentStore()

    def setup_components(self):
        from haystack.components.retrievers import InMemoryEmbeddingRetriever
        from haystack.components.generators import OpenAIGenerator

        retriever = InMemoryEmbeddingRetriever(document_store=self.document_store)
        generator = OpenAIGenerator()

        self.pipeline.add_component("retriever", retriever)
        self.pipeline.add_component("generator", generator)

        # Connect components
        self.pipeline.connect("retriever", "generator")

    def run_query(self, query):
        result = self.pipeline.run({
            "retriever": {"query": query}
        })
        return result["generator"]["replies"][0]

# Document preprocessing
def preprocess_documents():
    from haystack import Document
    from haystack.components.preprocessors import DocumentSplitter

    docs = [
        Document(content="Long document content here..."),
        Document(content="Another long document...")
    ]

    splitter = DocumentSplitter(split_by="sentence", split_length=2)
    split_docs = splitter.run(documents=docs)

    return split_docs["documents"]

# Embedding components
def setup_embeddings():
    from haystack.components.embedders import SentenceTransformersTextEmbedder
    from haystack.components.embedders import SentenceTransformersDocumentEmbedder

    text_embedder = SentenceTransformersTextEmbedder()
    doc_embedder = SentenceTransformersDocumentEmbedder()

    # Embed text
    text_result = text_embedder.run(text="Query text")
    doc_result = doc_embedder.run(documents=documents)

    return text_result, doc_result

# Question answering pipeline
def qa_pipeline_setup():
    from haystack import Pipeline
    from haystack.components.builders import AnswerBuilder

    pipeline = Pipeline()

    # Add QA-specific components
    answer_builder = AnswerBuilder()
    pipeline.add_component("answer_builder", answer_builder)

    # Connect for QA
    pipeline.connect("retriever.documents", "answer_builder")

    return pipeline

# Multi-modal Haystack patterns
def multimodal_setup():
    from haystack.components.retrievers import InMemoryEmbeddingRetriever
    from haystack.document_stores import InMemoryDocumentStore

    # Document store for different modalities
    text_store = InMemoryDocumentStore()

    # Retrievers for different content types
    text_retriever = InMemoryEmbeddingRetriever(document_store=text_store)

    return text_store, text_retriever

# Evaluation patterns
def evaluate_pipeline():
    from haystack.evaluation import EvaluationRunResult

    # Evaluate retrieval
    retrieval_result = pipeline.run({"query": "test query"})

    # Process results
    documents = retrieval_result["retriever"]["documents"]

    return documents

# Async Haystack operations
async def async_haystack_pipeline():
    pipeline = Pipeline()

    # Async pipeline execution
    result = await pipeline.arun({
        "text_embedder": {"text": "async query"}
    })

    return result