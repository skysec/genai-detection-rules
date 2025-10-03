#!/usr/bin/env python3
"""
Negative test cases for LlamaIndex embeddings detection.
These patterns should NOT be detected by the detect-embeddings-llamaindex.yaml rule.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
import torch
import pandas as pd

# Generic ML/NLP libraries (not LlamaIndex embeddings)
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import spacy
import nltk

# Non-LlamaIndex transformers
from transformers import AutoModel, AutoTokenizer
import torch.nn as nn

# Generic encoding functions (not embeddings)
def simple_encode(texts):
    """Generic encoding function"""
    vectorizer = TfidfVectorizer()
    return vectorizer.fit_transform(texts)

def encode_text(text):
    """Base64 encoding (not embeddings)"""
    import base64
    return base64.b64encode(text.encode()).decode()

# Generic similarity functions (not LlamaIndex)
def calculate_similarity(vec1, vec2):
    return cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))

def similarity_score(text1, text2):
    """Jaccard similarity (not embeddings)"""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    return intersection / union if union > 0 else 0

# Generic model classes (not LlamaIndex)
class SimpleEmbedding:
    def __init__(self, model_name="generic-model"):
        self.model_name = model_name

    def get_text_embedding(self, text):
        # This is NOT LlamaIndex get_text_embedding
        return [len(text.split())]

    def get_query_embedding(self, query):
        # This is NOT LlamaIndex get_query_embedding
        return [len(query.split())]

    def encode(self, texts):
        return [len(text.split()) for text in texts]

# Generic transformer (not LlamaIndex)
class CustomTransformer(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.Transformer(d_model=d_model)

    def forward(self, x):
        return self.transformer(self.embedding(x))

# Generic vector store (not LlamaIndex)
class GenericVectorStore:
    def __init__(self):
        self.vectors = []

    def from_documents(self, docs, embed_model=None):
        # Not LlamaIndex from_documents
        return self

    def add_documents(self, docs):
        self.vectors.extend(docs)

    def search(self, query, k=5):
        return self.vectors[:k]

# Generic index (not LlamaIndex)
class GenericIndex:
    def __init__(self, documents=None):
        self.documents = documents or []

    @classmethod
    def from_documents(cls, documents, embed_model=None, service_context=None, storage_context=None):
        # Not LlamaIndex VectorStoreIndex
        return cls(documents)

    def query(self, text):
        return f"Generic query result for: {text}"

# Generic settings (not LlamaIndex)
class GenericSettings:
    def __init__(self):
        self.embed_model = None

    def configure(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

# Generic database operations
import sqlite3

def database_operations():
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()

    # Create table
    cursor.execute('''
        CREATE TABLE documents (
            id INTEGER PRIMARY KEY,
            content TEXT,
            embedding BLOB
        )
    ''')

    # Insert documents
    documents = ['doc1', 'doc2', 'doc3']
    for doc in documents:
        cursor.execute("INSERT INTO documents (content) VALUES (?)", (doc,))

    conn.commit()
    conn.close()

# Generic API client
class APIClient:
    def __init__(self, endpoint):
        self.endpoint = endpoint

    def get_text_embedding(self, text):
        # Mock API call (not LlamaIndex)
        return {"embedding": [0.1, 0.2, 0.3]}

    def get_query_embedding(self, query):
        # Mock API call (not LlamaIndex)
        return {"embedding": [0.1, 0.2, 0.3]}

# TensorFlow/Keras models (not LlamaIndex)
def tensorflow_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size=10000, embedding_dim=128),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    return model

# PyTorch models (not LlamaIndex)
def pytorch_model():
    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(10000, 128)
            self.lstm = nn.LSTM(128, 64)
            self.fc = nn.Linear(64, 1)

        def forward(self, x):
            x = self.embedding(x)
            x, _ = self.lstm(x)
            return self.fc(x[:, -1, :])

    return SimpleNet()

# Spacy NLP (not LlamaIndex)
def spacy_processing():
    import spacy

    # Load spacy model (not LlamaIndex)
    nlp = spacy.load("en_core_web_sm")

    texts = ["Hello world", "Machine learning"]
    processed = []

    for text in texts:
        doc = nlp(text)
        # Get spacy vectors (not LlamaIndex embeddings)
        if doc.has_vector:
            processed.append(doc.vector)

    return processed

# Generic feature extraction (not LlamaIndex)
def extract_features(texts):
    features = []
    for text in texts:
        # Simple feature extraction
        features.append({
            'length': len(text),
            'words': len(text.split()),
            'chars': len(set(text))
        })
    return features

# Generic search functions (not semantic search)
def keyword_search(query, documents):
    results = []
    query_words = set(query.lower().split())

    for doc in documents:
        doc_words = set(doc.lower().split())
        overlap = len(query_words & doc_words)
        if overlap > 0:
            results.append((doc, overlap))

    results.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in results]

# Generic utilities (not LlamaIndex)
def save_model(model, path):
    import pickle
    with open(path, 'wb') as f:
        pickle.dump(model, f)

def load_model(path):
    """Generic model loading (not LlamaIndex)"""
    import pickle
    with open(path, 'rb') as f:
        return pickle.load(f)

# Generic preprocessing (not embeddings)
def preprocess_texts(texts):
    processed = []
    for text in texts:
        # Simple cleaning
        clean_text = text.lower().strip()
        processed.append(clean_text)
    return processed

# Non-embedding vector operations
def create_random_vectors(size, dimension):
    return np.random.randn(size, dimension)

def vector_operations():
    # Generic numpy operations (not embeddings)
    vec1 = np.array([1, 2, 3, 4])
    vec2 = np.array([5, 6, 7, 8])

    # Basic vector math
    addition = vec1 + vec2
    multiplication = vec1 * vec2
    dot_product = np.dot(vec1, vec2)

    return addition, multiplication, dot_product

# Generic configuration and utilities
CONFIG = {
    'model_name': 'generic-model',
    'max_length': 512,
    'batch_size': 32
}

def load_config():
    return CONFIG

# Generic evaluation (not LlamaIndex)
def evaluate_model(predictions, targets):
    correct = sum(1 for p, t in zip(predictions, targets) if abs(p - t) < 0.1)
    return correct / len(predictions)

# Generic file operations
def save_embeddings(embeddings, filename):
    np.save(filename, embeddings)

def load_embeddings(filename):
    return np.load(filename)

# Generic classes with similar method names (but not LlamaIndex)
class GenericOpenAI:
    def __init__(self):
        self.model = "generic"

    def get_text_embedding(self, text):
        # Not LlamaIndex - just returns text length
        return len(text)

    def get_query_embedding(self, query):
        # Not LlamaIndex - just returns query length
        return len(query)

class CustomHuggingFace:
    def __init__(self):
        self.model = "custom"

    def encode(self, texts):
        # Not LlamaIndex encoding
        return [hash(text) for text in texts]

class GenericChroma:
    def __init__(self, chroma_collection=None):
        self.collection = chroma_collection

    def search(self, query, k=5):
        # Generic search (not LlamaIndex ChromaVectorStore)
        return []

# Non-LlamaIndex imports with similar names
from some_other_library import VectorStoreIndex as GenericVectorIndex
from another_library import Settings as GenericSettings

# Generic usage that might look like LlamaIndex but isn't
generic_index = GenericVectorIndex()
generic_settings = GenericSettings()

# Variables with LlamaIndex-like names but different context
embed_model = "just a string variable name"
vector_store = []
storage_context = {}

# Functions with similar signatures but not LlamaIndex
def create_embeddings(texts):
    return [hash(text) for text in texts]

def from_documents(docs, embed_model=None, service_context=None):
    # Generic document processing (not LlamaIndex)
    return docs

def get_text_embedding(text):
    # Generic function (not LlamaIndex method)
    return [ord(c) for c in text[:5]]

# Method names that exist in other contexts
class GenericService:
    def from_defaults(self, **kwargs):
        # Generic from_defaults method (not LlamaIndex)
        return self

# Document class that's not LlamaIndex
class Document:
    def __init__(self, text, metadata=None):
        self.text = text
        self.metadata = metadata or {}

# Storage context that's not LlamaIndex
class StorageContext:
    def __init__(self, vector_store=None):
        self.vector_store = vector_store

    @classmethod
    def from_defaults(cls, vector_store=None):
        # Not LlamaIndex StorageContext
        return cls(vector_store)

# Base embedding class that's not LlamaIndex
class BaseEmbedding:
    def __init__(self):
        self.model_name = "base"

    def embed(self, texts):
        return [len(text) for text in texts]