#!/usr/bin/env python3
"""
Negative test cases for Haystack embeddings detection.
These patterns should NOT be detected by the detect-embeddings-haystack.yaml rule.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
import torch
import pandas as pd

# Generic ML/NLP libraries (not Haystack embeddings)
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import spacy
import nltk

# Non-Haystack transformers
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

# Generic similarity functions (not Haystack)
def calculate_similarity(vec1, vec2):
    return cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))

def similarity_score(text1, text2):
    """Jaccard similarity (not embeddings)"""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    return intersection / union if union > 0 else 0

# Generic model classes (not Haystack)
class SimpleTextEmbedder:
    def __init__(self, model_name="generic-model"):
        self.model_name = model_name

    def run(self, text=None, documents=None):
        # This is NOT Haystack run method
        if text:
            return {"embedding": [len(text.split())]}
        if documents:
            return {"documents": documents}

    def encode(self, texts):
        return [len(text.split()) for text in texts]

# Generic transformer (not Haystack)
class CustomTransformer(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.Transformer(d_model=d_model)

    def forward(self, x):
        return self.transformer(self.embedding(x))

# Generic document store (not Haystack)
class GenericDocumentStore:
    def __init__(self):
        self.documents = []

    def write_documents(self, docs):
        # Not Haystack write_documents
        self.documents.extend(docs)

    def get_all_documents(self):
        return self.documents

# Generic retriever (not Haystack)
class GenericRetriever:
    def __init__(self, document_store=None):
        self.document_store = document_store

    def retrieve(self, query, top_k=10):
        # Generic retrieval (not Haystack embedding retrieval)
        return self.document_store.get_all_documents()[:top_k]

# Generic pipeline (not Haystack)
class DataPipeline:
    def __init__(self):
        self.components = {}
        self.connections = []

    def add_component(self, name, component):
        # Not Haystack add_component
        self.components[name] = component

    def connect(self, source, target):
        # Not Haystack connect
        self.connections.append((source, target))

    def run(self, data):
        result = data
        for name, component in self.components.items():
            if hasattr(component, 'process'):
                result = component.process(result)
        return result

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

    def run(self, text=None, documents=None):
        # Mock API call (not Haystack)
        if text:
            return {"embedding": [0.1, 0.2, 0.3]}
        if documents:
            return {"documents": documents}

    def embed(self, texts):
        return {"embeddings": [[0.1, 0.2, 0.3] for _ in texts]}

# TensorFlow/Keras models (not Haystack)
def tensorflow_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size=10000, embedding_dim=128),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    return model

# PyTorch models (not Haystack)
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

# Spacy NLP (not Haystack)
def spacy_processing():
    import spacy

    # Load spacy model (not Haystack)
    nlp = spacy.load("en_core_web_sm")

    texts = ["Hello world", "Machine learning"]
    processed = []

    for text in texts:
        doc = nlp(text)
        # Get spacy vectors (not Haystack embeddings)
        if doc.has_vector:
            processed.append(doc.vector)

    return processed

# Generic feature extraction (not Haystack)
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

# Generic utilities (not Haystack)
def save_model(model, path):
    import pickle
    with open(path, 'wb') as f:
        pickle.dump(model, f)

def load_model(path):
    """Generic model loading (not Haystack)"""
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

# Generic evaluation (not Haystack)
def evaluate_model(predictions, targets):
    correct = sum(1 for p, t in zip(predictions, targets) if abs(p - t) < 0.1)
    return correct / len(predictions)

# Generic file operations
def save_embeddings(embeddings, filename):
    np.save(filename, embeddings)

def load_embeddings(filename):
    return np.load(filename)

# Generic classes with similar method names (but not Haystack)
class GenericDocumentEmbedder:
    def __init__(self):
        self.model = "generic"

    def run(self, documents=None):
        # Not Haystack - just returns document lengths
        if documents:
            return {"embeddings": [len(doc) for doc in documents]}
        return {}

class CustomTextEmbedder:
    def __init__(self):
        self.model = "custom"

    def run(self, text=None):
        # Not Haystack - just returns text length
        if text:
            return {"embedding": len(text)}
        return {}

class InMemoryRetriever:
    def __init__(self, document_store=None):
        self.store = document_store or []

    def retrieve(self, query_embedding=None, top_k=10):
        # Not Haystack retrieval
        return self.store[:top_k]

# Non-Haystack imports with similar names
from some_other_library import Pipeline as GenericPipeline
from another_library import DocumentStore as GenericStore

# Generic usage that might look like Haystack but isn't
generic_pipeline = GenericPipeline()
generic_store = GenericStore()

# Variables with Haystack-like names but different context
text_embedder = "just a string variable name"
doc_embedder = lambda x: x
document_store = []

# Functions with similar signatures but not Haystack
def create_embeddings(texts):
    return [hash(text) for text in texts]

def process_documents(docs, embedder=None):
    return [doc.upper() for doc in docs]

def write_documents(docs):
    # Generic write function (not Haystack document store)
    with open("docs.txt", "w") as f:
        for doc in docs:
            f.write(doc + "\n")

# Method names that exist in other contexts
class GenericComponent:
    def connect(self, source, target):
        # Generic connect method (not Haystack pipeline)
        print(f"Connecting {source} to {target}")

    def add_component(self, name, component):
        # Generic add_component (not Haystack)
        print(f"Adding {name}: {component}")

# Document class that's not Haystack
class Document:
    def __init__(self, content, metadata=None):
        self.content = content
        self.metadata = metadata or {}