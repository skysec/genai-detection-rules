#!/usr/bin/env python3
"""
Negative test cases for HuggingFace embeddings detection.
These patterns should NOT be detected by the detect-embeddings-huggingface.yaml rule.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
import torch
import pandas as pd

# Generic ML/NLP libraries (not HuggingFace embeddings)
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import spacy
import nltk

# Non-HuggingFace transformers
from tensorflow.keras.models import Model
import torch.nn as nn

# Generic encoding/embedding functions (not HuggingFace)
def simple_encode(texts):
    """Generic encoding function"""
    vectorizer = TfidfVectorizer()
    return vectorizer.fit_transform(texts)

def encode_text(text):
    """Base64 encoding (not embeddings)"""
    import base64
    return base64.b64encode(text.encode()).decode()

# Generic similarity functions (not HuggingFace)
def calculate_similarity(vec1, vec2):
    return cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))

def similarity_score(text1, text2):
    """Jaccard similarity (not embeddings)"""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    return intersection / union if union > 0 else 0

# Generic model classes (not HuggingFace)
class SimpleModel:
    def __init__(self, model_name="generic-model"):
        self.model_name = model_name

    def encode(self, texts):
        # This is NOT HuggingFace encode
        return [len(text.split()) for text in texts]

    def predict(self, inputs):
        return [0.5] * len(inputs)

# Generic transformer (not HuggingFace)
class CustomTransformer(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.Transformer(d_model=d_model)

    def forward(self, x):
        return self.transformer(self.embedding(x))

# Generic evaluation (not HuggingFace)
def evaluate_model(predictions, targets):
    correct = sum(1 for p, t in zip(predictions, targets) if abs(p - t) < 0.1)
    return correct / len(predictions)

# Non-HuggingFace pipeline
class DataPipeline:
    def __init__(self):
        self.steps = []

    def add_step(self, step):
        self.steps.append(step)

    def run(self, data):
        result = data
        for step in self.steps:
            result = step(result)
        return result

# Generic feature extraction (not HuggingFace)
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

# Generic utilities (not HuggingFace)
def save_model(model, path):
    import pickle
    with open(path, 'wb') as f:
        pickle.dump(model, f)

def from_pretrained(model_path):
    """Generic model loading (not HuggingFace)"""
    import pickle
    with open(model_path, 'rb') as f:
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
            vector BLOB
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

    def encode(self, texts):
        # Mock API call (not HuggingFace)
        return {"embeddings": [[0.1, 0.2, 0.3] for _ in texts]}

    def predict(self, inputs):
        return {"predictions": [0.5] * len(inputs)}

# TensorFlow/Keras models (not HuggingFace)
def tensorflow_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size=10000, embedding_dim=128),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    return model

# PyTorch models (not HuggingFace)
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

# Spacy NLP (not HuggingFace)
def spacy_processing():
    import spacy

    # Load spacy model (not HuggingFace)
    nlp = spacy.load("en_core_web_sm")

    texts = ["Hello world", "Machine learning"]
    processed = []

    for text in texts:
        doc = nlp(text)
        # Get spacy vectors (not HuggingFace embeddings)
        if doc.has_vector:
            processed.append(doc.vector)

    return processed

# Generic file operations
def save_embeddings(embeddings, filename):
    np.save(filename, embeddings)

def load_embeddings(filename):
    return np.load(filename)