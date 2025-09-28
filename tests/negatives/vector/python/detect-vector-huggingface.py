#!/usr/bin/env python3
"""
Negative test cases for HuggingFace vector/embedding detection.
These patterns should NOT be detected by the detect-vector-huggingface.yaml rule.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
import torch
import pandas as pd

# Generic ML/NLP libraries (not HuggingFace specific)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import spacy
import nltk

# Non-HuggingFace transformers or models
from tensorflow.keras.models import Sequential
from torch.nn import Transformer
import keras

# Generic vector operations (not HuggingFace specific)
def compute_similarity(vec1, vec2):
    return cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))

def generic_encode(texts):
    vectorizer = TfidfVectorizer()
    return vectorizer.fit_transform(texts)

# Non-HuggingFace semantic search
class GenericSearchEngine:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.embeddings = None

    def encode(self, texts):
        return self.vectorizer.fit_transform(texts)

    def search(self, query, top_k=5):
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.embeddings)
        return similarities

# Generic document processing
def process_documents(docs):
    results = []
    for doc in docs:
        # Generic text processing
        processed = doc.lower().strip()
        results.append(processed)
    return results

# Non-vector database operations
class SimpleDatabase:
    def __init__(self):
        self.documents = []

    def add_documents(self, docs):
        self.documents.extend(docs)

    def search(self, query):
        # Simple text matching (no vectors)
        results = []
        for doc in self.documents:
            if query.lower() in doc.lower():
                results.append(doc)
        return results

# Generic machine learning patterns
def train_classifier():
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier

    model = LogisticRegression()
    rf_model = RandomForestClassifier()
    return model, rf_model

# Generic neural networks (not transformers)
def create_neural_network():
    model = Sequential([
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

# Non-HuggingFace transformers
def pytorch_transformer():
    transformer = torch.nn.Transformer(
        d_model=512,
        nhead=8,
        num_encoder_layers=6
    )
    return transformer

# Generic text similarity (not embedding-based)
def text_similarity(text1, text2):
    from difflib import SequenceMatcher
    return SequenceMatcher(None, text1, text2).ratio()

# Generic data processing
def preprocess_data(data):
    df = pd.DataFrame(data)
    df['processed'] = df['text'].str.lower()
    return df

# Generic file operations
def save_model(model, path):
    import pickle
    with open(path, 'wb') as f:
        pickle.dump(model, f)

def load_model(path):
    import pickle
    with open(path, 'rb') as f:
        return pickle.load(f)

# Generic evaluation metrics
def evaluate_model(predictions, targets):
    from sklearn.metrics import accuracy_score, precision_score
    accuracy = accuracy_score(targets, predictions)
    precision = precision_score(targets, predictions)
    return accuracy, precision

# Non-vector search engines
class KeywordSearchEngine:
    def __init__(self):
        self.index = {}

    def build_index(self, documents):
        for i, doc in enumerate(documents):
            words = doc.split()
            for word in words:
                if word not in self.index:
                    self.index[word] = []
                self.index[word].append(i)

    def search(self, query):
        words = query.split()
        results = set()
        for word in words:
            if word in self.index:
                results.update(self.index[word])
        return list(results)

# Generic API clients
class APIClient:
    def __init__(self, base_url):
        self.base_url = base_url

    def predict(self, data):
        # Generic API call
        return {"prediction": "example"}

# Generic utility functions that might be confused with HuggingFace
def encode_text(text):
    # Base64 encoding (not embeddings)
    import base64
    return base64.b64encode(text.encode()).decode()

def similarity_score(a, b):
    # Generic similarity (not cosine similarity)
    return len(set(a) & set(b)) / len(set(a) | set(b))

# Database operations (not vector stores)
def database_operations():
    import sqlite3
    conn = sqlite3.connect('example.db')
    cursor = conn.cursor()

    # Create table
    cursor.execute('''CREATE TABLE documents
                     (id INTEGER PRIMARY KEY, content TEXT)''')

    # Insert documents
    cursor.execute("INSERT INTO documents (content) VALUES (?)", ("example text",))

    # Search documents
    cursor.execute("SELECT * FROM documents WHERE content LIKE ?", ("%example%",))
    results = cursor.fetchall()

    conn.close()
    return results