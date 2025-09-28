#!/usr/bin/env python3
"""
Negative test cases for Haystack vector/retrieval detection.
These patterns should NOT be detected by the detect-vector-haystack.yaml rule.
"""

import json
import sqlite3
from typing import List, Dict
import pandas as pd

# Generic pipeline libraries (not Haystack)
from sklearn.pipeline import Pipeline as SklearnPipeline
from transformers import pipeline as hf_pipeline

# Generic document storage (not Haystack)
class SimpleDocumentStore:
    def __init__(self):
        self.documents = []

    def add_documents(self, docs):
        self.documents.extend(docs)

    def search(self, query):
        results = []
        for doc in self.documents:
            if query.lower() in doc.lower():
                results.append(doc)
        return results

# Generic retriever (not Haystack)
class BasicRetriever:
    def __init__(self, document_store):
        self.store = document_store

    def retrieve(self, query, top_k=5):
        all_docs = self.store.documents
        # Simple keyword matching
        scores = []
        for doc in all_docs:
            score = len([word for word in query.split() if word.lower() in doc.lower()])
            scores.append((doc, score))

        # Sort and return top-k
        scores.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scores[:top_k]]

# Generic generator (not Haystack)
class SimpleGenerator:
    def __init__(self, model_name="gpt-3.5-turbo"):
        self.model = model_name

    def generate(self, prompt):
        # Mock generation
        return f"Generated response for: {prompt}"

# Generic document class (not Haystack)
class SimpleDocument:
    def __init__(self, content, metadata=None):
        self.content = content
        self.metadata = metadata or {}

# Generic pipeline (not Haystack)
class BasicPipeline:
    def __init__(self):
        self.components = {}
        self.connections = []

    def add_component(self, name, component):
        self.components[name] = component

    def connect(self, source, target):
        self.connections.append((source, target))

    def run(self, inputs):
        # Mock pipeline execution
        return {"results": "processed"}

# Generic embedder (not Haystack)
class SimpleEmbedder:
    def __init__(self, model="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = model

    def embed(self, texts):
        # Mock embedding
        import numpy as np
        return np.random.randn(len(texts), 384)

# Generic prompt builder
class SimplePromptBuilder:
    def __init__(self, template):
        self.template = template

    def build(self, **kwargs):
        return self.template.format(**kwargs)

# Database operations (not vector store)
def database_operations():
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()

    # Create table
    cursor.execute('''
        CREATE TABLE documents (
            id INTEGER PRIMARY KEY,
            content TEXT,
            metadata TEXT
        )
    ''')

    # Insert documents
    documents = [
        "Document 1 content",
        "Document 2 content",
        "Document 3 content"
    ]

    for doc in documents:
        cursor.execute("INSERT INTO documents (content) VALUES (?)", (doc,))

    conn.commit()
    return conn

# Generic text processing
def process_text(text):
    # Simple text cleaning
    cleaned = text.lower().strip()
    words = cleaned.split()
    return words

# Generic search functionality
def search_documents(query, documents):
    results = []
    query_words = set(query.lower().split())

    for doc in documents:
        doc_words = set(doc.lower().split())
        overlap = len(query_words & doc_words)
        if overlap > 0:
            results.append((doc, overlap))

    # Sort by relevance
    results.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in results]

# Generic ML pipeline
def ml_pipeline():
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    # Sklearn pipeline (not Haystack)
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('classifier', LogisticRegression())
    ])

    return pipeline

# Generic API client
class APIClient:
    def __init__(self, base_url):
        self.base_url = base_url

    def query(self, prompt):
        # Mock API call
        return {"response": f"Answer to: {prompt}"}

    def generate(self, context, question):
        # Mock generation
        return f"Based on {context}, the answer is..."

# Generic evaluation
def evaluate_results(predictions, ground_truth):
    correct = 0
    for pred, truth in zip(predictions, ground_truth):
        if pred == truth:
            correct += 1
    return correct / len(predictions)

# Generic data processing
def preprocess_data(data):
    df = pd.DataFrame(data)
    df['cleaned'] = df['text'].str.lower().str.strip()
    return df

# Generic caching
class Cache:
    def __init__(self):
        self.cache = {}

    def get(self, key):
        return self.cache.get(key)

    def set(self, key, value):
        self.cache[key] = value

# Generic configuration
CONFIG = {
    "model_name": "gpt-3.5-turbo",
    "max_tokens": 100,
    "temperature": 0.7
}

# Generic utility functions
def chunk_text(text, chunk_size=1000):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])
    return chunks

def calculate_similarity(text1, text2):
    from difflib import SequenceMatcher
    return SequenceMatcher(None, text1, text2).ratio()

# Generic file operations
def save_results(results, filename):
    with open(filename, 'w') as f:
        json.dump(results, f)

def load_config(filename):
    with open(filename, 'r') as f:
        return json.load(f)

# HuggingFace transformers (not Haystack)
def use_transformers():
    from transformers import pipeline

    # This is HuggingFace, not Haystack
    qa_pipeline = pipeline("question-answering")
    text_gen = pipeline("text-generation")

    answer = qa_pipeline(question="What is AI?", context="AI is artificial intelligence")
    generated = text_gen("The future of AI is")

    return answer, generated

# Generic neural network
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.layers(x)

# Generic data loading
def load_documents(directory):
    import os
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            with open(os.path.join(directory, filename), 'r') as f:
                documents.append(f.read())
    return documents