#!/usr/bin/env python3
"""
Negative test cases for LangChain vector/retrieval detection.
These patterns should NOT be detected by the detect-vector-langchain.yaml rule.
"""

import requests
import json
from typing import List, Dict
import pandas as pd

# Generic libraries (not LangChain)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import sqlite3

# Generic chain/pipeline libraries (not LangChain)
from transformers import pipeline
import torch.nn as nn

# Non-LangChain document processing
class SimpleDocumentLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load(self):
        with open(self.file_path, 'r') as f:
            return f.read()

# Generic text splitting (not LangChain)
def simple_text_splitter(text, chunk_size=1000):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])
    return chunks

# Non-vector database operations
class SimpleDatabase:
    def __init__(self):
        self.conn = sqlite3.connect(':memory:')
        self.cursor = self.conn.cursor()
        self.cursor.execute('''
            CREATE TABLE documents (id INTEGER PRIMARY KEY, content TEXT)
        ''')

    def add_documents(self, docs):
        for doc in docs:
            self.cursor.execute("INSERT INTO documents (content) VALUES (?)", (doc,))
        self.conn.commit()

    def search(self, query):
        self.cursor.execute(
            "SELECT content FROM documents WHERE content LIKE ?",
            (f"%{query}%",)
        )
        return [row[0] for row in self.cursor.fetchall()]

# Generic similarity search (not vector-based)
def keyword_similarity_search(query, documents, k=5):
    scores = []
    query_words = set(query.lower().split())

    for doc in documents:
        doc_words = set(doc.lower().split())
        intersection = len(query_words & doc_words)
        union = len(query_words | doc_words)
        similarity = intersection / union if union > 0 else 0
        scores.append(similarity)

    # Get top-k
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [documents[i] for i in top_indices]

# Generic QA system (not RAG)
class SimpleQASystem:
    def __init__(self, knowledge_base):
        self.knowledge_base = knowledge_base

    def run(self, question):
        # Simple keyword matching
        question_words = question.lower().split()
        best_answer = ""
        best_score = 0

        for entry in self.knowledge_base:
            entry_words = entry.lower().split()
            score = len(set(question_words) & set(entry_words))
            if score > best_score:
                best_score = score
                best_answer = entry

        return best_answer

# Generic retrieval (not vector-based)
class KeywordRetriever:
    def __init__(self, documents):
        self.documents = documents

    def retrieve(self, query, top_k=3):
        # Simple TF-IDF based retrieval
        vectorizer = TfidfVectorizer()
        doc_vectors = vectorizer.fit_transform(self.documents)
        query_vector = vectorizer.transform([query])

        similarities = cosine_similarity(query_vector, doc_vectors).flatten()
        top_indices = similarities.argsort()[-top_k:][::-1]

        return [self.documents[i] for i in top_indices]

# Generic file operations
def save_data(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f)

def load_data(filename):
    with open(filename, 'r') as f:
        return json.load(f)

# Generic API client
class GenericAPIClient:
    def __init__(self, base_url):
        self.base_url = base_url

    def query(self, prompt):
        response = requests.post(
            f"{self.base_url}/generate",
            json={"prompt": prompt}
        )
        return response.json()

# Generic neural network chains
class SimpleNeuralChain(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.chain = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.chain(x)

# Generic document processing
def process_documents(docs):
    processed = []
    for doc in docs:
        # Simple processing
        clean_doc = doc.strip().lower()
        processed.append(clean_doc)
    return processed

# Generic search functionality
def search_documents(query, document_store):
    results = []
    for doc in document_store:
        if query.lower() in doc.lower():
            results.append(doc)
    return results

# Generic embedding (not vector store)
def create_word_embeddings(vocabulary, dimension=100):
    embeddings = {}
    for word in vocabulary:
        embeddings[word] = np.random.randn(dimension)
    return embeddings

# Generic similarity function
def calculate_similarity(text1, text2):
    from difflib import SequenceMatcher
    return SequenceMatcher(None, text1, text2).ratio()

# Generic data pipeline
class DataPipeline:
    def __init__(self):
        self.steps = []

    def add_step(self, step_func):
        self.steps.append(step_func)

    def run(self, data):
        result = data
        for step in self.steps:
            result = step(result)
        return result

# Generic caching
class SimpleCache:
    def __init__(self):
        self.cache = {}

    def get(self, key):
        return self.cache.get(key)

    def set(self, key, value):
        self.cache[key] = value

# Generic configuration
CONFIG = {
    "chunk_size": 1000,
    "top_k": 5,
    "similarity_threshold": 0.7
}

# Generic utility functions
def chunk_text(text, size=1000):
    return [text[i:i+size] for i in range(0, len(text), size)]

def filter_results(results, threshold=0.5):
    return [r for r in results if r.get('score', 0) > threshold]

# Generic evaluation
def evaluate_system(predictions, ground_truth):
    correct = sum(1 for p, gt in zip(predictions, ground_truth) if p == gt)
    return correct / len(predictions)

# Non-LangChain transformers usage
def use_huggingface_transformers():
    from transformers import pipeline

    # This is HuggingFace transformers, not LangChain
    classifier = pipeline("sentiment-analysis")
    qa_pipeline = pipeline("question-answering")

    result = classifier("I love this!")
    answer = qa_pipeline(question="What is AI?", context="AI is artificial intelligence")

    return result, answer