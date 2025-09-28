#!/usr/bin/env python3
"""
Negative test cases for LlamaIndex vector/retrieval detection.
These patterns should NOT be detected by the detect-vector-llamaindex.yaml rule.
"""

import json
import pandas as pd
from typing import List, Dict, Any
import sqlite3

# Generic indexing (not LlamaIndex)
class SimpleIndex:
    def __init__(self, documents):
        self.documents = documents
        self.index = self._build_index()

    def _build_index(self):
        word_index = {}
        for i, doc in enumerate(self.documents):
            words = doc.lower().split()
            for word in words:
                if word not in word_index:
                    word_index[word] = []
                word_index[word].append(i)
        return word_index

    def search(self, query):
        words = query.lower().split()
        doc_scores = {}
        for word in words:
            if word in self.index:
                for doc_id in self.index[word]:
                    doc_scores[doc_id] = doc_scores.get(doc_id, 0) + 1
        return sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

# Generic vector store (not LlamaIndex)
class SimpleVectorStore:
    def __init__(self):
        self.vectors = []
        self.metadata = []

    def add(self, vector, meta=None):
        self.vectors.append(vector)
        self.metadata.append(meta or {})

    def search(self, query_vector, top_k=5):
        import numpy as np
        similarities = []
        for i, vec in enumerate(self.vectors):
            sim = np.dot(query_vector, vec) / (np.linalg.norm(query_vector) * np.linalg.norm(vec))
            similarities.append((i, sim))
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

# Generic document reader (not LlamaIndex)
class SimpleDocumentReader:
    def __init__(self, directory):
        self.directory = directory

    def load_data(self):
        import os
        documents = []
        for filename in os.listdir(self.directory):
            if filename.endswith('.txt'):
                with open(os.path.join(self.directory, filename), 'r') as f:
                    documents.append(f.read())
        return documents

# Generic query engine (not LlamaIndex)
class SimpleQueryEngine:
    def __init__(self, index):
        self.index = index

    def query(self, question):
        # Simple keyword matching
        results = self.index.search(question)
        if results:
            best_doc_id = results[0][0]
            return self.index.documents[best_doc_id]
        return "No results found"

# Generic retriever (not LlamaIndex)
class BasicRetriever:
    def __init__(self, documents):
        self.documents = documents

    def retrieve(self, query, top_k=3):
        # Simple TF-IDF retrieval
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        vectorizer = TfidfVectorizer()
        doc_vectors = vectorizer.fit_transform(self.documents)
        query_vector = vectorizer.transform([query])

        similarities = cosine_similarity(query_vector, doc_vectors).flatten()
        top_indices = similarities.argsort()[-top_k:][::-1]

        return [self.documents[i] for i in top_indices]

# Generic document class (not LlamaIndex)
class SimpleDocument:
    def __init__(self, text, metadata=None):
        self.text = text
        self.metadata = metadata or {}

    def __str__(self):
        return self.text

# Generic storage context (not LlamaIndex)
class SimpleStorageContext:
    def __init__(self, vector_store=None):
        self.vector_store = vector_store
        self.documents = []

    @classmethod
    def from_defaults(cls, vector_store=None):
        return cls(vector_store)

# Generic query bundle (not LlamaIndex)
class SimpleQueryBundle:
    def __init__(self, query_str):
        self.query_str = query_str

# Generic node with score (not LlamaIndex)
class SimpleNodeWithScore:
    def __init__(self, node, score):
        self.node = node
        self.score = score

# Database operations (not vector)
def database_operations():
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()

    # Create table
    cursor.execute('''
        CREATE TABLE documents (
            id INTEGER PRIMARY KEY,
            content TEXT,
            score REAL
        )
    ''')

    # Insert documents
    documents = [
        ("Document 1", 0.8),
        ("Document 2", 0.9),
        ("Document 3", 0.7)
    ]

    for content, score in documents:
        cursor.execute("INSERT INTO documents (content, score) VALUES (?, ?)", (content, score))

    conn.commit()

    # Query with score filtering
    cursor.execute("SELECT * FROM documents WHERE score > ?", (0.75,))
    results = cursor.fetchall()

    conn.close()
    return results

# Generic embedding (not LlamaIndex)
def simple_embeddings():
    import numpy as np

    def encode_text(text):
        # Simple hash-based encoding
        return np.array([hash(word) % 1000 for word in text.split()[:10]])

    texts = ["Hello world", "Machine learning", "Data science"]
    embeddings = [encode_text(text) for text in texts]

    return embeddings

# Generic evaluation (not LlamaIndex)
def evaluate_system():
    predictions = ["answer1", "answer2", "answer3"]
    ground_truth = ["answer1", "answer2", "answer4"]

    accuracy = sum(1 for p, g in zip(predictions, ground_truth) if p == g) / len(predictions)
    return accuracy

# Generic memory/chat (not LlamaIndex)
class SimpleChatMemory:
    def __init__(self, max_turns=10):
        self.history = []
        self.max_turns = max_turns

    def add_turn(self, user_message, assistant_response):
        self.history.append({
            "user": user_message,
            "assistant": assistant_response
        })
        if len(self.history) > self.max_turns:
            self.history.pop(0)

    def get_context(self):
        return "\n".join([f"User: {turn['user']}\nAssistant: {turn['assistant']}"
                         for turn in self.history])

# Generic file operations
def file_operations():
    documents = ["doc1.txt", "doc2.txt", "doc3.txt"]

    # Save documents
    for i, doc in enumerate(documents):
        with open(f"document_{i}.txt", "w") as f:
            f.write(doc)

    # Load documents
    loaded = []
    for i in range(len(documents)):
        with open(f"document_{i}.txt", "r") as f:
            loaded.append(f.read())

    return loaded

# Generic search and filtering
def search_and_filter():
    data = [
        {"text": "AI is amazing", "score": 0.9},
        {"text": "Machine learning rocks", "score": 0.8},
        {"text": "Data science is cool", "score": 0.7},
        {"text": "Programming is fun", "score": 0.6}
    ]

    # Filter by score
    high_score_items = [item for item in data if item["score"] > 0.75]

    # Generic similarity calculation
    def similarity(text1, text2):
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        return intersection / union if union > 0 else 0

    # Find similar items
    query = "AI machine learning"
    similarities = []
    for item in data:
        sim = similarity(query, item["text"])
        similarities.append((item, sim))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities

# Generic API client
class GenericAPIClient:
    def __init__(self, endpoint):
        self.endpoint = endpoint

    def query(self, prompt):
        # Mock API call
        return {"response": f"Generated response for: {prompt}"}

    def retrieve(self, query):
        # Mock retrieval
        return [f"Document about {query}", f"Another doc about {query}"]

# Generic data processing
def process_data():
    df = pd.DataFrame({
        'text': ['Document 1', 'Document 2', 'Document 3'],
        'score': [0.8, 0.9, 0.7],
        'category': ['A', 'B', 'A']
    })

    # Filter and process
    high_score = df[df['score'] > 0.75]
    grouped = df.groupby('category')['score'].mean()

    return high_score, grouped

# Generic ML models (not LlamaIndex)
def ml_models():
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier

    # Generic classifiers
    lr_model = LogisticRegression()
    rf_model = RandomForestClassifier()

    # Mock training data
    import numpy as np
    X = np.random.randn(100, 10)
    y = np.random.randint(0, 2, 100)

    # Train models
    lr_model.fit(X, y)
    rf_model.fit(X, y)

    # Predictions
    predictions = lr_model.predict(X[:10])

    return predictions

# Generic text processing
def text_processing():
    texts = [
        "This is a sample document",
        "Another document for testing",
        "Final document in the collection"
    ]

    # Simple processing
    processed = []
    for text in texts:
        words = text.lower().split()
        processed.append(" ".join(words))

    return processed

# Generic configuration and utilities
CONFIG = {
    "top_k": 5,
    "similarity_threshold": 0.7,
    "model_name": "generic-model"
}

def load_config():
    return CONFIG

def save_results(results, filename):
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)