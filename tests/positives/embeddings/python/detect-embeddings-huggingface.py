#!/usr/bin/env python3
"""
Positive test cases for HuggingFace embeddings detection.
These patterns should be detected by the detect-embeddings-huggingface.yaml rule.
"""

# Core SentenceTransformers imports - VERY HIGH CONFIDENCE
from sentence_transformers import SentenceTransformer
from sentence_transformers import SentenceTransformer, util
import sentence_transformers
from sentence_transformers import util

# HuggingFace Transformers embedding imports - HIGH CONFIDENCE
from transformers import AutoTokenizer, AutoModel
from transformers import AutoModel
from transformers import pipeline

# SentenceTransformer model instantiation - VERY HIGH CONFIDENCE
model = SentenceTransformer("all-MiniLM-L6-v2")
embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
embedder = SentenceTransformer()

# Popular embedding models - HIGH CONFIDENCE
model1 = SentenceTransformer("all-MiniLM-L6-v2")
model2 = SentenceTransformer("all-mpnet-base-v2")
model3 = SentenceTransformer("paraphrase-MiniLM-L6-v2")
model4 = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")
custom_model = SentenceTransformer("sentence-transformers/custom-model")

# AutoModel embedding patterns - HIGH CONFIDENCE
bert_model = AutoModel.from_pretrained("bert-base-uncased")
embedding_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Embedding generation via encode() - VERY HIGH CONFIDENCE
embeddings = model.encode(texts)
text_embeddings = embedding_model.encode(["Hello world", "How are you?"])
embeddings = model.encode()
tensor_embeddings = model.encode(texts, convert_to_tensor=True)
progress_embeddings = model.encode(texts, show_progress_bar=False)

# SentenceTransformers utility functions - VERY HIGH CONFIDENCE
results = sentence_transformers.util.semantic_search(query_emb, corpus_emb)
search_results = util.semantic_search(query_embedding, doc_embeddings)
similarities = sentence_transformers.util.cos_sim(emb1, emb2)
cosine_scores = util.cos_sim(embeddings1, embeddings2)

# Model similarity methods - HIGH CONFIDENCE
similarity_scores = model.similarity(texts1, texts2)

# HuggingFace pipeline for embeddings - HIGH CONFIDENCE
feature_extractor = pipeline("feature-extraction")
extraction_pipeline = pipeline("feature-extraction", model="bert-base-uncased")
extractor = pipeline("feature-extraction")

# Model management - HIGH CONFIDENCE
loaded_model = SentenceTransformer.from_pretrained("custom-model")
model.save_pretrained("./my-embedding-model")

# Cross-encoder patterns - HIGH CONFIDENCE
from sentence_transformers import CrossEncoder
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
scores = cross_encoder.predict(sentence_pairs)

# Advanced embedding operations - MEDIUM CONFIDENCE
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
evaluator = EmbeddingSimilarityEvaluator(sentences1, sentences2, scores)
results = model.evaluate(evaluator)

# Real-world usage examples
def setup_embedding_pipeline():
    # Load embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Process documents
    documents = [
        "Machine learning is a subset of AI",
        "Deep learning uses neural networks",
        "Transformers revolutionized NLP"
    ]

    # Generate embeddings
    doc_embeddings = model.encode(documents, convert_to_tensor=True)

    # Query embedding
    query = "What is artificial intelligence?"
    query_embedding = model.encode([query])

    # Semantic search
    hits = util.semantic_search(query_embedding, doc_embeddings, top_k=3)

    return hits

def cross_encoder_reranking():
    from sentence_transformers import CrossEncoder

    # Load cross-encoder for reranking
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    # Candidate pairs
    pairs = [
        ['What is machine learning?', 'ML is a subset of AI'],
        ['What is machine learning?', 'Python is a programming language']
    ]

    # Get relevance scores
    scores = cross_encoder.predict(pairs)
    return scores

# Advanced patterns with transformers
def transformers_embeddings():
    from transformers import AutoModel, AutoTokenizer
    import torch

    # Load model and tokenizer
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    # Tokenize and get embeddings
    texts = ["Hello world", "Machine learning"]
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)

    return embeddings

# Embedding similarity analysis
class EmbeddingAnalyzer:
    def __init__(self):
        self.model = SentenceTransformer('all-mpnet-base-v2')

    def analyze_similarity(self, texts1, texts2):
        # Generate embeddings
        embeddings1 = self.model.encode(texts1)
        embeddings2 = self.model.encode(texts2)

        # Calculate similarities
        similarities = util.cos_sim(embeddings1, embeddings2)

        return similarities

    def get_embeddings(self, texts):
        return self.model.encode(texts, show_progress_bar=True)

# Multi-language embeddings
def multilingual_embeddings():
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    texts = [
        "Hello world",
        "Hola mundo",
        "Bonjour le monde"
    ]

    embeddings = model.encode(texts)
    similarities = model.similarity(texts, texts)

    return embeddings, similarities

# Batch processing with embeddings
def batch_embedding_processing():
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Large batch of documents
    documents = [f"Document {i} with content about AI" for i in range(1000)]

    # Process in batches
    batch_size = 32
    all_embeddings = []

    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        batch_embeddings = model.encode(batch, show_progress_bar=False)
        all_embeddings.extend(batch_embeddings)

    return all_embeddings