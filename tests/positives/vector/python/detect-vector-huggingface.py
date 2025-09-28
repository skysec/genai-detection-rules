#!/usr/bin/env python3
"""
Positive test cases for HuggingFace vector/embedding detection.
These patterns should be detected by the detect-vector-huggingface.yaml rule.
"""

# Core SentenceTransformers imports - VERY HIGH CONFIDENCE
from sentence_transformers import SentenceTransformer
from sentence_transformers import SentenceTransformer, util
import sentence_transformers
from sentence_transformers import util
from sentence_transformers.util import semantic_search
from sentence_transformers.util import cos_sim

# HuggingFace Transformers embedding patterns - HIGH CONFIDENCE
from transformers import AutoTokenizer, AutoModel
from transformers import AutoModel
model = AutoModel.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# SentenceTransformer instantiation - VERY HIGH CONFIDENCE
model = SentenceTransformer("all-MiniLM-L6-v2")
embedding_model = SentenceTransformer("all-mpnet-base-v2")
st_model = SentenceTransformer()

# Embedding generation - HIGH CONFIDENCE
embeddings = model.encode(sentences)
text_embeddings = embedding_model.encode(["Hello world", "How are you?"])
embeddings = model.encode()

# HuggingFace specific vector operations - VERY HIGH CONFIDENCE
results = semantic_search(query_embedding, corpus_embeddings)
scores = util.semantic_search(query_emb, doc_embeddings)
similarity = sentence_transformers.util.semantic_search(q_emb, c_emb)
cosine_scores = cos_sim(emb1, emb2)
similarity_matrix = util.cos_sim(embeddings1, embeddings2)

# Popular HuggingFace embedding models - HIGH CONFIDENCE
model1 = SentenceTransformer("all-MiniLM-L6-v2")
model2 = SentenceTransformer("all-mpnet-base-v2")
model3 = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")
model4 = SentenceTransformer("paraphrase-MiniLM-L6-v2")
custom_model = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")

# HuggingFace Hub model loading - HIGH CONFIDENCE
loaded_model = SentenceTransformer.from_pretrained("custom-model")
model.save_pretrained("./my-model")

# Sentence-transformers specific operations - HIGH CONFIDENCE
similarities = model.similarity(texts1, texts2)
embeddings = model.encode(texts, convert_to_tensor=True)
embeddings = model.encode(texts, show_progress_bar=False)

# Cross-encoder patterns (HuggingFace) - HIGH CONFIDENCE
from sentence_transformers import CrossEncoder
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
scores = cross_encoder.predict(sentence_pairs)

# HuggingFace embedding evaluation - MEDIUM CONFIDENCE
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
evaluator = EmbeddingSimilarityEvaluator(sentences1, sentences2, scores)
results = model.evaluate(evaluator)

# Real-world usage examples
def setup_semantic_search():
    # Load model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Documents to search
    documents = [
        "The cat sat on the mat",
        "Python is a programming language",
        "Machine learning is fascinating"
    ]

    # Create embeddings
    doc_embeddings = model.encode(documents)

    # Query
    query = "What is Python?"
    query_embedding = model.encode([query])

    # Search
    hits = util.semantic_search(query_embedding, doc_embeddings, top_k=3)
    return hits

def cross_encoder_reranking():
    from sentence_transformers import CrossEncoder

    # Load cross-encoder for reranking
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    # Candidate pairs
    pairs = [['Query', 'Passage 1'], ['Query', 'Passage 2']]

    # Get relevance scores
    scores = cross_encoder.predict(pairs)
    return scores

# Advanced HuggingFace patterns
class SemanticSearchEngine:
    def __init__(self):
        self.model = SentenceTransformer('all-mpnet-base-v2')
        self.index = None

    def build_index(self, documents):
        embeddings = self.model.encode(documents, convert_to_tensor=True)
        self.index = embeddings

    def search(self, query, top_k=5):
        query_embedding = self.model.encode([query])
        hits = util.semantic_search(query_embedding, self.index, top_k=top_k)
        return hits[0]

    def similarity(self, text1, text2):
        return self.model.similarity([text1], [text2])

# Model fine-tuning patterns
def fine_tune_model():
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Evaluation during training
    evaluator = EmbeddingSimilarityEvaluator(sentences1, sentences2, scores)
    model.evaluate(evaluator)

    return model