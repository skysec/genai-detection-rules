#!/usr/bin/env python3
"""
Positive test cases for LangChain embeddings detection.
These patterns should be detected by the detect-embeddings-langchain.yaml rule.
"""

# Core LangChain embedding imports - VERY HIGH CONFIDENCE
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import OpenAIEmbeddings as LegacyOpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings as HFEmbeddings
from langchain_cohere import CohereEmbeddings

# Additional embedding imports
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.embeddings import BedrockEmbeddings
from langchain.embeddings import CohereEmbeddings as LegacyCohereEmbeddings

# Vector store imports - HIGH CONFIDENCE
from langchain_chroma import Chroma
from langchain_pinecone import Pinecone
from langchain_weaviate import Weaviate
from langchain_community.vectorstores import FAISS

# Embedding model instantiation - VERY HIGH CONFIDENCE
embeddings = OpenAIEmbeddings()
embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")
hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
cohere_embeddings = CohereEmbeddings()

# Different embedding providers
openai_emb = OpenAIEmbeddings(openai_api_key="sk-...")
cohere_emb = CohereEmbeddings(cohere_api_key="...")
hf_emb = HFEmbeddings()

# Explicit embedding usage - VERY HIGH CONFIDENCE
document_embeddings = embeddings.embed_documents(["Hello world", "How are you?"])
query_embedding = embeddings.embed_query("What is machine learning?")
embedded_docs = embedding_model.embed_documents(documents)
embedded_query = embedding_model.embed_query(query)

# Vector store creation with embeddings - HIGH CONFIDENCE
vectorstore = Chroma.from_documents(documents, embeddings)
pinecone_store = Pinecone.from_documents(docs, embedding_model)
weaviate_store = Weaviate.from_documents(documents, embeddings)
faiss_store = FAISS.from_documents(documents, embeddings)

# Vector store with explicit embedding parameter
chroma_db = Chroma.from_documents(documents, embeddings=embeddings)
pinecone_db = Pinecone.from_documents(docs, embeddings=hf_embeddings)

# Embedding function parameter usage - HIGH CONFIDENCE
vectorstore = Chroma(
    collection_name="my_collection",
    embedding_function=embeddings
)

db = FAISS(
    index=faiss_index,
    docstore=docstore,
    embeddings=embedding_model
)

# Real-world usage examples
def setup_langchain_embeddings():
    # Initialize OpenAI embeddings
    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        openai_api_key="sk-..."
    )

    # Process documents
    documents = [
        "LangChain is a framework for building applications with LLMs",
        "Vector databases store high-dimensional vectors",
        "Embeddings capture semantic meaning of text"
    ]

    # Generate embeddings
    doc_embeddings = embeddings.embed_documents(documents)

    # Create vector store
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name="langchain_demo"
    )

    return vectorstore

def huggingface_langchain_example():
    # HuggingFace embeddings via LangChain
    hf_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': False}
    )

    # Embed documents
    texts = ["Machine learning", "Deep learning", "Natural language processing"]
    embeddings = hf_embeddings.embed_documents(texts)

    # Create FAISS vector store
    vectorstore = FAISS.from_documents(
        documents=texts,
        embedding=hf_embeddings
    )

    return embeddings, vectorstore

def cohere_embeddings_example():
    # Cohere embeddings
    cohere_emb = CohereEmbeddings(
        model="embed-english-v3.0",
        cohere_api_key="..."
    )

    # Embed query and documents
    query = "What is artificial intelligence?"
    query_embedding = cohere_emb.embed_query(query)

    documents = ["AI is machine intelligence", "ML is a subset of AI"]
    doc_embeddings = cohere_emb.embed_documents(documents)

    return query_embedding, doc_embeddings

# Advanced patterns with vector stores
class EmbeddingPipeline:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = None

    def create_vectorstore(self, documents):
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings
        )

    def similarity_search(self, query, k=5):
        return self.vectorstore.similarity_search(query, k=k)

    def embed_and_store(self, texts):
        # Embed documents
        embeddings = self.embeddings.embed_documents(texts)

        # Store in vector database
        self.vectorstore = FAISS.from_documents(
            documents=texts,
            embedding=self.embeddings
        )

# Multiple embedding providers
def multi_provider_setup():
    providers = {
        'openai': OpenAIEmbeddings(),
        'cohere': CohereEmbeddings(),
        'huggingface': HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    }

    vectorstores = {}
    documents = ["Sample document 1", "Sample document 2"]

    for name, embedding in providers.items():
        # Create embeddings
        doc_embs = embedding.embed_documents(documents)
        query_emb = embedding.embed_query("test query")

        # Create vector store
        vectorstores[name] = Chroma.from_documents(
            documents=documents,
            embedding=embedding
        )

    return vectorstores

# Retrieval Augmented Generation (RAG) setup
def rag_with_embeddings():
    from langchain.chains import RetrievalQA
    from langchain_openai import OpenAI

    # Setup embeddings
    embeddings = OpenAIEmbeddings()

    # Create vector store from documents
    documents = ["Document 1 content", "Document 2 content"]
    vectorstore = Chroma.from_documents(documents, embeddings)

    # Create retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Setup QA chain
    llm = OpenAI()
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )

    return qa_chain

# Batch processing with embeddings
def batch_embedding_processing():
    embeddings = HuggingFaceEmbeddings()

    # Large batch of documents
    documents = [f"Document {i} about machine learning" for i in range(100)]

    # Process in batches
    batch_size = 10
    all_embeddings = []

    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        batch_embeddings = embeddings.embed_documents(batch)
        all_embeddings.extend(batch_embeddings)

    # Create vector store from all embeddings
    vectorstore = FAISS.from_documents(
        documents=documents,
        embedding=embeddings
    )

    return vectorstore

# Custom embedding configurations
custom_openai = OpenAIEmbeddings(
    model="text-embedding-3-small",
    dimensions=512,
    openai_api_key="sk-custom-key"
)

custom_hf = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    model_kwargs={'device': 'cuda'},
    encode_kwargs={'batch_size': 32}
)

# Legacy imports (should still be detected)
from langchain.embeddings.openai import OpenAIEmbeddings as LegacyOpenAI
legacy_embeddings = LegacyOpenAI()