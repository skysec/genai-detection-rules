#!/usr/bin/env python3
"""
Positive test cases for LangChain vector/retrieval detection.
These patterns should be detected by the detect-vector-langchain.yaml rule.
"""

# Modern vector store imports - VERY HIGH CONFIDENCE
from langchain_chroma import Chroma
from langchain_pinecone import PineconeVectorStore
from langchain_weaviate import WeaviateVectorStore
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Chroma as CommunityChroma
from langchain_community.vectorstores import Pinecone
from langchain.vectorstores import FAISS as LegacyFAISS

# Vector store creation
vectorstore = Chroma.from_documents(documents, embeddings)
faiss_store = FAISS.from_documents(docs, embedding_function)
vectorstore = Chroma(collection_name="test", embedding_function=embeddings)

# Similarity search
results = vectorstore.similarity_search("What is AI?")
top_docs = vectorstore.similarity_search(query, k=5)
scored_results = vectorstore.similarity_search_with_score(query)
relevant_docs = vectorstore.similarity_search_with_relevance_scores(query, k=3)
results = vectorstore.similarity_search()

# Retriever patterns
retriever = vectorstore.as_retriever()
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
retriever = vectorstore.as_retriever()
search_kwargs = {"k": 10}

# Modern RAG chain imports and creation - HIGH CONFIDENCE
from langchain.chains import RetrievalQA
from langchain.chains.retrieval_qa import RetrievalQA
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
qa = RetrievalQA.from_chain_type()
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
retrieval_chain = create_retrieval_chain(retriever, document_chain)
doc_chain = create_stuff_documents_chain(llm, prompt)

# RAG execution
answer = qa.run("What is machine learning?")
response = qa.run()
response = qa.run()

# Document loader for RAG
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
loader = TextLoader("document.txt")
documents = loader.load()

# Text splitter for chunking
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

# Vector store persistence
vectorstore.persist()
vectorstore.save_local("./vector_store")
vectorstore = Chroma.from_documents(documents, embeddings, persist_directory="./chroma_db")

# Advanced retrieval patterns - MEDIUM CONFIDENCE (remove generic string patterns)
retriever = vectorstore.as_retriever()
similarity_retriever = vectorstore.as_retriever(search_type="similarity")
mmr_retriever = vectorstore.as_retriever(search_type="mmr")
stuff_qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff")
map_reduce_qa = RetrievalQA.from_chain_type(llm=llm, chain_type="map_reduce")

# Real-world RAG implementation
def setup_rag_pipeline():
    from langchain.document_loaders import TextLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain.chains import RetrievalQA
    from langchain_openai import OpenAIEmbeddings, OpenAI

    # Load documents
    loader = TextLoader("data.txt")
    documents = loader.load()

    # Split text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    texts = text_splitter.split_documents(documents)

    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(texts, embeddings)

    # Create retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Create QA chain
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        chain_type="stuff",
        retriever=retriever
    )

    return qa

# Multiple vector stores
def multi_vectorstore_setup():
    from langchain_chroma import Chroma
    from langchain_pinecone import PineconeVectorStore
    from langchain_community.vectorstores import FAISS

    # Different vector stores
    chroma_store = Chroma.from_documents(docs, embeddings)
    pinecone_store = PineconeVectorStore.from_documents(docs, embeddings)
    faiss_store = FAISS.from_documents(docs, embeddings)

    # Similarity searches
    chroma_results = chroma_store.similarity_search(query)
    pinecone_results = pinecone_store.similarity_search(query, k=5)
    faiss_results = faiss_store.similarity_search_with_score(query)

    return chroma_results, pinecone_results, faiss_results

# Advanced retrieval patterns
class CustomRetriever:
    def __init__(self, vectorstore):
        self.retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "lambda_mult": 0.25}
        )

    def get_relevant_documents(self, query):
        return self.retriever.get_relevant_documents(query)

# Modern LangChain patterns (v0.1+)
def modern_langchain_rag():
    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain_core.prompts import ChatPromptTemplate

    # Create prompt
    prompt = ChatPromptTemplate.from_template("""
    Answer the following question based on the provided context:
    <context>{context}</context>
    Question: {input}
    """)

    # Create document chain
    document_chain = create_stuff_documents_chain(llm, prompt)

    # Create retrieval chain
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    return retrieval_chain

# Vector store with metadata filtering
def metadata_filtering():
    from langchain_community.vectorstores import Chroma

    vectorstore = Chroma.from_documents(
        documents,
        embeddings,
        metadatas=[{"source": "doc1"}, {"source": "doc2"}]
    )

    # Search with metadata filter
    results = vectorstore.similarity_search(
        query,
        filter={"source": "doc1"}
    )

    return results

# Async patterns
async def async_rag():
    from langchain.chains import RetrievalQA

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )

    result = await qa.arun("What is the capital of France?")
    return result