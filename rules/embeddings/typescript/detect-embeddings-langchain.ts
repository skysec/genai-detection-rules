// POSITIVE TEST CASES - Should be detected by the rule

// Embedding imports - SHOULD MATCH
import { OpenAIEmbeddings } from "@langchain/openai";
import { HuggingFaceInferenceEmbeddings } from "@langchain/community/embeddings/hf";
import { CohereEmbeddings } from "@langchain/cohere";

// Vector store imports - SHOULD MATCH
import { Chroma } from "@langchain/community/vectorstores/chroma";
import { Pinecone } from "@langchain/pinecone";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";

// Embedding instantiation - SHOULD MATCH
const embeddings = new OpenAIEmbeddings({
    openAIApiKey: process.env.OPENAI_API_KEY,
    modelName: "text-embedding-ada-002"
});

const huggingFaceEmbeddings = new HuggingFaceInferenceEmbeddings({
    model: "sentence-transformers/all-MiniLM-L6-v2",
    apiKey: process.env.HUGGINGFACE_API_KEY
});

const cohereEmbeddings = new CohereEmbeddings({
    apiKey: process.env.COHERE_API_KEY,
    model: "embed-english-v2.0"
});

// Vector store creation with embeddings - SHOULD MATCH
const vectorStore = await Chroma.fromDocuments(
    documents,
    embeddings,
    {
        collectionName: "my_collection",
        url: "http://localhost:8000"
    }
);

const pineconeStore = await Pinecone.fromDocuments(
    docs,
    embeddings,
    {
        pineconeIndex: index,
        namespace: "test"
    }
);

// Embedding usage patterns - SHOULD MATCH
const docEmbeddings = await embeddings.embedDocuments([
    "Document 1 content",
    "Document 2 content"
]);

const queryEmbedding = await embeddings.embedQuery("What is AI?");

const vectors = embeddings.embedDocuments(textChunks);
const queryVector = embeddings.embedQuery(userQuery);

// Vector store initialization with embeddings - SHOULD MATCH
const chromaStore = new Chroma({
    embeddings: embeddings,
    collectionName: "documents",
    url: "http://localhost:8000"
});

// Similarity search (implies embeddings) - SHOULD MATCH
const similarDocs = await vectorStore.similaritySearch("AI applications", 5);
const searchResults = await vectorStore.similaritySearchWithScore("machine learning");
const results = vectorStore.similaritySearch(query, 3);

// Document processing for embeddings - SHOULD MATCH
const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200
});

const splitDocs = await textSplitter.splitDocuments(documents);

// Embedding-based retrieval - SHOULD MATCH
const retriever = vectorStore.asRetriever();
const configuredRetriever = vectorStore.asRetriever({
    searchKwargs: { k: 5 }
});

// Vector store configuration - SHOULD MATCH
const vectorStoreConfig = {
    collectionName: "my_documents",
    textKey: "content",
    metadataKey: "metadata",
    embeddings: embeddings
};

// Function using embeddings - SHOULD MATCH
async function createEmbeddingBasedRetrieval() {
    const embeddings = new OpenAIEmbeddings();

    const vectorStore = await Chroma.fromDocuments(
        documents,
        embeddings,
        { collectionName: "knowledge_base" }
    );

    return vectorStore.asRetriever({ searchKwargs: { k: 3 } });
}

// RAG implementation - SHOULD MATCH
async function ragQuery(question: string) {
    const retriever = vectorStore.asRetriever();
    const relevantDocs = await retriever.getRelevantDocuments(question);

    return relevantDocs;
}

// NEGATIVE TEST CASES - Should NOT be detected

// Regular imports
import { Document } from "@langchain/core/documents";
import { BaseMessage } from "@langchain/core/messages";
import { OpenAI } from "@langchain/openai";

// Regular database operations
import { DatabaseConnection } from "./database";
import { UserRepository } from "./repositories/user";

const db = new DatabaseConnection({
    host: "localhost",
    port: 5432,
    database: "myapp"
});

// Regular search operations
const searchResults = await database.search("users", { name: "John" });
const filteredData = items.filter(item => item.category === "tech");

// Regular text processing
const processedText = text.toLowerCase().trim();
const words = sentence.split(" ");

// Regular API calls
const response = await fetch("/api/data");
const json = await response.json();

// Regular array operations
const documents = ["doc1", "doc2", "doc3"];
const queries = ["query1", "query2"];

// Regular class instantiation
const service = new DataService({
    apiKey: process.env.API_KEY,
    baseUrl: "https://api.example.com"
});