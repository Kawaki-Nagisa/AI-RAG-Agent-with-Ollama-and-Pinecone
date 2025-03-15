from langchain.memory.buffer import ConversationBufferMemory
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import CSVLoader
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os
import time
import logging
import hashlib

# Constants
INDEX_NAME = "test-w"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 250
EMBEDDING_DIMENSION = 768
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "deepseek-r1:1.5b"
TEMPERATURE = 0.4

# Configure logging
logging.basicConfig(level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s")

# Load environment variables
def load_env_vars():
    load_dotenv()
    pine_key = os.getenv("PINECONE_API_KEY")
    if not pine_key:
        logging.error("Pinecone API key not found in environment variables")
        raise ValueError("Pinecone API key not found in environment variables")
    return pine_key

# Initialize Pinecone
def initialize_pinecone(api_key):
    print("Initializing Pinecone...")
    pc = Pinecone(api_key=api_key)
    return pc

# Create or validate Pinecone index
def setup_pinecone_index(pc, index_name):
    try:
        print(f"Checking if Pinecone index '{index_name}' exists...")
        pc.describe_index(index_name)
    except Exception:
        print(f"Index '{index_name}' not found. Creating a new index...")
        pc.create_index(
            name=index_name,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        print(f"Index '{index_name}' created successfully.")

# Load documents
def load_documents(file_path):
    print(f"Loading Provided file: {file_path}...")
    loader = CSVLoader(file_path)
    start_time = time.time()
    documents = loader.load()
    print(f"Loaded {len(documents)} records in {time.time() - start_time:.2f} seconds.")
    return documents

# Split documents into chunks
def split_documents(pages):
    print("Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False
    )
    start_time = time.time()
    chunks = splitter.split_documents(pages)
    print(f"Split into {len(chunks)} chunks in {time.time() - start_time:.2f} seconds.")
    return chunks

# Generate embeddings for chunks
def create_embeddings(chunks, embedding_model):
    print("Creating embeddings for chunks...")
    embeddings = OllamaEmbeddings(model=embedding_model)
    start_time = time.time()
    vectors = embeddings.embed_documents([chunk.page_content for chunk in chunks])
    print(f"Created {len(vectors)} embeddings in {time.time() - start_time:.2f} seconds.")
    return vectors, embeddings

# Upload chunks to Pinecone Vector Store
def upload_to_vector_store(pc, index_name, embeddings, chunks):
    print("Uploading documents to Pinecone Vector Store...")
    vector_store = PineconeVectorStore(index=pc.Index(index_name), embedding=embeddings)
    texts = [chunk.page_content for chunk in chunks]
    ids = [str(i) for i in range(1, len(texts) + 1)]
    start_time = time.time()
    vector_store.add_documents(documents=chunks, ids=ids)
    print(f"Uploaded {len(chunks)} documents in {time.time() - start_time:.2f} seconds.")
    return vector_store

# Generate response using RAG
def generate_rag_response(vector_store, embeddings, query, memory):
    embedded_query = embeddings.embed_query(query)
    results = vector_store._similarity_search_with_relevance_scores(query=query, k=5)
    
    rag_response = "\n".join(
        f"Context Chunk {idx + 1} (Relevance Score: {score:.2f}):\n{chunk.page_content}\n"
        for idx, (chunk, score) in enumerate(results)
    )
    history = "\n".join([msg.content for msg in memory.chat_memory.messages])
    prompt = (
        f"You are an intelligent, helpful AI chat assistant using retrieval-augmented generation (RAG). "
        f"Carefully analyze the retrieved context and conversation history to provide an accurate response.\n\n"
        f"Skip Printing the Thinking Logic and keep your answers short simple and to the point with a focus on user understanding.\n"
        f"Retrieved Context:\n{rag_response}\n\n"
        f"Conversation History:\n{history}\n\n"
        f"User Query: {query}\n\n"
        f"Provide a comprehensive and precise response."
    )
    llm = ChatOllama(model=LLM_MODEL, temperature=TEMPERATURE)
    response = llm.invoke(prompt)
    memory.chat_memory.add_user_message(query)
    memory.chat_memory.add_ai_message(response.content)
    return response

def check_index_empty(pc, index_name):
    """Check if the Pinecone index is empty"""
    try:
        index = pc.Index(index_name)
        stats = index.describe_index_stats()
        return stats.total_vector_count == 0
    except Exception as e:
        logging.error(f"Error checking index: {e}")
        return True

def initialize_embeddings():
    """Initialize embeddings model once"""
    print("Initializing embeddings model...")
    return OllamaEmbeddings(model=EMBEDDING_MODEL)

# Main function
def main():
    # Load environment variables
    pine_key = load_env_vars()

    # Initialize Pinecone
    pc = initialize_pinecone(pine_key)

    # Setup Pinecone index
    setup_pinecone_index(pc, INDEX_NAME)

    # Initialize embeddings model (only once)
    embeddings = initialize_embeddings()

    # Check if documents already exist in the index
    if check_index_empty(pc, INDEX_NAME):
        print("Index is empty. Processing and uploading documents...")
        # Load and process documents
        file_path = "restaurants.csv"
        documents = load_documents(file_path)
        chunks = split_documents(documents)
        
        # Create embeddings
        vectors = embeddings.embed_documents([chunk.page_content for chunk in chunks])

        # Upload to vector store
        vector_store = upload_to_vector_store(pc, INDEX_NAME, embeddings, chunks)
    else:
        print("Documents already exist in the index. Skipping processing and upload.")
        vector_store = PineconeVectorStore(index=pc.Index(INDEX_NAME), embedding=embeddings)

    # Set up memory and handle user query
    memory = ConversationBufferMemory(return_messages=True)
    query = "Give me a list of possible rates for Biryani with respect to delivcery time"
    response = generate_rag_response(vector_store, embeddings, query, memory)

    # Display response
    print("--- AI Response ---")
    print(response.content)
    print("--- End of AI Response ---")

if __name__ == "__main__":
    main()
