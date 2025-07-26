import os
import json
import faiss
import numpy as np
from openai import OpenAI
from typing import List, Dict, Tuple
import logging
import time # Import time for sleep

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')

class EmbeddingClient:
    """
    A client to interact with the Fireworks AI embedding API.
    """
    def __init__(self, api_key: str):
        self.client = OpenAI(
            base_url="https://api.fireworks.ai/inference/v1",
            api_key=api_key,
        )
        self.model = "nomic-ai/nomic-embed-text-v1.5" # Recommended embedding model

    def get_embedding(self, text: str, is_query: bool = False, dimensions: int = None) -> List[float]:
        """
        Generates an embedding for the given text using the configured model.
        Applies prefixes ('search_query:', 'search_document:') as required by Nomic models.
        """
        prefix = "search_query: " if is_query else "search_document: "
        input_text = prefix + text
        
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=input_text,
                dimensions=dimensions # Optional: for variable dimensions
            )
            return response.data[0].embedding
        except Exception as e:
            logging.error(f"Error getting embedding for text: '{input_text[:50]}...' Error: {e}")
            return []

class DocumentChunk:
    """
    Represents a chunk of text from a document, along with its metadata.
    """
    def __init__(self, content: str, source_file: str, chunk_id: int, original_data: Dict = None):
        self.content = content
        self.source_file = source_file
        self.chunk_id = chunk_id
        self.original_data = original_data # Store original document data for richer context if needed

    def to_dict(self):
        return {
            "content": self.content,
            "source_file": self.source_file,
            "chunk_id": self.chunk_id,
            "original_data": self.original_data
        }

def load_and_chunk_conversations(repo_path: str) -> List[DocumentChunk]:
    """
    Loads all JSON files from the 'knowledge' subdirectory and creates chunks.
    It now expects each JSON file to be a list containing objects with a 'content' field.
    """
    knowledge_dir = os.path.join(repo_path, 'knowledge')
    chunks: List[DocumentChunk] = []
    
    if not os.path.isdir(knowledge_dir):
        logging.error(f"Knowledge directory '{knowledge_dir}' not found for embedding.")
        return []

    for filename in os.listdir(knowledge_dir):
        if filename.endswith('.json'):
            filepath = os.path.join(knowledge_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # Expecting data to be a list of objects, each with a 'content' field
                    if isinstance(data, list):
                        for i, doc_entry in enumerate(data):
                            if isinstance(doc_entry, dict) and "content" in doc_entry and isinstance(doc_entry["content"], str):
                                chunk = DocumentChunk(
                                    content=doc_entry["content"],
                                    source_file=filename,
                                    chunk_id=i, # Use the index of the doc_entry within the file
                                    original_data=doc_entry # Store the whole document entry
                                )
                                chunks.append(chunk)
                            else:
                                logging.warning(f"Skipping malformed document entry in {filename}. Expected dict with 'content' string: {doc_entry}")
                    else:
                        logging.warning(f"Skipping non-list JSON file: {filepath}. Please ensure the root is a JSON array.")

            except json.JSONDecodeError:
                logging.error(f"Error decoding JSON from file: {filepath}. Please check its JSON syntax.")
            except Exception as e:
                logging.error(f"An unexpected error occurred while processing {filepath}: {e}")
    
    logging.debug(f"Loaded and chunked {len(chunks)} documents from '{knowledge_dir}'.")
    return chunks

def build_faiss_index(chunks: List[DocumentChunk], embedding_client: EmbeddingClient) -> Tuple[faiss.Index, List[DocumentChunk]]:
    """
    Generates embeddings for all chunks and builds a FAISS index.
    Returns the FAISS index and the list of chunks (preserving order).
    """
    if not chunks:
        logging.info("No chunks to build FAISS index from.")
        return None, []

    logging.debug(f"Building FAISS index for {len(chunks)} chunks...")
    embeddings = []
    valid_chunks = [] # Keep track of chunks for which embeddings were successfully generated

    for i, chunk in enumerate(chunks):
        embedding = embedding_client.get_embedding(chunk.content, is_query=False)
        if embedding:
            embeddings.append(embedding)
            valid_chunks.append(chunk)
        else:
            logging.warning(f"Could not get embedding for chunk {i} from file {chunk.source_file}. Skipping.")
        
        # Add a small delay to avoid hitting rate limits
        time.sleep(0.05) # Sleep for 50 milliseconds (adjust if needed)

    if not embeddings:
        logging.info("No valid embeddings generated. FAISS index cannot be built.")
        return None, []

    embeddings_np = np.array(embeddings).astype('float32')
    embedding_dimension = embeddings_np.shape[1]

    # Initialize a FAISS index (FlatL2 for L2 distance, suitable for similarity search)
    index = faiss.IndexFlatL2(embedding_dimension)
    index.add(embeddings_np) # Add the embeddings to the index

    logging.debug(f"FAISS index built with {index.ntotal} vectors.")
    return index, valid_chunks

def search_faiss_index(query_text: str, faiss_index: faiss.Index, document_chunks: List[DocumentChunk], embedding_client: EmbeddingClient, k: int = 3) -> List[DocumentChunk]:
    """
    Searches the FAISS index for the top-k most similar document chunks to the query.
    Returns the list of relevant DocumentChunk objects.
    """
    if faiss_index is None or not document_chunks:
        logging.warning("FAISS index not initialized or no document chunks available.")
        return []

    logging.debug(f"Searching FAISS index for query: '{query_text}' (top {k} results)")
    query_embedding = embedding_client.get_embedding(query_text, is_query=True)
    if not query_embedding:
        logging.error("Could not get embedding for the query.")
        return []

    query_embedding_np = np.array([query_embedding]).astype('float32')

    # Perform the search
    distances, indices = faiss_index.search(query_embedding_np, k) # D: distances, I: indices

    relevant_chunks: List[DocumentChunk] = []
    for i, idx in enumerate(indices[0]):
        if idx != -1: # Ensure a valid index was returned
            # Retrieve the original DocumentChunk object using the index
            relevant_chunks.append(document_chunks[idx])
            logging.debug(f"Found relevant chunk (distance: {distances[0][i]:.4f}) from {document_chunks[idx].source_file}, chunk_id: {document_chunks[idx].chunk_id}")
    
    return relevant_chunks

if __name__ == "__main__":
    # This block is for independent testing of embedding_search_util.py
    from dotenv import load_dotenv
    load_dotenv() # Load env vars for testing

    FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")
    if not FIREWORKS_API_KEY:
        print("FIREWORKS_API_KEY not set. Cannot run standalone test.")
    else:
        # Define a dummy REPO_PATH for testing purposes
        test_repo_path = os.path.abspath(os.path.dirname(__file__))

        # Create a dummy knowledge directory and file for testing if they don't exist
        test_knowledge_dir = os.path.join(test_repo_path, 'knowledge')
        if not os.path.exists(test_knowledge_dir):
            os.makedirs(test_knowledge_dir)
            logging.info(f"Created test knowledge directory: {test_knowledge_dir}")
        
        test_json_file = os.path.join(test_knowledge_dir, 'test_conversation.json')
        if not os.path.exists(test_json_file):
            dummy_data = [
                {"content": "I need help with my new project management system."},
                {"content": "What specific issues are you facing with the project management system?"},
                {"content": "It's about task assignment and tracking progress."},
                {"content": "Okay, I can help with task assignment. Do you have a document on best practices for project management?"},
                {"content": "Tell me about the Q3 marketing budget discussion."},
                {"content": "The Q3 marketing budget discussion covered allocations for digital campaigns and influencer partnerships."},
                {"content": "What was the main topic of the last team meeting?"},
                {"content": "The last team meeting focused on quarterly goals and upcoming client presentations."},
            ]
            with open(test_json_file, 'w', encoding='utf-8') as f:
                json.dump(dummy_data, f, indent=4)
            logging.info(f"Created dummy test conversation file: {test_json_file}")


        embedding_client = EmbeddingClient(FIREWORKS_API_KEY)

        # 1. Load and Chunk Documents
        all_chunks = load_and_chunk_conversations(test_repo_path)

        # 2. Build FAISS Index
        faiss_index, indexed_chunks = build_faiss_index(all_chunks, embedding_client)

        if faiss_index and indexed_chunks:
            # 3. Test Search
            query = "What was discussed about the marketing budget?"
            relevant_docs = search_faiss_index(query, faiss_index, indexed_chunks, embedding_client, k=2)

            print(f"\n--- Search Results for '{query}' ---")
            if relevant_docs:
                for doc in relevant_docs:
                    print(f"Source: {doc.source_file}, Chunk ID: {doc.chunk_id}")
                    print(f"Content: {doc.content[:100]}...")
                    print("-" * 20)
            else:
                print("No relevant documents found.")

            query_2 = "Tell me about project management best practices."
            relevant_docs_2 = search_faiss_index(query_2, faiss_index, indexed_chunks, embedding_client, k=1)
            print(f"\n--- Search Results for '{query_2}' ---")
            if relevant_docs_2:
                for doc in relevant_docs_2:
                    print(f"Source: {doc.source_file}, Chunk ID: {doc.chunk_id}")
                    print(f"Content: {doc.content[:100]}...")
                    print("-" * 20)
            else:
                print("No relevant documents found.")
        else:
            print("FAISS index or chunks not successfully initialized. Cannot run search tests.")