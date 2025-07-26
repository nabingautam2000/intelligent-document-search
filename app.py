import os
import json
import re
from openai import OpenAI
from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from typing import List, Dict
import logging
import datetime # Import datetime for timestamps

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')

load_dotenv()

app = Flask(__name__, static_folder='public')
CORS(app)

FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")
REPO_PATH = "C:\\Users\\nabin\\OneDrive\\Desktop\\BOOT" # Ensure this path is correct for your system

logging.debug(f"REPO_PATH currently set to: '{REPO_PATH}' (Hardcoded)")

if not FIREWORKS_API_KEY:
    raise ValueError("FIREWORKS_API_KEY environment variable not set. Please set it in your .env file.")
if not os.path.exists(REPO_PATH):
    logging.warning(f"REPO_PATH '{REPO_PATH}' does not exist. File search functionality may be impaired.")

client = OpenAI(
    base_url="https://api.fireworks.ai/inference/v1",
    api_key=FIREWORKS_API_KEY,
)

# Initialize EmbeddingClient (ensure embedding_search_util.py is correct)
from embedding_search_util import EmbeddingClient, load_and_chunk_conversations, build_faiss_index, search_faiss_index, DocumentChunk
embedding_client = EmbeddingClient(FIREWORKS_API_KEY)

faiss_index = None
indexed_document_chunks = []

CONVERSATION_FILE = os.path.join(REPO_PATH, 'public', "chat_completions.json")

tools = [
    {
        "type": "function",
        "function": {
            "name": "search_filenames",
            "description": "Searches for files in the repository whose names match a given search term, and ranks them by suitability. Returns a list of matching filenames. **Use this tool if the user asks to 'list files' or 'find files named X' or 'show documents about Y' where Y is a filename-like term, or 'what files are there in X folder'.**",
            "parameters": {
                "type": "object",
                "properties": {
                    "search_term": {
                        "type": "string",
                        "description": "The term to search for within file names (e.g., 'project plan', 'budget report', 'travel', 'knowledge')."
                    }
                },
                "required": ["search_term"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_file_content",
            "description": "Searches for a specific string within the content of files in the repository. Returns a list of filenames that contain the string. **Use this tool if the user asks 'Are there files mentioning X' or 'find documents containing Y' or 'what files contain Z'.**",
            "parameters": {
                "type": "object",
                "properties": {
                    "search_string": {
                        "type": "string",
                        "description": "The string to search for inside the file content (e.g., 'artificial intelligence', 'financial data', 'marketing budget')."
                    }
                },
                "required": ["search_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "semantic_search_knowledge_base",
            "description": "Performs a semantic search over the knowledge base (JSON conversation files) to find relevant information. Use this for questions that require understanding the meaning of the query rather than just keyword matching, especially if the user asks for 'information about X', 'details on Y', 'summarize Z', 'advice on A', or 'tips for B' from the database/knowledge base/files. It should return a concise, grounded answer from the relevant content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The user's query for semantic search (e.g., 'What are the main takeaways from the recent marketing meeting?', 'Tell me about the new client proposal details', 'What are advantages of adults learning guitar?')."
                    }
                },
                "required": ["query"],
            },
        },
    }
]

# Assuming search_files and repo_search are imported and available
import search_files
import repo_search

def execute_search_filenames(search_term: str) -> list:
    logging.debug(f"Entering execute_search_filenames for term: '{search_term}'")
    try:
        all_files = []
        for root, dirs, filenames in os.walk(REPO_PATH):
            for fname in filenames:
                relative_path = os.path.relpath(os.path.join(root, fname), REPO_PATH)
                all_files.append(relative_path)
        
        all_files = list(set(all_files))

        logging.debug(f"Found {len(all_files)} files in REPO_PATH (including subfolders).")
        if not all_files:
            logging.debug("No files found in REPO_PATH by os.walk.")
            return []

        file_scores = {file: search_files.calculate_suitability_score(file, search_term) for file in all_files}
        sorted_files = sorted(file_scores.items(), key=lambda x: (-x[1][0], -x[1][1], x[0]))
        found_files = [file for file, (score, _) in sorted_files if score > 0]
        logging.debug(f"search_filenames found: {found_files}")
        return found_files
    except Exception as e:
        logging.error(f"Error executing search_filenames: {e}")
        return []

def execute_search_file_content(search_string: str) -> list:
    logging.debug(f"Entering execute_search_file_content for string: '{search_string}'")
    try:
        found_files = repo_search.search_string_in_repo(REPO_PATH, search_string)
        logging.debug(f"search_file_content found: {found_files}")
        return found_files
    except Exception as e:
        logging.error(f"Error executing search_file_content: {e}")
        return []

def execute_semantic_search_knowledge_base(query: str) -> List[Dict]:
    global faiss_index, indexed_document_chunks
    logging.debug(f"Entering execute_semantic_search_knowledge_base for query: '{query}'")
    if faiss_index is None or not indexed_document_chunks:
        logging.warning("FAISS index not initialized. Attempting to re-initialize.")
        all_chunks = load_and_chunk_conversations(REPO_PATH)
        faiss_index, indexed_document_chunks = build_faiss_index(all_chunks, embedding_client)
        if faiss_index is None:
            logging.error("FAISS index still not initialized after re-attempt.")
            return []

    relevant_chunks = search_faiss_index(query, faiss_index, indexed_document_chunks, embedding_client, k=3)
    
    formatted_results = []
    for chunk in relevant_chunks:
        formatted_results.append({
            "content": chunk.content,
            "source_file": chunk.source_file,
            "chunk_id": chunk.chunk_id
        })
    logging.debug(f"Semantic search found {len(formatted_results)} relevant chunks.")
    return formatted_results


def load_conversation() -> List[Dict]:
    """Loads conversation history from the JSON file."""
    if os.path.exists(CONVERSATION_FILE):
        try:
            with open(CONVERSATION_FILE, "r", encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            logging.warning(f"{CONVERSATION_FILE} is corrupt or empty. Starting new conversation.")
            return []
    return []

def save_conversation(conversation_history: List[Dict]):
    """Saves the current conversation history to the JSON file."""
    try:
        os.makedirs(os.path.dirname(CONVERSATION_FILE), exist_ok=True)
        with open(CONVERSATION_FILE, "w", encoding='utf-8') as f:
            json.dump(conversation_history, f, indent=4)
    except Exception as e:
            logging.error(f"Error saving conversation: {e}")

# Define the system message once
SYSTEM_MESSAGE = {"role": "system", "content": "You are a helpful assistant. You have access to file search tools to find relevant documents. Use these tools when a user's query can be answered by searching your local files. When answering based on retrieved information, provide a concise summary or answer using only the details found in the provided context. If the retrieved information is relevant but doesn't fully answer the query, or if the user's phrasing needs a slight interpretation of the context, do your best to summarize the relevant parts. If no relevant information is found, state 'There is no information available in the database regarding that query.' For general questions not explicitly requesting a database search, use your comprehensive general knowledge."}

# Load conversation history and ensure system message is correctly set on startup
conversation_history_backend = load_conversation()
if not conversation_history_backend or conversation_history_backend[0].get("role") != "system" or conversation_history_backend[0].get("content") != SYSTEM_MESSAGE["content"]:
    # If history is empty, missing system message, or system message content is outdated, reset it
    conversation_history_backend = [SYSTEM_MESSAGE]
    save_conversation(conversation_history_backend)

def _clean_messages_for_api(messages_list: List[Dict]) -> List[Dict]:
    """
    Cleans a list of message dictionaries, ensuring only 'role', 'content',
    and 'tool_calls' (if present and valid) are included for API calls.
    Removes 'id' and 'timestamp' fields which are for local history management.
    """
    cleaned_messages = []
    for msg in messages_list:
        cleaned_msg = {"role": msg["role"], "content": msg["content"]}
        # Only include tool_calls if role is 'assistant' and it's present
        if msg["role"] == "assistant" and "tool_calls" in msg and msg["tool_calls"] is not None:
            # Ensure tool_calls are correctly formatted for the API
            # They might be Pydantic objects or already dicts
            if isinstance(msg["tool_calls"], list) and all(hasattr(tc, 'model_dump') for tc in msg["tool_calls"]):
                cleaned_msg["tool_calls"] = [tc.model_dump() for tc in msg["tool_calls"]]
            elif isinstance(msg["tool_calls"], list) and all(isinstance(tc, dict) for tc in msg["tool_calls"]):
                cleaned_msg["tool_calls"] = msg["tool_calls"]
            else:
                logging.warning(f"Skipping malformed tool_calls in message: {msg}")

        cleaned_messages.append(cleaned_msg)
    return cleaned_messages


def get_chat_response(user_message: str, chat_id: str) -> str: # Added chat_id parameter
    global conversation_history_backend, faiss_index, indexed_document_chunks

    # Append user message with chat_id and timestamp for local history storage
    conversation_history_backend.append({"id": chat_id, "role": "user", "content": user_message, "timestamp": datetime.datetime.now().isoformat()})

    final_assistant_reply = ""

    user_message_lower = user_message.lower()
    
    explicit_db_search_keywords = ["database", "files", "records", "our system", "our knowledge base", "from the database", "from our files"]
    user_requested_db_search = any(keyword in user_message_lower for keyword in explicit_db_search_keywords)

    # Removed direct keyword checks for list_folder_keywords and specific_file_keywords.
    # The model's tool calling will handle these intents.

    # Updated semantic search keywords for better detection
    semantic_search_keywords = ["summarize", "main points", "details about", "what is", "tell me about", "explain", "information on", "context on", "about the", "advice on", "tips for", "how to", "advantages of", "disadvantages of", "explain about"]


    try:
        # Prioritize semantic search if keywords and database request are present
        if user_requested_db_search and any(keyword in user_message_lower for keyword in semantic_search_keywords):
            logging.debug("Detected strong request for semantic search. Directly invoking semantic_search_knowledge_base.")
            relevant_chunks_data = execute_semantic_search_knowledge_base(user_message)
            
            if relevant_chunks_data:
                context_for_llm = ""
                for chunk in relevant_chunks_data:
                    context_for_llm += f"From {chunk['source_file']}:\n{chunk['content']}\n\n"
                
                # Prepare messages for API: includes full history + new semantic query prompt
                temp_conversation_history_for_api = _clean_messages_for_api(conversation_history_backend)
                
                # Append the specific prompt for semantic search after cleaning
                temp_conversation_history_for_api.append({"role": "user", "content": f"Based *ONLY* on the following provided information, answer the user's query: '{user_message}'. If the information does not contain enough details to fully answer, state that you can only provide details based on the available information. If the retrieved information is relevant but doesn't fully answer the query, or if the user's phrasing needs a slight interpretation of the context, do your best to summarize the relevant parts.\n\nRelevant Information:\n{context_for_llm.strip()}"})


                chat_completion = client.chat.completions.create(
                    model="accounts/fireworks/models/llama-v3p1-8b-instruct",
                    messages=temp_conversation_history_for_api, # Use the cleaned list for API call
                    temperature=0.1,
                    max_tokens=500
                )
                final_assistant_reply = chat_completion.choices[0].message.content.strip()
                logging.debug(f"Assistant reply after semantic search: {final_assistant_reply}")
            else:
                final_assistant_reply = "I couldn't find relevant information in the knowledge base for your query. There is no information available in the database regarding that query."
                logging.debug(f"Semantic search found no results. Reply: {final_assistant_reply}")

        # Fallback to model's tool choice or general knowledge if no specific intent is directly matched
        else:
            logging.debug("No direct intent matched. Letting model decide tool use or general knowledge.")
            
            # Clean messages before sending to API for initial tool-choice decision
            api_messages_for_llm_decision = _clean_messages_for_api(conversation_history_backend)

            chat_completion = client.chat.completions.create(
                model="accounts/fireworks/models/llama-v3p1-8b-instruct",
                messages=api_messages_for_llm_decision, # Use the cleaned list for API call
                tools=tools,
                temperature=0.1,
                max_tokens=1024
            )

            agent_response_message = chat_completion.choices[0].message
            logging.debug(f"Agent response message: {agent_response_message}")

            if agent_response_message.tool_calls:
                logging.debug("Model requested a tool call.")
                tool_output_content = []
                for tool_call in agent_response_message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)

                    logging.debug(f"Tool requested: {function_name} with args: {function_args}")

                    tool_output_list = []
                    if function_name == "search_filenames":
                        tool_output_list = execute_search_filenames(search_term=function_args.get("search_term"))
                    elif function_name == "search_file_content":
                        tool_output_list = execute_search_file_content(search_string=function_args.get("search_string"))
                    elif function_name == "semantic_search_knowledge_base":
                        tool_output_list = execute_semantic_search_knowledge_base(query=function_args.get("query"))
                    else:
                        logging.error(f"Unknown tool {function_name}")
                        tool_output_content.append(f"Error: Unknown tool '{function_name}'.")
                        continue
                    
                    if tool_output_list:
                        tool_output_content.append(json.dumps(tool_output_list))
                    else:
                        tool_output_content.append("No information found relevant to this tool call.")

                # Append assistant's tool call and tool's response to history (for local storage)
                conversation_history_backend.append(
                    {
                        "id": chat_id, # Include chat_id for tool calls as well
                        "role": "assistant",
                        "content": None, # Content is None when tool_calls are present
                        "tool_calls": [tool_call.model_dump() for tool_call in agent_response_message.tool_calls],
                        "timestamp": datetime.datetime.now().isoformat()
                    }
                )
                conversation_history_backend.append(
                    {
                        "id": chat_id, # Include chat_id for tool messages
                        "role": "tool",
                        "content": "\n".join(tool_output_content),
                        "timestamp": datetime.datetime.now().isoformat()
                    }
                )
                
                # Prepare messages for API after tool execution
                api_messages_after_tool = _clean_messages_for_api(conversation_history_backend)

                next_chat_completion = client.chat.completions.create(
                    model="accounts/fireworks/models/llama-v3p1-8b-instruct",
                    messages=api_messages_after_tool, # Use the cleaned list for API call
                    tools=tools, # Pass tools again for the next turn
                    temperature=0.1,
                    max_tokens=500
                )
                final_assistant_reply = next_chat_completion.choices[0].message.content.strip()
                logging.debug(f"Assistant reply after tool execution: {final_assistant_reply}")
            else:
                final_assistant_reply = agent_response_message.content.strip()
                logging.debug(f"No tool call was made by the model. Using direct reply (general knowledge): {final_assistant_reply}")

        if not final_assistant_reply.strip():
            final_assistant_reply = "I'm sorry, I couldn't process that request or no relevant information was found."

    except Exception as e:
        final_assistant_reply = f"An error occurred while processing your request: {e}"
        logging.error(f"An error occurred in get_chat_response: {e}")
    
    # Append the final assistant's reply with chat_id and timestamp for local history storage
    conversation_history_backend.append({"id": chat_id, "role": "assistant", "content": final_assistant_reply, "timestamp": datetime.datetime.now().isoformat()})
    save_conversation(conversation_history_backend)
    return final_assistant_reply

@app.route('/')
def index():
    logging.debug("Attempting to serve index.html from 'public' folder.")
    return send_from_directory('public', 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    logging.debug(f"Attempting to serve static file: {filename} from 'public' folder.")
    return send_from_directory('public', filename)

@app.route('/search', methods=['POST'])
def search_endpoint():
    user_query = request.json.get('query')
    chat_id = request.json.get('chat_id') # Get chat_id from frontend
    if not user_query:
        return jsonify({"error": "No query provided"}), 400
    if not chat_id: # Ensure chat_id is provided for new session management
        return jsonify({"error": "No chat_id provided"}), 400

    logging.debug(f"Received query from frontend: {user_query}")
    
    assistant_reply = get_chat_response(user_query, chat_id) # Pass chat_id to get_chat_response

    return jsonify({"result": assistant_reply})

@app.route('/clear_chat', methods=['POST'])
def clear_chat_endpoint():
    global conversation_history_backend
    # Reset conversation history to just the system message
    conversation_history_backend = [SYSTEM_MESSAGE]
    save_conversation(conversation_history_backend)
    logging.info("Chat history cleared successfully.")
    return jsonify({"message": "Chat history cleared successfully."})


if __name__ == '__main__':
    logging.debug("Initializing FAISS index on app startup...")
    knowledge_full_path = os.path.join(REPO_PATH, 'knowledge')
    if not os.path.exists(knowledge_full_path):
        os.makedirs(knowledge_full_path)
        logging.info(f"Created directory: {knowledge_full_path}")
        dummy_conversation = [
            {"content": "Hey Bob, did you get my email about the project deadline extension?"},
            {"content": "Hi Alice, yes I did. Thanks for the heads-up! I'll update the team."},
            {"content": "Great! Also, check the email regarding the Q3 marketing budget. It has some important updates."},
            {"content": "Will do. Is there a specific email about the new client proposal?"},
            {"content": "Yes, I sent that yesterday. Subject: 'New Client Proposal - Initial Draft'."}
        ]
        with open(os.path.join(knowledge_full_path, 'conversation_1.json'), 'w', encoding='utf-8') as f:
            json.dump(dummy_conversation, f, indent=4)
        logging.info(f"Created dummy conversation file: {os.path.join(knowledge_full_path, 'conversation_1.json')}")

    all_chunks_for_indexing = load_and_chunk_conversations(REPO_PATH)
    faiss_index, indexed_document_chunks = build_faiss_index(all_chunks_for_indexing, embedding_client)
    if faiss_index is None:
        logging.warning("FAISS index could not be built during startup. Semantic search may not work.")

    public_full_path = os.path.join(REPO_PATH, 'public')
    if not os.path.exists(public_full_path):
        os.makedirs(public_full_path)
        logging.info(f"Created directory: {public_full_path}")
    
    # Initialize chat_completions.json with just the system message if it's empty or malformed
    if not os.path.exists(CONVERSATION_FILE):
        logging.info(f"Creating empty chat_completions.json at {CONVERSATION_FILE}")
        with open(CONVERSATION_FILE, 'w', encoding='utf-8') as f:
            json.dump([SYSTEM_MESSAGE], f)
    else:
        # Ensure it has the correct system message on startup
        current_history = load_conversation()
        if not current_history or current_history[0].get("role") != "system" or current_history[0].get("content") != SYSTEM_MESSAGE["content"]:
            logging.info("chat_completions.json missing correct system message or content; re-initializing.")
            save_conversation([SYSTEM_MESSAGE])


    app.run(debug=True)