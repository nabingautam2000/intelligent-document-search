# Intelligent Document Search

This project is an AI-powered document assistant that helps you intelligently interact with your local files. It provides a conversational interface where you can ask natural language questions to find and retrieve information.

## Features

* **Intelligent Document Search:** The system offers advanced search capabilities for your local files, going beyond simple keyword matching.
    * **Semantic Search:** Understands the meaning of your natural language questions to retrieve and summarize relevant content from your JSON knowledge base files.
    * **Filename Search:** Allows you to efficiently find files based on keywords present in their names.
    * **File Content Search:** Enables you to locate documents that contain specific phrases or strings within their actual content.

* **Conversational Interface:** Features a user-friendly chat-like interface for intuitive interaction, allowing you to ask questions and receive responses directly.

* **Knowledge Base Management:** Indexes your collection of JSON conversation files, forming a structured knowledge base for quick and efficient information retrieval.

* **Smart Tool Selection (AI-Powered):** The underlying AI model intelligently determines which of its available search tools (semantic search, filename search, or content search) is best suited to answer your specific query.

* **Grounded Responses:** Provides answers that are strictly based on the retrieved content from your files, ensuring factual accuracy and preventing the generation of unverified information (hallucinations).

* **Chat History:** Maintains a chronological record of your past conversations, displayed in a sidebar, for easy reference and continuity.

## Setup Instructions

To get this project up and running on your local machine, follow these steps:

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/nabingautam2000/intelligent-document-search.git
    cd intelligent-document-search
    ```

2.  **Install Dependencies:**
    It's recommended to use a virtual environment.
    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    
    pip install -r requirements.txt
    ```

3.  **Set up Environment Variables:**
    * Create a file named `.env` in the root directory of the project (the same folder as `app.py`).
    * Inside `.env`, add your Fireworks AI API key:
        ```
        FIREWORKS_API_KEY="YOUR_FIREWORKS_API_KEY_HERE"
        ```
    * Replace `"YOUR_FIREWORKS_API_KEY_HERE"` with your actual API key from Fireworks AI.

4.  **Prepare Knowledge Base Files:**
    * Place your JSON knowledge base files (e.g., `career_advice.json`, `fitness_motivation.json`, etc.) into the `knowledge/` subdirectory.
    * Ensure these JSON files are formatted as a list containing objects with a `content` field. Example:
        ```json
        [
          {
            "content": "Your conversation text goes here..."
          }
        ]
        ```

5.  **Run the Application:**
    ```bash
    python app.py
    ```
    The application will typically run on `http://127.0.0.1:5000/`.

6.  **Access the Interface:**
    Open your web browser and navigate to `http://127.0.0.1:5000/`.

## Usage

* Type your questions or search queries into the chat input field at the bottom.
* The sidebar will display your chat history.
* Use the search bar in the sidebar to filter past conversations.
* Click "+ New chat" to start a fresh conversation session.

## Troubleshooting

* **`429 Too Many Requests`:** If you encounter this error during startup or interaction, you might be hitting API rate limits. A small `time.sleep()` is already included in `embedding_search_util.py` to mitigate this. If it persists, Fireworks AI might have strict free-tier limits, or you might need a brief pause between queries.
* **"Error processing your request: Error code: 400"**: Ensure your `chat_completions.json` file starts with only the system message and that your `app.py` code is the latest version. This error often indicates malformed messages sent to the LLM API.
* **Incorrect File Listings / Hallucinations:** If the system lists non-existent files for `search_filenames` or `search_file_content` queries, ensure your `knowledge` folder contains *only* the expected JSON files and that `app.py` is the latest version with the explicit tool descriptions.

---
