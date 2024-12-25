# ChatwithURL

## AI-Powered Documentation Retrieval Assistant

An intelligent documentation retrieval system that uses LLMs (Large Language Models) and vector search to provide accurate answers to questions about any web content or documentation.

## Features

- **URL Content Processing**: Extract and process content from any web URL
- **Intelligent Text Chunking**: Splits content into meaningful chunks while preserving context
- **Vector Search**: Uses Chroma DB for efficient similarity search
- **LLM Integration**: Powered by Ollama for natural language understanding
- **Web Interface**: Simple and intuitive UI for asking questions
- **Session Management**: Maintains separate contexts for different users

## Prerequisites

- Python 3.10 or higher
- Ollama server running locally or remotely
- Required Python packages (see Installation)

## Installation

1. Clone the repository:
```bash
git clone git@github.com:rubeyyyy/ChatwithURL.git
cd ChatwithURL
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Project Structure

```
.
├── main.py              # FastAPI application entry point
├── utils.py          # Core processing functions
├── static/
│   └── index.html      # Web interface
├── requirements.txt    # Project dependencies
└── README.md
```

## Configuration

1. Set up your Ollama server URL or add you LLM in in `utils.py`:
```python
base_url = 'https://your-ollama-server.com'  # Replace with your Ollama server URL
```

2. Configure the vector store directory in `utils.py`:
```python
VECTORSTORE_DIR = "path/to/vectorstore"  # Replace with your preferred directory
```

## Usage

1. Start the FastAPI server:
```bash
python main.py
```

2. Open your web browser and navigate to:
```
http://localhost:8000
```

3. In the web interface:
   - Enter a URL to process
   - Wait for content processing to complete
   - Ask questions about the content
   - Receive AI-powered answers based on the content

## API Endpoints

- `POST /set_url/`: Process a new URL
  ```json
  {
    "url": "https://example.com",
    "session_id": "unique-session-id"
  }
  ```

- `POST /ask_question/`: Ask a question about the processed content
  ```json
  {
    "question": "What is this article about?",
    "session_id": "unique-session-id"
  }
  ```

## Web Interface

The web interface provides:
- URL input field
- Question input field
- Response display area
- Session management
- Error handling and status messages

## Error Handling

The system handles various error cases:
- Invalid URLs
- Unreachable content
- Processing failures
- LLM server issues
- Session management errors

## Technical Details

### Content Processing
- Fetches URL and extract the content
- Split the text using RecursiveCharacterTextSplitter 
- Preserves document structure

### Vector Search
- Uses Chroma DB for vector storage
- HuggingFace embeddings (all-MiniLM-L6-v2)
- Efficient similarity search
- Persistent storage

### LLM Integration
- Uses Chatbedrock for natural language processing
- Context-aware responses
- Source attribution
- Confidence scoring

## Acknowledgments

- Anthropic model for LLM capabilities
- LangChain for the chain infrastructure
- ChromaDB for vector storage
- HuggingFace for embeddings