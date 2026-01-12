# PDF RAG Chatbot

A local chatbot that answers questions about PDF documents using RAG (Retrieval-Augmented Generation).

## Features

- Upload PDF and ask questions
- Get answers with page citations
- 100% free - runs locally with Ollama
- Fast semantic search with ChromaDB

## Tech Stack

- FastAPI, Ollama (Llama 3.2), ChromaDB, sentence-transformers, PyPDF2

## Installation

```bash
# Clone and setup
git clone https://github.com/parsasamandi/pdf-chatbot.git
cd pdf-chatbot
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Install Ollama from https://ollama.ai
ollama pull llama3.2:1b
```

## Usage

```bash
# Terminal 1: Start Ollama
ollama serve

# Terminal 2: Start app
python3 main.py

# Open browser: http://localhost:8000
```

## How It Works

1. Upload PDF → Extract text → Chunk into pieces
2. Convert chunks to embeddings → Store in ChromaDB
3. User asks question → Find relevant chunks
4. Send chunks + question to Ollama → Get answer

## Requirements

- Python 3.8+
- Ollama installed
- 4GB+ RAM

## License

MIT