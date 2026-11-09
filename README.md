# Local PDF Question Answering Chatbot

A Streamlit app that lets you ask questions about an uploaded PDF. It extracts page text, splits it into overlapping chunks, searches those chunks with FAISS, and uses a local Ollama language model to generate answers with page references.

## Features

- Upload a PDF and build a searchable index from its text.
- Retrieve relevant passages with local embeddings and FAISS similarity search.
- Answer questions with Ollama and show the source pages and retrieved text.
- Keep the language model and embedding workflow on your machine.

## Requirements

- Python 3.10 or newer
- [Ollama](https://ollama.com/) installed and running
- The default models, or models configured through environment variables:

```powershell
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

## Run locally

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

Then open the local Streamlit URL, upload a text-based PDF, and ask a question in the chat.

## Configuration

The app loads optional settings from a `.env` file. Defaults are shown below:

```dotenv
LLM_MODEL=llama3.2:3b
EMBEDDING_MODEL=nomic-embed-text
OLLAMA_BASE_URL=http://localhost:11434
CHUNK_SIZE=500
CHUNK_OVERLAP=100
TOP_K=4
```

The `.env` file, virtual environment, Python cache, and generated FAISS index are excluded from Git by `.gitignore`.

## Project files

- `app.py` — Streamlit user interface.
- `rag_pipeline.py` — PDF extraction, chunking, embeddings, FAISS search, and answer generation.
- `requirements.txt` — Python dependencies.
