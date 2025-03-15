# FastAPI Backend with LangChain RAG and Streaming

This project implements a FastAPI backend that uses LangChain for Retrieval-Augmented Generation (RAG) with streaming capabilities.

## Features

- FastAPI backend with Server-Sent Events (SSE) for streaming responses
- LangChain integration with RAG architecture
- Custom callback handler to stream LangChain events
- FAISS vector database for document retrieval
- Streaming token-by-token responses from the LLM

## Requirements

- Python 3.8+
- OpenAI API key

## Installation

1. Clone the repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set up your environment variables:

Edit the `.env` file and add your OpenAI API key:

```
OPENAI_API_KEY=your_openai_api_key_here
```

## Running the Server

```bash
python run.py
```

The server will start at `http://localhost:8000`.

## API Endpoints

### GET /stream

Streams a response to a query using the RAG pipeline.

**Query Parameters:**
- `query` (string, required): The question to answer

**Response:**
Server-Sent Events (SSE) stream containing:
- LLM tokens as they're generated
- LangChain internal events (chain start/end, tool usage, etc.)

### POST /documents

Adds documents to the vector store.

**Request Body:**
```json
{
  "documents": [
    {
      "content": "Document text content",
      "metadata": {"source": "optional metadata"}
    }
  ]
}
```

**Response:**
```json
{
  "message": "Adding X documents to the vector store"
}
```

## Project Structure

- `app/main.py`: FastAPI routes and application setup
- `app/rag_chain.py`: RAG chain implementation using LangChain
- `app/callbacks.py`: Custom callback handler for streaming events
- `run.py`: Entry point to run the application

## Frontend Integration

This backend is designed to work with a frontend that can consume SSE streams. The frontend should:

1. Connect to the `/stream` endpoint with the query parameter
2. Process the incoming SSE events
3. Display the tokens as they arrive
4. Optionally, display the LangChain events for transparency

Example frontend code (JavaScript):

```javascript
const query = "Your question here";
const eventSource = new EventSource(`http://localhost:8000/stream?query=${encodeURIComponent(query)}`);

eventSource.onmessage = (event) => {
  const token = event.data;
  // Append token to UI or process LangChain event
  console.log(token);
};

eventSource.onerror = (error) => {
  console.error("EventSource error:", error);
  eventSource.close();
};
```
