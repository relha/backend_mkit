import os
from typing import List, Optional
from fastapi import FastAPI, Request, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel
from langchain.schema import Document
from dotenv import load_dotenv

from .rag_chain import RAGChain

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Initialize FastAPI app
app = FastAPI(title="RAG API with Streaming")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG Chain
rag_chain = RAGChain()

# Models
class DocumentInput(BaseModel):
    content: str
    metadata: Optional[dict] = None

class DocumentsInput(BaseModel):
    documents: List[DocumentInput]

# Dependency to ensure RAG chain is initialized
async def get_rag_chain():
    if rag_chain.vector_store is None:
        await rag_chain.initialize()
    return rag_chain

@app.on_event("startup")
async def startup_event():
    """Initialize the RAG chain on startup."""
    await rag_chain.initialize()

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "RAG API with Streaming is running"}

@app.get("/stream")
async def stream_response(
    query: str,
    rag_chain: RAGChain = Depends(get_rag_chain)
):
    """Stream a response for the given query."""
    if not query:
        raise HTTPException(status_code=400, detail="Query parameter is required")
    
    async def event_generator():
        async for token in rag_chain.query_with_streaming(query):
            yield {"event": "message", "data": token}
    
    return EventSourceResponse(event_generator())

@app.post("/documents")
async def add_documents(
    documents_input: DocumentsInput,
    background_tasks: BackgroundTasks,
    rag_chain: RAGChain = Depends(get_rag_chain)
):
    """Add documents to the vector store."""
    documents = [
        Document(page_content=doc.content, metadata=doc.metadata or {})
        for doc in documents_input.documents
    ]
    
    background_tasks.add_task(rag_chain.add_documents, documents)
    
    return {"message": f"Adding {len(documents)} documents to the vector store"}
