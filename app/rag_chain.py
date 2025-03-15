import os
import asyncio
from typing import AsyncIterable, Dict, List, Optional
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI  # Importation mise à jour
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from .callbacks import StreamingCallbackHandler

class RAGChain:
    """RAG Chain implementation using LangChain."""
    
    def __init__(self, vector_store_path: str = "vectorstore"):
        """Initialize the RAG Chain with a vector store path."""
        self.embeddings = OpenAIEmbeddings()
        self.vector_store_path = vector_store_path
        self.vector_store = None
        
    async def initialize(self, documents: Optional[List[Document]] = None):
        """Initialize or load the vector store."""
        if os.path.exists(self.vector_store_path) and not documents:
            # Load existing vector store
            self.vector_store = FAISS.load_local(
                self.vector_store_path, 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        elif documents:
            # Create new vector store from documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=100
            )
            split_documents = text_splitter.split_documents(documents)
            self.vector_store = FAISS.from_documents(split_documents, self.embeddings)
            self.vector_store.save_local(self.vector_store_path)
        else:
            # Create empty vector store
            self.vector_store = FAISS.from_documents(
                [Document(page_content="Initial document")], self.embeddings
            )
            self.vector_store.save_local(self.vector_store_path)
            
    def _create_chain(self, streaming_queue):
        """Create the RAG chain with streaming capabilities."""
        # Create the language model with streaming
        llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0.3,
            streaming=True,
            callbacks=[StreamingCallbackHandler(streaming_queue)]
        )
        
        # Create the prompt template
        prompt = ChatPromptTemplate.from_template("""
        Answer the following question based on the provided context. 
        If you don't know the answer, just say you don't know.
        
        Context:
        {context}
        
        Question:
        {input}
        
        Answer:
        """)
        
        # Create the document chain
        document_chain = create_stuff_documents_chain(llm, prompt)
        
        # Create the retrieval chain
        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        return create_retrieval_chain(retriever, document_chain)
        
    async def query_with_streaming(self, query: str) -> AsyncIterable[str]:
        """Query the RAG chain and stream the results."""
        # Create a queue for streaming results
        queue = asyncio.Queue()
        
        # Create the chain with the streaming queue
        chain = self._create_chain(queue)
        
        # Run the chain in a separate task
        async def run_chain():
            try:
                await chain.ainvoke({"input": query})
                # Signal the end of the stream
                await queue.put(None)
            except Exception as e:
                await queue.put(f"[Error] {str(e)}")
                await queue.put(None)
                
        # Start the chain execution
        asyncio.create_task(run_chain())
        
        # Stream results from the queue
        while True:
            token = await queue.get()
            if token is None:
                break
            yield token
            
    async def add_documents(self, documents: List[Document]):
        """Add documents to the vector store."""
        if self.vector_store is None:
            await self.initialize(documents)
        else:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=100
            )
            split_documents = text_splitter.split_documents(documents)
            self.vector_store.add_documents(split_documents)
            self.vector_store.save_local(self.vector_store_path)