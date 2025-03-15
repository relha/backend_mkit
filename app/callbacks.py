import asyncio
from typing import Union, Any, Dict, List, Optional
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish, LLMResult

class StreamingCallbackHandler(BaseCallbackHandler):
    """Callback handler for streaming LangChain events to the client."""
    
    def __init__(self, queue):
        """Initialize with a queue to push events to."""
        self.queue = queue
        self.loop = asyncio.get_event_loop()
        
    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Run when LLM starts."""
        asyncio.run_coroutine_threadsafe(
            self.queue.put("[LLM Start] Generating response..."),
            self.loop
        )
        
    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        asyncio.run_coroutine_threadsafe(
            self.queue.put(token),
            self.loop
        )
        
    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run when LLM ends."""
        asyncio.run_coroutine_threadsafe(
            self.queue.put("[LLM End]"),
            self.loop
        )
        
    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Run when LLM errors."""
        asyncio.run_coroutine_threadsafe(
            self.queue.put(f"[LLM Error] {str(error)}"),
            self.loop
        )
        
    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Run when chain starts."""
        chain_type = serialized.get("name", "Chain")
        asyncio.run_coroutine_threadsafe(
            self.queue.put(f"[{chain_type} Start]"),
            self.loop
        )
        
    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Run when chain ends."""
        asyncio.run_coroutine_threadsafe(
            self.queue.put("[Chain End]"),
            self.loop
        )
        
    def on_chain_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Run when chain errors."""
        asyncio.run_coroutine_threadsafe(
            self.queue.put(f"[Chain Error] {str(error)}"),
            self.loop
        )
        
    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> None:
        """Run when tool starts."""
        tool_name = serialized.get("name", "Tool")
        asyncio.run_coroutine_threadsafe(
            self.queue.put(f"[{tool_name} Start] Using {tool_name}..."),
            self.loop
        )
        
    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """Run when tool ends."""
        asyncio.run_coroutine_threadsafe(
            self.queue.put("[Tool End]"),
            self.loop
        )
        
    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Run when tool errors."""
        asyncio.run_coroutine_threadsafe(
            self.queue.put(f"[Tool Error] {str(error)}"),
            self.loop
        )
        
    def on_text(self, text: str, **kwargs: Any) -> None:
        """Run on arbitrary text."""
        asyncio.run_coroutine_threadsafe(
            self.queue.put(text),
            self.loop
        )
        
    def on_retriever_start(
        self, serialized: Dict[str, Any], query: str, **kwargs: Any
    ) -> None:
        """Run when retriever starts."""
        asyncio.run_coroutine_threadsafe(
            self.queue.put(f"[Retriever Start] Searching for relevant documents..."),
            self.loop
        )
        
    def on_retriever_end(
        self, documents: List[Any], **kwargs: Any
    ) -> None:
        """Run when retriever ends."""
        asyncio.run_coroutine_threadsafe(
            self.queue.put(f"[Retriever End] Found {len(documents)} relevant documents"),
            self.loop
        )