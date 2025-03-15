from typing import Any, Dict, List, Optional
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish, LLMResult

class StreamingCallbackHandler(BaseCallbackHandler):
    """Callback handler for streaming LangChain events to the client."""
    
    def __init__(self, queue):
        """Initialize with a queue to push events to."""
        self.queue = queue
        
    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Run when LLM starts."""
        self.queue.put("[LLM Start] Generating response...")
        
    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        self.queue.put(token)
        
    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run when LLM ends."""
        self.queue.put("[LLM End]")
        
    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Run when LLM errors."""
        self.queue.put(f"[LLM Error] {str(error)}")
        
    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Run when chain starts."""
        chain_type = serialized.get("name", "Chain")
        self.queue.put(f"[{chain_type} Start]")
        
    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Run when chain ends."""
        self.queue.put("[Chain End]")
        
    def on_chain_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Run when chain errors."""
        self.queue.put(f"[Chain Error] {str(error)}")
        
    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> None:
        """Run when tool starts."""
        tool_name = serialized.get("name", "Tool")
        self.queue.put(f"[{tool_name} Start] Using {tool_name}...")
        
    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """Run when tool ends."""
        self.queue.put("[Tool End]")
        
    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Run when tool errors."""
        self.queue.put(f"[Tool Error] {str(error)}")
        
    def on_text(self, text: str, **kwargs: Any) -> None:
        """Run on arbitrary text."""
        self.queue.put(text)
        
    def on_retriever_start(
        self, serialized: Dict[str, Any], query: str, **kwargs: Any
    ) -> None:
        """Run when retriever starts."""
        self.queue.put(f"[Retriever Start] Searching for relevant documents...")
        
    def on_retriever_end(
        self, documents: List[Any], **kwargs: Any
    ) -> None:
        """Run when retriever ends."""
        self.queue.put(f"[Retriever End] Found {len(documents)} relevant documents")
