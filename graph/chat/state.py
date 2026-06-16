from typing import List, Optional, TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages

class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    doc_ids: List[str]
    chunks: List[str]
    sources: List[str]
    answer: str
    error: Optional[str]