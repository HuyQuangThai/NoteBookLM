from typing import List

from pydantic import BaseModel


class ChatRequest(BaseModel):
    question: str
    doc_ids: List[str]
    thread_id: str