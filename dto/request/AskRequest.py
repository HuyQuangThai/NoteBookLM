from typing import List

from pydantic import BaseModel


class AskRequest(BaseModel):
    question: str
    doc_ids: List[str]