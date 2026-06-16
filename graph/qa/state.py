from pydantic import BaseModel, Field
from typing import TypedDict, List, Optional, Literal


class GradeAnswer(BaseModel):
    binary_score: Literal["yes", "no"]  = Field(
        description="Do the chunks contain enough information to answer the question? Return 'yes' or 'no' ",
    )


class QAState(TypedDict):
    question: str
    doc_ids: List[str]
    chunks: List[str]
    is_relevant: bool
    answer: str
    sources: List[dict]
    error: Optional[str]