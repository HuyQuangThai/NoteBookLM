from typing import TypedDict, List, Optional, Annotated
from operator import add


class SummaryState(TypedDict):
    doc_id: str
    chunks: List[str]
    summaries: Annotated[list[str], add]
    final_summary: str
    insights: List[str]
    tags: List[str]
    error: Optional[str]
    batches: List[str]

class MapState(TypedDict):
    batch: List[str]
