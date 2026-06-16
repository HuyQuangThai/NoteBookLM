from typing import TypedDict, List, Optional, Literal

class IngestionState(TypedDict):
    file_path: str
    raw_text: str
    chunks: List[str]
    doc_id: str
    error: Optional[str]