from pydantic import BaseModel


class SummaryRequest(BaseModel):
   doc_id: str