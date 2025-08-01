from pydantic import BaseModel
from typing import List, Optional

class QueryRequest(BaseModel):
    question: str
    use_workflow: bool = False

class QueryResponse(BaseModel):
    answer: str
    sources: List[str] = []
    workflow_steps: Optional[List[str]] = None
