from pydantic import BaseModel
from typing import List, Dict, Optional

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    source_companies: List[Dict] = []
    total_sources: int = 0
    query_processed: str = ""
    status: str = "success"

class InitializeRequest(BaseModel):
    anthropic_api_key: Optional[str] = None
    use_sample_data: bool = False
    sample_data: List[Dict] = []
    force_reinit: bool = False

class SystemStatus(BaseModel):
    initialized: bool
    status: str
    message: str
    statistics: Optional[Dict] = None
