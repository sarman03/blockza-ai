from fastapi import APIRouter, HTTPException
from shared.models import QueryRequest, QueryResponse
from .rag import DirectoryRAG
from .workflow import DirectoryWorkflow

router = APIRouter()
rag_system = DirectoryRAG()
workflow_system = DirectoryWorkflow()

@router.post("/query", response_model=QueryResponse)
async def query_directory(request: QueryRequest):
    try:
        if request.use_workflow:
            result = await workflow_system.run(request.question)
            return QueryResponse(
                answer=result["answer"],
                sources=result.get("sources", []),
                workflow_steps=result.get("steps", [])
            )
        else:
            result = rag_system.query(request.question)
            return QueryResponse(
                answer=result["answer"],
                sources=result.get("sources", [])
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ingest")
async def ingest_data():
    """Fetch and ingest data from directory API"""
    try:
        await rag_system.ingest_directory_data()
        return {"message": "Data ingested successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
