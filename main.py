from fastapi import FastAPI
from apis.directory.endpoints import router as directory_router

app = FastAPI(title="RAG + LangGraph API")

app.include_router(directory_router, prefix="/directory", tags=["directory"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)