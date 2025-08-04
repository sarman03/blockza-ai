from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from apis.directory.router import router as directory_router

app = FastAPI(title="RAG + LangGraph API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the directory router
app.include_router(directory_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)