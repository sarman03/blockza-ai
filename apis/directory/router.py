"""
FastAPI Router for Directory RAG System
Provides RESTful API interface for directory queries
"""

import os
import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from rag.directory_rag import DirectoryRAGSystem

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/rag", tags=["directory-rag"])

# Global RAG system instance
rag_system = None

# Pydantic models for request/response
class InitializeRequest(BaseModel):
    anthropic_api_key: Optional[str] = None
    use_sample_data: bool = False
    sample_data: List[Dict] = []
    force_reinit: bool = False

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    source_companies: List[Dict] = []
    total_sources: int = 0
    query_processed: str = ""
    status: str = "success"

@router.get('/health')
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Directory RAG API",
        "version": "1.0.0"
    }

@router.post('/initialize')
async def initialize_rag_system(request: InitializeRequest):
    """
    Initialize the RAG system with directories data
    """
    global rag_system
    
    try:
        # Check if already initialized and not forcing reinit
        if rag_system and not request.force_reinit:
            return {
                "message": "RAG system already initialized",
                "status": "already_initialized"
            }
        
        # Get configuration
        anthropic_key = request.anthropic_api_key or os.getenv('ANTHROPIC_API_KEY')
        use_sample = request.use_sample_data
        sample_data = request.sample_data
        
        if not anthropic_key:
            raise HTTPException(
                status_code=400,
                detail="Anthropic API key is required. Provide 'anthropic_api_key' in request body or set ANTHROPIC_API_KEY environment variable"
            )
        
        logger.info("🚀 Starting RAG system initialization...")
        
        # Initialize RAG system
        rag_system = DirectoryRAGSystem(anthropic_api_key=anthropic_key)
        
        # Initialize with data
        init_result = rag_system.initialize_system(
            use_sample_data=use_sample,
            sample_data=sample_data
        )
        
        logger.info("✅ RAG system initialized successfully")
        
        return {
            "message": "RAG system initialized successfully",
            "statistics": init_result,
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"❌ Initialization failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initialize RAG system: {str(e)}"
        )

@router.post('/query')
async def query_directories(request: QueryRequest):
    """
    Query the RAG system for directory information
    """
    global rag_system
    
    if not rag_system:
        raise HTTPException(
            status_code=400,
            detail="RAG system not initialized. Please initialize the system first using /api/rag/initialize"
        )
    
    try:
        question = request.question.strip()
        if not question:
            raise HTTPException(
                status_code=400,
                detail="Question cannot be empty"
            )
        
        # Process the query
        result = rag_system.query_directories(question)
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Query processing failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process query: {str(e)}"
        )

@router.get('/categories')
async def get_categories():
    """Get summary of all available categories and their statistics"""
    global rag_system
    
    if not rag_system:
        raise HTTPException(
            status_code=400,
            detail="RAG system not initialized"
        )
    
    try:
        summary = rag_system.get_categories_summary()
        return summary
        
    except Exception as e:
        logger.error(f"❌ Failed to get categories: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve categories: {str(e)}"
        )

@router.get('/status')
async def get_system_status():
    """Get current system status and statistics"""
    global rag_system
    
    if not rag_system:
        return {
            "initialized": False,
            "status": "not_initialized",
            "message": "RAG system is not initialized"
        }
    
    try:
        # Get system statistics
        stats = rag_system.get_categories_summary()
        
        return {
            "initialized": True,
            "status": "ready",
            "message": "RAG system is ready for queries",
            "statistics": {
                "total_companies": stats.get("total_companies", 0),
                "total_categories": stats.get("total_categories", 0),
                "available_categories": list(stats.get("categories", {}).keys())
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get status: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"System error: {str(e)}"
        )

@router.get('/sample-queries')
async def get_sample_queries():
    """Get sample queries that users can try"""
    sample_queries = [
        "Give me 5 Web3 directories with their founders",
        "Show me all verified companies sorted by category",
        "Tell me about Chainsight and what they do",
        "List all FinTech companies in the directory",
        "Who founded Emurgo Kepple Ventures?",
        "Give me the most popular directories by views",
        "Show me all SaaS companies with their verification status",
        "What categories of companies are available?",
        "Give me information about blockchain companies",
        "List all companies founded by women entrepreneurs"
    ]
    
    return {
        "sample_queries": sample_queries,
        "usage_tips": [
            "Ask for companies by category (Web3, SaaS, FinTech, etc.)",
            "Request specific information about founders",
            "Ask for verified companies only",
            "Request companies sorted by popularity",
            "Ask about specific company details"
        ]
    } 