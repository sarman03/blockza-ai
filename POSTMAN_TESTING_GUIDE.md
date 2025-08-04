# Postman Testing Guide for RAG Directory API

## API Base URL
```
http://localhost:8000
```

## Environment Variables Setup
Create a new environment in Postman with these variables:
- `base_url`: `http://localhost:8000`
- `anthropic_api_key`: Your Anthropic API key

## 1. Health Check
**Method:** GET  
**URL:** `{{base_url}}/api/rag/health`

**Expected Response:**
```json
{
  "status": "healthy",
  "service": "Directory RAG API",
  "version": "1.0.0"
}
```

## 2. Initialize RAG System
**Method:** POST  
**URL:** `{{base_url}}/api/rag/initialize`  
**Headers:** `Content-Type: application/json`

**Request Body:**
```json
{
  "anthropic_api_key": "{{anthropic_api_key}}",
  "use_sample_data": false,
  "sample_data": [],
  "force_reinit": false
}
```

**Expected Response:**
```json
{
  "message": "RAG system initialized successfully",
  "statistics": {
    "total_companies": 374,
    "categories": ["Web3", "FinTech", "SaaS", ...],
    "verified_count": 25
  },
  "status": "success"
}
```

## 3. Query Directories
**Method:** POST  
**URL:** `{{base_url}}/api/rag/query`  
**Headers:** `Content-Type: application/json`

**Request Body:**
```json
{
  "question": "What are the best Web3 companies in the directory?"
}
```

**Expected Response:**
```json
{
  "answer": "Based on the directory data, here are some notable Web3 companies...",
  "source_companies": [
    {
      "name": "Company Name",
      "category": "Web3",
      "verification": "verified",
      "snippet": "Company description..."
    }
  ],
  "total_sources": 5,
  "query_processed": "What are the best Web3 companies in the directory?",
  "status": "success"
}
```

## 4. Get Categories Summary
**Method:** GET  
**URL:** `{{base_url}}/api/rag/categories`

**Expected Response:**
```json
{
  "total_categories": 15,
  "categories": {
    "Web3": {
      "count": 45,
      "verified": 12,
      "companies": [...]
    }
  },
  "total_companies": 374
}
```

## 5. Get System Status
**Method:** GET  
**URL:** `{{base_url}}/api/rag/status`

**Expected Response:**
```json
{
  "initialized": true,
  "status": "ready",
  "message": "RAG system is ready for queries",
  "statistics": {
    "total_companies": 374,
    "total_categories": 15,
    "available_categories": ["Web3", "FinTech", "SaaS", ...]
  }
}
```

## 6. Get Sample Queries
**Method:** GET  
**URL:** `{{base_url}}/api/rag/sample-queries`

**Expected Response:**
```json
{
  "sample_queries": [
    "Give me 5 Web3 directories with their founders",
    "Show me all verified companies sorted by category",
    "Tell me about Chainsight and what they do"
  ],
  "usage_tips": [
    "Ask for companies by category (Web3, SaaS, FinTech, etc.)",
    "Request specific information about founders"
  ]
}
```

## Testing Workflow

### Step 1: Start the Server
```bash
python main.py
```

### Step 2: Test Health Check
- Send GET request to health endpoint
- Verify server is running

### Step 3: Initialize RAG System
- Send POST request to initialize endpoint
- Include your Anthropic API key
- Verify initialization is successful

### Step 4: Test Queries
- Send various POST requests to query endpoint
- Test different types of questions:
  - Category-based: "Show me all Web3 companies"
  - Specific: "Tell me about [Company Name]"
  - Comparison: "Compare [Company A] and [Company B]"
  - General: "What are the most popular companies?"

### Step 5: Check System Status
- Verify system is ready for queries
- Check statistics and available categories

## Sample Test Questions

1. **Category Queries:**
   - "What are the best Web3 companies?"
   - "Show me all FinTech companies"
   - "List all SaaS companies with their verification status"

2. **Specific Company Queries:**
   - "Tell me about Chainsight"
   - "What does [Company Name] do?"
   - "Who founded [Company Name]?"

3. **Comparison Queries:**
   - "Compare the top 3 Web3 companies"
   - "Which is better: [Company A] or [Company B]?"

4. **General Queries:**
   - "What are the most popular companies by views?"
   - "Show me all verified companies"
   - "What categories of companies are available?"

## Error Handling

### Common Errors:
1. **400 Bad Request:** Missing or invalid Anthropic API key
2. **400 Bad Request:** RAG system not initialized
3. **500 Internal Server Error:** API connection issues
4. **500 Internal Server Error:** LLM processing errors

### Troubleshooting:
- Check if server is running on port 8000
- Verify Anthropic API key is valid
- Ensure RAG system is initialized before querying
- Check network connectivity to external APIs

## Postman Collection Setup

1. **Create a new collection** called "RAG Directory API"
2. **Import environment variables** for base_url and anthropic_api_key
3. **Create request templates** for each endpoint
4. **Set up pre-request scripts** if needed for authentication
5. **Add test scripts** to validate responses

## Performance Testing

### Load Testing:
- Test with multiple concurrent requests
- Monitor response times
- Check memory usage during initialization
- Verify vector store performance

### Query Testing:
- Test with various question types
- Verify response quality and relevance
- Check source attribution accuracy
- Test error handling with invalid queries 