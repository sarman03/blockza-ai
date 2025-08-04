"""
Complete RAG System for Directory API using Anthropic Claude and HuggingFace Embeddings
Based on latest LangChain documentation and best practices
"""

import os
import json
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import requests
from datetime import datetime
import logging

# Core LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_anthropic import ChatAnthropic

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DirectoryEntry:
    """Data class for directory entries with all relevant fields"""
    id: str
    name: str
    slug: str
    short_description: str
    detail: str
    category: str
    subcategory: str
    founder_name: str
    founder_details: str
    verification_status: str
    likes: int
    views: int
    created_at: str
    url: str
    social_links: Dict[str, str]

class DirectoryRAGSystem:
    """
    Complete RAG System for Directory Queries
    Features:
    - HuggingFace free embeddings
    - Anthropic Claude for intelligent responses
    - Category-wise sorting and filtering
    - Comprehensive directory information retrieval
    """
    
    def __init__(self, anthropic_api_key: str = None):
        """
        Initialize the RAG system with Anthropic API key
        
        Args:
            anthropic_api_key: Anthropic API key (defaults to env variable)
        """
        self.anthropic_api_key = anthropic_api_key or os.getenv('ANTHROPIC_API_KEY')
        if not self.anthropic_api_key:
            raise ValueError("Anthropic API key is required. Set ANTHROPIC_API_KEY environment variable.")
        
        # Initialize Anthropic Claude
        self.llm = ChatAnthropic(
            api_key=self.anthropic_api_key,
            model="claude-3-5-sonnet-20240620",  # Latest stable model
            temperature=0.1,
            max_tokens=2000
        )
        
        # Initialize HuggingFace embeddings (free, no API key needed)
        logger.info("Loading HuggingFace embeddings model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",  # Fast and effective
            model_kwargs={'device': 'cpu'},  # Use CPU for compatibility
            encode_kwargs={'normalize_embeddings': False}
        )
        
        # Text splitter for document chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Storage
        self.vector_store = None
        self.qa_chain = None
        self.directories_data = []
        
        # API configuration
        from config import DIRECTORY_API_URL
        self.api_base_url = DIRECTORY_API_URL
        
        logger.info("DirectoryRAGSystem initialized successfully")
    
    def clean_html_content(self, html_content: str) -> str:
        """Clean HTML tags and extract readable text"""
        if not html_content:
            return ""
        
        # Remove HTML tags
        clean_text = re.sub(r'<[^>]+>', ' ', html_content)
        # Remove extra whitespace and special characters
        clean_text = re.sub(r'\s+', ' ', clean_text)
        clean_text = re.sub(r'[^\w\s\-\.,;:!?()]', ' ', clean_text)
        return clean_text.strip()
    
    def fetch_directories_from_api(self) -> List[Dict[str, Any]]:
        """
        Fetch directories from your API endpoint
        
        Returns:
            List of directory dictionaries
        """
        try:
            logger.info(f"Fetching directories from: {self.api_base_url}")
            response = requests.get(self.api_base_url, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                directories = data.get('data', []) if isinstance(data, dict) else data
                logger.info(f"Successfully fetched {len(directories)} directories")
                return directories
            else:
                logger.error(f"API request failed with status: {response.status_code}")
                return []
                
        except requests.RequestException as e:
            logger.error(f"Error fetching directories: {str(e)}")
            return []
    
    def parse_directory_data(self, raw_data: List[Dict[str, Any]]) -> List[DirectoryEntry]:
        """
        Parse raw directory data into structured DirectoryEntry objects
        
        Args:
            raw_data: Raw directory data from API
            
        Returns:
            List of DirectoryEntry objects
        """
        directories = []
        
        for item in raw_data:
            try:
                # Handle social links
                social_links = item.get('socialLinks', {})
                if isinstance(social_links, str):
                    try:
                        social_links = json.loads(social_links)
                    except:
                        social_links = {}
                
                directory = DirectoryEntry(
                    id=item.get('_id', ''),
                    name=item.get('name', ''),
                    slug=item.get('slug', ''),
                    short_description=item.get('shortDescription', ''),
                    detail=self.clean_html_content(item.get('detail', '')),
                    category=item.get('category', ''),
                    subcategory=item.get('subcategory', ''),
                    founder_name=item.get('founderName', ''),
                    founder_details=item.get('founderDetails', ''),
                    verification_status=item.get('verificationStatus', ''),
                    likes=item.get('likes', 0),
                    views=item.get('views', 0),
                    created_at=item.get('createdAt', ''),
                    url=item.get('url', ''),
                    social_links=social_links
                )
                directories.append(directory)
                
            except Exception as e:
                logger.warning(f"Error parsing directory entry: {str(e)}")
                continue
        
        logger.info(f"Successfully parsed {len(directories)} directories")
        return directories
    
    def create_enhanced_documents(self, directories: List[DirectoryEntry]) -> List[Document]:
        """
        Create comprehensive LangChain documents optimized for category-wise queries
        
        Args:
            directories: List of DirectoryEntry objects
            
        Returns:
            List of Document objects with rich metadata
        """
        documents = []
        
        for directory in directories:
            # Create comprehensive content for better retrieval
            content_sections = [
                f"Company Name: {directory.name}",
                f"Category: {directory.category}",
                f"Subcategory: {directory.subcategory}",
                f"Description: {directory.short_description}",
                f"Detailed Information: {directory.detail}",
                f"Founder: {directory.founder_name}",
                f"Founder Background: {directory.founder_details}",
                f"Verification Status: {directory.verification_status}",
                f"Company URL: {directory.url}",
                f"Popularity: {directory.likes} likes, {directory.views} views"
            ]
            
            # Add social media presence
            social_info = []
            for platform, link in directory.social_links.items():
                if link:
                    social_info.append(f"{platform}: {link}")
            
            if social_info:
                content_sections.append(f"Social Media: {', '.join(social_info)}")
            
            content = "\n".join(content_sections)
            
            # Enhanced metadata for better filtering and sorting
            metadata = {
                "id": directory.id,
                "name": directory.name,
                "slug": directory.slug,
                "category": directory.category.lower() if directory.category else "",
                "subcategory": directory.subcategory.lower() if directory.subcategory else "",
                "founder_name": directory.founder_name,
                "verification_status": directory.verification_status,
                "likes": directory.likes,
                "views": directory.views,
                "created_at": directory.created_at,
                "url": directory.url,
                "has_social_links": bool(any(directory.social_links.values())),
                "content_type": "directory_entry"
            }
            
            doc = Document(
                page_content=content,
                metadata=metadata
            )
            documents.append(doc)
        
        logger.info(f"Created {len(documents)} enhanced documents")
        return documents
    
    def build_vector_store(self, documents: List[Document]):
        """
        Build FAISS vector store with optimized chunking
        
        Args:
            documents: List of Document objects
        """
        logger.info("Building vector store...")
        
        # Split documents into optimal chunks
        texts = self.text_splitter.split_documents(documents)
        
        # Create FAISS vector store
        self.vector_store = FAISS.from_documents(
            documents=texts,
            embedding=self.embeddings
        )
        
        logger.info(f"Vector store built with {len(texts)} document chunks")
    
    def setup_qa_chain(self):
        """Setup the QA chain with specialized prompt for directory queries"""
        
        # Specialized prompt template for directory and company queries
        prompt_template = """
        You are a knowledgeable assistant specializing in startup and company directory information. 
        Your role is to provide accurate, well-organized information about companies, founders, and business categories.

        Context Information:
        {context}

        User Question: {question}

        Instructions:
        1. **Category-wise Organization**: When asked for multiple companies, always group them by category (Web3, SaaS, FinTech, etc.)
        2. **Complete Information**: Include company name, founder, category, description, and verification status
        3. **Sorting Priority**: Present verified companies first, then sort by popularity (likes/views)
        4. **Format Guidelines**:
           - Use clear headings for categories
           - Number the companies within each category
           - Include founder information when available
           - Mention verification status
           - Add brief, relevant descriptions

        5. **Specific Query Handling**:
           - For "give me X companies" → group by category, show most relevant ones
           - For category requests → focus on that specific category
           - For individual company queries → provide comprehensive details
           - For founder queries → include their background and company details

        6. **Response Structure**:
           ```
           ## [Category Name]
           1. **Company Name** (Verification Status)
              - Founder: [Name]
              - Description: [Brief description]
              - Popularity: [Likes/Views if relevant]
           ```

        Answer:
        """
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create retrieval QA chain with optimized settings
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={
                    "k": 15,  # Retrieve more documents for better category coverage
                    "fetch_k": 30  # Fetch more candidates before filtering
                }
            ),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
        
        logger.info("QA chain configured successfully")
    
    def initialize_system(self, use_sample_data: bool = False, sample_data: List[Dict] = None):
        """
        Initialize the complete RAG system
        
        Args:
            use_sample_data: Whether to use provided sample data
            sample_data: Sample data to use if use_sample_data is True
        """
        logger.info("🚀 Initializing Directory RAG System...")
        
        # Step 1: Fetch or use provided data
        if use_sample_data and sample_data:
            raw_directories = sample_data
            logger.info(f"📊 Using provided sample data: {len(raw_directories)} entries")
        else:
            raw_directories = self.fetch_directories_from_api()
            
        if not raw_directories:
            raise ValueError("❌ No directory data available")
        
        # Step 2: Parse and structure data
        self.directories_data = self.parse_directory_data(raw_directories)
        
        # Step 3: Create enhanced documents
        documents = self.create_enhanced_documents(self.directories_data)
        
        # Step 4: Build vector store
        self.build_vector_store(documents)
        
        # Step 5: Setup QA chain
        self.setup_qa_chain()
        
        # Summary statistics
        categories = set(d.category for d in self.directories_data if d.category)
        verified_count = sum(1 for d in self.directories_data if d.verification_status == 'verified')
        
        logger.info("✅ RAG System Initialization Complete!")
        logger.info(f"📈 Statistics:")
        logger.info(f"   - Total Companies: {len(self.directories_data)}")
        logger.info(f"   - Categories: {len(categories)} ({', '.join(sorted(categories))})")
        logger.info(f"   - Verified Companies: {verified_count}")
        
        return {
            "status": "success",
            "total_companies": len(self.directories_data),
            "categories": list(categories),
            "verified_count": verified_count
        }
    
    def query_directories(self, question: str) -> Dict[str, Any]:
        """
        Query the RAG system for directory information
        
        Args:
            question: User question about directories/companies
            
        Returns:
            Comprehensive response with categorized information
        """
        if not self.qa_chain:
            raise ValueError("❌ RAG system not initialized. Call initialize_system() first.")
        
        try:
            logger.info(f"🔍 Processing query: {question}")
            
            # Get response from RAG chain
            result = self.qa_chain.invoke({"query": question})
            
            # Extract source information for transparency
            source_companies = []
            for doc in result.get("source_documents", []):
                metadata = doc.metadata
                source_companies.append({
                    "name": metadata.get("name", ""),
                    "category": metadata.get("category", ""),
                    "verification": metadata.get("verification_status", ""),
                    "snippet": doc.page_content[:150] + "..."
                })
            
            # Remove duplicates from sources
            unique_sources = []
            seen_names = set()
            for source in source_companies:
                if source["name"] and source["name"] not in seen_names:
                    unique_sources.append(source)
                    seen_names.add(source["name"])
            
            response = {
                "answer": result["result"],
                "source_companies": unique_sources[:10],  # Limit to top 10 sources
                "total_sources": len(unique_sources),
                "query_processed": question,
                "status": "success"
            }
            
            logger.info(f"✅ Query processed successfully. Found {len(unique_sources)} relevant companies")
            return response
            
        except Exception as e:
            logger.error(f"❌ Error processing query: {str(e)}")
            return {
                "answer": f"Sorry, I encountered an error while processing your query: {str(e)}",
                "source_companies": [],
                "total_sources": 0,
                "query_processed": question,
                "status": "error",
                "error": str(e)
            }
    
    def get_categories_summary(self) -> Dict[str, Any]:
        """Get a summary of all available categories and their company counts"""
        if not self.directories_data:
            return {"error": "No data available"}
        
        category_stats = {}
        for directory in self.directories_data:
            category = directory.category or "Uncategorized"
            if category not in category_stats:
                category_stats[category] = {
                    "count": 0,
                    "verified": 0,
                    "companies": []
                }
            
            category_stats[category]["count"] += 1
            if directory.verification_status == "verified":
                category_stats[category]["verified"] += 1
            
            category_stats[category]["companies"].append({
                "name": directory.name,
                "founder": directory.founder_name,
                "verified": directory.verification_status == "verified"
            })
        
        return {
            "total_categories": len(category_stats),
            "categories": category_stats,
            "total_companies": len(self.directories_data)
        }

# Global instance for the Flask app
rag_system = None