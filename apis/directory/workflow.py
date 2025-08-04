import asyncio
from typing import Dict, List, TypedDict, Optional, Literal
from langgraph.graph import StateGraph, END
from langchain.schema import Document
from shared.llm import get_llm
from rag.directory_rag import DirectoryRAGSystem
import logging

logger = logging.getLogger(__name__)

class EnhancedWorkflowState(TypedDict):
    question: str
    question_type: str  # categorized, general, specific, comparison
    retrieved_docs: List[Dict]
    filtered_docs: List[Dict]
    answer: str
    confidence_score: float
    steps: List[str]
    metadata: Dict
    need_refinement: bool
    iteration_count: int

class OptimizedDirectoryWorkflow:
    def __init__(self, rag_instance: DirectoryRAGSystem = None):
        self.rag = rag_instance if rag_instance else DirectoryRAGSystem()
        self.llm = get_llm()
        self.graph = self._build_graph()
        self.max_iterations = 3
        
    def _build_graph(self):
        """Build the enhanced workflow graph"""
        workflow = StateGraph(EnhancedWorkflowState)
        
        # Add nodes
        workflow.add_node("analyze_question", self._analyze_question_node)
        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("filter_grade", self._filter_grade_node)
        workflow.add_node("generate", self._generate_node)
        workflow.add_node("evaluate", self._evaluate_node)
        workflow.add_node("refine", self._refine_node)
        
        # Add edges
        workflow.set_entry_point("analyze_question")
        workflow.add_edge("analyze_question", "retrieve")
        workflow.add_edge("retrieve", "filter_grade")
        
        workflow.add_conditional_edges(
            "filter_grade",
            self._should_generate,
            {
                "generate": "generate",
                "retrieve": "retrieve",  # If no good docs, try retrieve again
                "end": END
            }
        )
        
        workflow.add_edge("generate", "evaluate")
        
        workflow.add_conditional_edges(
            "evaluate",
            self._should_refine,
            {
                "refine": "refine",
                "end": END
            }
        )
        
        workflow.add_edge("refine", "generate")
        
        return workflow.compile()
    
    def _analyze_question_node(self, state: EnhancedWorkflowState) -> EnhancedWorkflowState:
        """Analyze and categorize the question"""
        question = state["question"].lower()
        
        # Categorize question type
        if any(word in question for word in ["category", "type of", "what kind"]):
            question_type = "categorized"
        elif any(word in question for word in ["compare", "vs", "versus", "difference"]):
            question_type = "comparison"
        elif any(word in question for word in ["list", "all", "companies", "show me"]):
            question_type = "general"
        else:
            question_type = "specific"
        
        state.update({
            "question_type": question_type,
            "metadata": {"original_question": state["question"]},
            "steps": ["Analyzed question type: " + question_type],
            "iteration_count": 0,
            "need_refinement": False
        })
        
        return state
    
    def _retrieve_node(self, state: EnhancedWorkflowState) -> EnhancedWorkflowState:
        """Enhanced retrieval with different strategies based on question type"""
        try:
            question_type = state.get("question_type", "general")
            question = state["question"]
            
            # Adjust retrieval parameters based on question type
            if question_type == "comparison":
                top_k = 8  # More docs for comparison
                use_compression = False
            elif question_type == "general":
                top_k = 10  # Many docs for general queries
                use_compression = True
            else:
                top_k = 5  # Focused results for specific queries
                use_compression = True
            
            # Perform retrieval
            result = self.rag.query(question, use_compression=use_compression, top_k=top_k)
            
            # Convert to document format for processing
            retrieved_docs = []
            for i, source_detail in enumerate(result.get("source_details", [])):
                retrieved_docs.append({
                    "content": f"Company: {source_detail.get('company', 'Unknown')}",
                    "metadata": source_detail,
                    "relevance_score": 1.0 - (i * 0.1)  # Decreasing relevance
                })
            
            state.update({
                "retrieved_docs": retrieved_docs,
                "confidence_score": result.get("confidence_score", 0.5),
                "steps": state["steps"] + [f"Retrieved {len(retrieved_docs)} documents"]
            })
            
        except Exception as e:
            logger.error(f"Retrieval error: {str(e)}")
            state.update({
                "retrieved_docs": [],
                "confidence_score": 0.0,
                "steps": state["steps"] + [f"Retrieval failed: {str(e)}"]
            })
        
        return state
    
    def _filter_grade_node(self, state: EnhancedWorkflowState) -> EnhancedWorkflowState:
        """Enhanced document filtering and grading"""
        retrieved_docs = state.get("retrieved_docs", [])
        question = state["question"]
        question_type = state.get("question_type", "general")
        
        if not retrieved_docs:
            state.update({
                "filtered_docs": [],
                "steps": state["steps"] + ["No documents to filter"]
            })
            return state
        
        # Filter based on question type and relevance
        filtered_docs = []
        
        for doc in retrieved_docs:
            relevance_score = self._calculate_document_relevance(doc, question, question_type)
            
            if relevance_score > 0.3:  # Threshold for relevance
                doc["relevance_score"] = relevance_score
                filtered_docs.append(doc)
        
        # Sort by relevance score
        filtered_docs.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        # Limit documents based on question type
        if question_type == "specific":
            filtered_docs = filtered_docs[:3]
        elif question_type == "comparison":
            filtered_docs = filtered_docs[:6]
        else:
            filtered_docs = filtered_docs[:5]
        
        state.update({
            "filtered_docs": filtered_docs,
            "steps": state["steps"] + [f"Filtered to {len(filtered_docs)} relevant documents"]
        })
        
        return state
    
    def _calculate_document_relevance(self, doc: Dict, question: str, question_type: str) -> float:
        """Calculate document relevance score"""
        content = doc.get("content", "").lower()
        question_lower = question.lower()
        
        # Base relevance from keyword matching
        question_words = set(question_lower.split())
        content_words = set(content.split())
        word_overlap = len(question_words & content_words)
        base_score = min(word_overlap / len(question_words), 1.0) if question_words else 0.0
        
        # Boost for verified companies
        metadata = doc.get("metadata", {})
        if metadata.get("verification") == "verified":
            base_score += 0.2
        
        # Boost based on question type
        if question_type == "categorized" and "category" in content:
            base_score += 0.15
        elif question_type == "comparison" and any(word in content for word in ["vs", "compare", "difference"]):
            base_score += 0.15
        
        return min(base_score, 1.0)
    
    def _generate_node(self, state: EnhancedWorkflowState) -> EnhancedWorkflowState:
        """Enhanced answer generation with context-aware prompting"""
        filtered_docs = state.get("filtered_docs", [])
        question = state["question"]
        question_type = state.get("question_type", "general")
        
        if not filtered_docs:
            state.update({
                "answer": "I don't have sufficient information to answer your question. Please try rephrasing or asking about specific companies in our directory.",
                "confidence_score": 0.0,
                "steps": state["steps"] + ["Generated fallback answer - no relevant documents"]
            })
            return state
        
        # Build context from filtered documents
        context_parts = []
        for i, doc in enumerate(filtered_docs):
            metadata = doc.get("metadata", {})
            context_parts.append(f"""
Document {i+1}:
Company: {metadata.get('company', 'Unknown')}
Category: {metadata.get('category', 'Unknown')}
Verification: {metadata.get('verification', 'Unknown')}
Content: {doc.get('content', '')}
""")
        
        context = "\n".join(context_parts)
        
        # Create question-type specific prompt
        prompt = self._create_contextual_prompt(question_type, context, question)
        
        try:
            response = self.llm.invoke(prompt)
            answer = response.content if hasattr(response, 'content') else str(response)
            
            state.update({
                "answer": answer,
                "steps": state["steps"] + ["Generated answer using filtered context"]
            })
            
        except Exception as e:
            logger.error(f"Generation error: {str(e)}")
            state.update({
                "answer": f"I encountered an error generating the answer: {str(e)}",
                "confidence_score": 0.0,
                "steps": state["steps"] + [f"Generation failed: {str(e)}"]
            })
        
        return state
    
    def _create_contextual_prompt(self, question_type: str, context: str, question: str) -> str:
        """Create context-aware prompts based on question type"""
        base_instruction = """You are an expert assistant for crypto and blockchain company information.
Use only the provided context to answer questions accurately and helpfully."""
        
        if question_type == "comparison":
            specific_instruction = """
For comparison questions:
- Compare the companies directly based on available information
- Highlight key differences and similarities
- Include verification status, categories, and notable features
- Be objective and factual"""
            
        elif question_type == "categorized":
            specific_instruction = """
For category-related questions:
- Group companies by their categories
- Provide clear categorization
- Include verification status for each company
- Mention key features of each category"""
            
        elif question_type == "specific":
            specific_instruction = """
For specific company questions:
- Provide detailed information about the requested company
- Include all available details: category, verification, features
- Be comprehensive but concise
- Mention team information if available"""
            
        else:  # general
            specific_instruction = """
For general questions:
- Provide a comprehensive overview
- Organize information logically
- Include verification status and categories
- Highlight notable companies or features"""
        
        prompt = f"""{base_instruction}

{specific_instruction}

Context:
{context}

Question: {question}

Answer:"""
        
        return prompt
    
    def _evaluate_node(self, state: EnhancedWorkflowState) -> EnhancedWorkflowState:
        """Evaluate answer quality and determine if refinement is needed"""
        answer = state.get("answer", "")
        question = state["question"]
        confidence_score = state.get("confidence_score", 0.5)
        iteration_count = state.get("iteration_count", 0)
        
        # Simple evaluation criteria
        needs_refinement = False
        
        # Check if answer is too short
        if len(answer.split()) < 20 and confidence_score > 0.3:
            needs_refinement = True
        
        # Check if answer doesn't address the question
        question_words = set(question.lower().split())
        answer_words = set(answer.lower().split())
        if len(question_words & answer_words) < 2:
            needs_refinement = True
        
        # Don't refine if we've already tried multiple times
        if iteration_count >= self.max_iterations:
            needs_refinement = False
        
        state.update({
            "need_refinement": needs_refinement,
            "steps": state["steps"] + [f"Evaluated answer quality - refinement needed: {needs_refinement}"]
        })
        
        return state
    
    def _refine_node(self, state: EnhancedWorkflowState) -> EnhancedWorkflowState:
        """Refine the answer based on evaluation feedback"""
        current_answer = state.get("answer", "")
        question = state["question"]
        filtered_docs = state.get("filtered_docs", [])
        iteration_count = state.get("iteration_count", 0)
        
        # Increment iteration count
        state["iteration_count"] = iteration_count + 1
        
        if not filtered_docs:
            state.update({
                "need_refinement": False,
                "steps": state["steps"] + ["Cannot refine - no source documents available"]
            })
            return state
        
        # Create refinement prompt
        context = "\n".join([doc.get("content", "") for doc in filtered_docs])
        
        refinement_prompt = f"""The previous answer needs improvement. Please provide a better, more comprehensive answer.

Previous Answer:
{current_answer}

Original Question: {question}

Context:
{context}

Please provide an improved answer that:
1. Is more detailed and comprehensive
2. Better addresses all aspects of the question
3. Uses specific information from the context
4. Is well-structured and easy to understand

Improved Answer:"""
        
        try:
            response = self.llm.invoke(refinement_prompt)
            refined_answer = response.content if hasattr(response, 'content') else str(response)
            
            state.update({
                "answer": refined_answer,
                "need_refinement": False,
                "steps": state["steps"] + [f"Refined answer (iteration {state['iteration_count']})"]
            })
            
        except Exception as e:
            logger.error(f"Refinement error: {str(e)}")
            state.update({
                "need_refinement": False,
                "steps": state["steps"] + [f"Refinement failed: {str(e)}"]
            })
        
        return state
    
    def _should_generate(self, state: EnhancedWorkflowState) -> Literal["generate", "retrieve", "end"]:
        """Decide whether to generate, retrieve more, or end"""
        filtered_docs = state.get("filtered_docs", [])
        iteration_count = state.get("iteration_count", 0)
        
        if not filtered_docs:
            if iteration_count < 2:
                return "retrieve"  # Try retrieving again
            else:
                return "end"  # Give up after 2 attempts
        
        return "generate"
    
    def _should_refine(self, state: EnhancedWorkflowState) -> Literal["refine", "end"]:
        """Decide whether to refine the answer or end"""
        need_refinement = state.get("need_refinement", False)
        iteration_count = state.get("iteration_count", 0)
        
        if need_refinement and iteration_count < self.max_iterations:
            return "refine"
        
        return "end"
    
    async def run(self, question: str) -> Dict:
        """Run the enhanced workflow"""
        if not hasattr(self.rag, 'vector_store') or not self.rag.vector_store:
            return {
                "answer": "Please load directory data first using ingest_directory_data()",
                "sources": [],
                "steps": ["No data loaded"],
                "confidence_score": 0.0,
                "question_type": "unknown"
            }
        
        initial_state: EnhancedWorkflowState = {
            "question": question,
            "question_type": "",
            "retrieved_docs": [],
            "filtered_docs": [],
            "answer": "",
            "confidence_score": 0.0,
            "steps": [],
            "metadata": {},
            "need_refinement": False,
            "iteration_count": 0
        }
        
        try:
            # Run the workflow
            result = await asyncio.get_event_loop().run_in_executor(
                None, self.graph.invoke, initial_state
            )
            
            # Extract source information
            sources = []
            for doc in result.get("filtered_docs", []):
                metadata = doc.get("metadata", {})
                source_id = metadata.get("id", "unknown")
                if source_id not in sources:
                    sources.append(source_id)
            
            return {
                "answer": result.get("answer", "No answer generated"),
                "sources": sources,
                "steps": result.get("steps", []),
                "confidence_score": result.get("confidence_score", 0.0),
                "question_type": result.get("question_type", "unknown"),
                "iterations": result.get("iteration_count", 0),
                "source_details": [doc.get("metadata", {}) for doc in result.get("filtered_docs", [])]
            }
            
        except Exception as e:
            logger.error(f"Workflow execution error: {str(e)}")
            return {
                "answer": f"Workflow execution failed: {str(e)}",
                "sources": [],
                "steps": ["Workflow failed"],
                "confidence_score": 0.0,
                "question_type": "error",
                "iterations": 0
            }

class AdvancedQueryProcessor:
    """Additional query processing utilities for enhanced functionality"""
    
    def __init__(self, rag_instance: DirectoryRAGSystem):
        self.rag = rag_instance
        self.llm = get_llm()
    
    async def process_multi_intent_query(self, question: str) -> Dict:
        """Process queries with multiple intents (e.g., "List DeFi companies and compare top 3")"""
        # Identify multiple intents
        intents = self._identify_query_intents(question)
        
        results = {}
        for intent_type, intent_query in intents.items():
            if intent_type == "list":
                results[intent_type] = await self._process_list_query(intent_query)
            elif intent_type == "compare":
                results[intent_type] = await self._process_comparison_query(intent_query)
            elif intent_type == "filter":
                results[intent_type] = await self._process_filter_query(intent_query)
        
        # Combine results
        combined_answer = self._combine_multi_intent_results(results, question)
        
        return {
            "answer": combined_answer,
            "sub_results": results,
            "intents_found": list(intents.keys())
        }
    
    def _identify_query_intents(self, question: str) -> Dict[str, str]:
        """Identify different intents in a complex query"""
        intents = {}
        question_lower = question.lower()
        
        # List intent
        if any(word in question_lower for word in ["list", "show", "all", "companies"]):
            intents["list"] = question
        
        # Comparison intent
        if any(word in question_lower for word in ["compare", "vs", "versus", "difference", "better"]):
            intents["compare"] = question
        
        # Filter intent
        if any(word in question_lower for word in ["category", "type", "verified", "partnership"]):
            intents["filter"] = question
        
        return intents
    
    async def _process_list_query(self, query: str) -> Dict:
        """Process list-type queries"""
        # Extract category if mentioned
        category_keywords = ["defi", "exchange", "wallet", "nft", "gaming", "dao"]
        detected_category = None
        
        for keyword in category_keywords:
            if keyword in query.lower():
                detected_category = keyword
                break
        
        if detected_category:
            companies = await self.rag.get_companies_by_category(detected_category)
        else:
            # General company listing
            result = self.rag.query(query, top_k=10)
            companies = result.get("source_details", [])
        
        return {
            "type": "list",
            "companies": companies,
            "category": detected_category
        }
    
    async def _process_comparison_query(self, query: str) -> Dict:
        """Process comparison queries"""
        result = self.rag.query(query, top_k=6, use_compression=False)
        
        return {
            "type": "comparison",
            "companies": result.get("source_details", []),
            "comparison_answer": result.get("answer", "")
        }
    
    async def _process_filter_query(self, query: str) -> Dict:
        """Process filter-based queries"""
        # Implementation for filtered queries
        result = self.rag.query(query, top_k=8)
        
        return {
            "type": "filter",
            "filtered_results": result.get("source_details", []),
            "filter_answer": result.get("answer", "")
        }
    
    def _combine_multi_intent_results(self, results: Dict, original_question: str) -> str:
        """Combine results from multiple intents into a coherent answer"""
        combined_parts = []
        
        if "list" in results:
            list_result = results["list"]
            companies = list_result.get("companies", [])
            if companies:
                combined_parts.append(f"Here are the companies I found:")
                for company in companies[:5]:  # Limit to top 5
                    combined_parts.append(f"• {company.get('name', 'Unknown')} ({company.get('category', 'Unknown category')})")
        
        if "compare" in results:
            compare_result = results["compare"]
            if compare_result.get("comparison_answer"):
                combined_parts.append("Comparison:")
                combined_parts.append(compare_result["comparison_answer"])
        
        if "filter" in results:
            filter_result = results["filter"]
            if filter_result.get("filter_answer"):
                combined_parts.append("Filtered results:")
                combined_parts.append(filter_result["filter_answer"])
        
        return "\n\n".join(combined_parts) if combined_parts else "I couldn't process your multi-part query effectively."