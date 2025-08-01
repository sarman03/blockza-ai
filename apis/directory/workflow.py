from typing import Dict, List, TypedDict
from langgraph.graph import StateGraph, END
from shared.llm import get_llm
from .rag import DirectoryRAG

class WorkflowState(TypedDict):
    question: str
    retrieved_docs: List[str]
    answer: str
    steps: List[str]

class DirectoryWorkflow:
    def __init__(self, rag_instance: DirectoryRAG = None):
        # Use provided RAG instance or create new one
        self.rag = rag_instance if rag_instance else DirectoryRAG()
        self.llm = get_llm()
        self.graph = self._build_graph()
        
    def _build_graph(self):
        workflow = StateGraph(WorkflowState)
        
        # Add nodes
        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("grade", self._grade_node)
        workflow.add_node("generate", self._generate_node)
        
        # Add edges
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "grade")
        workflow.add_conditional_edges(
            "grade",
            self._should_generate,
            {
                "generate": "generate",
                "retrieve": "retrieve"
            }
        )
        workflow.add_edge("generate", END)
        
        return workflow.compile()
    
    def _retrieve_node(self, state: WorkflowState) -> WorkflowState:
        """Retrieve relevant documents"""
        if not self.rag.vectorstore:
            state["retrieved_docs"] = []
            state["steps"] = state.get("steps", []) + ["No documents available"]
            return state
            
        retriever = self.rag.vectorstore.as_retriever(search_kwargs={"k": 3})
        docs = retriever.get_relevant_documents(state["question"])
        
        state["retrieved_docs"] = [doc.page_content for doc in docs]
        state["steps"] = state.get("steps", []) + ["Retrieved documents"]
        return state
    
    def _grade_node(self, state: WorkflowState) -> WorkflowState:
        """Grade document relevance"""
        # Simple grading - in production, use LLM to grade
        relevant_docs = []
        for doc in state["retrieved_docs"]:
            if any(word.lower() in doc.lower() for word in state["question"].split()):
                relevant_docs.append(doc)
        
        state["retrieved_docs"] = relevant_docs
        state["steps"] = state.get("steps", []) + ["Graded document relevance"]
        return state
    
    def _generate_node(self, state: WorkflowState) -> WorkflowState:
        """Generate answer from retrieved documents"""
        if not state["retrieved_docs"]:
            state["answer"] = "No relevant documents found to answer the question."
            state["steps"] = state.get("steps", []) + ["No relevant documents"]
            return state
            
        context = "\n\n".join(state["retrieved_docs"])
        prompt = f"""Answer the question based on the context below:

Context:
{context}

Question: {state['question']}

Answer:"""
        
        response = self.llm.invoke(prompt)
        state["answer"] = response.content
        state["steps"] = state.get("steps", []) + ["Generated answer"]
        return state
    
    def _should_generate(self, state: WorkflowState) -> str:
        """Decide whether to generate or retrieve more"""
        if len(state["retrieved_docs"]) > 0:
            return "generate"
        return "retrieve"
    
    async def run(self, question: str) -> Dict:
        """Run the workflow"""
        if not self.rag.vectorstore:
            return {
                "answer": "Please load directory data first using ingest_directory_data()",
                "sources": [],
                "steps": ["No data loaded"]
            }
            
        initial_state = {
            "question": question,
            "retrieved_docs": [],
            "answer": "",
            "steps": []
        }
        
        result = self.graph.invoke(initial_state)
        
        return {
            "answer": result["answer"],
            "sources": ["directory_api"],
            "steps": result["steps"]
        }
