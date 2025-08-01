import requests
from typing import Dict, List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from shared.llm import get_llm
from shared.vector_store import get_vector_store
from config import DIRECTORY_API_URL

class DirectoryRAG:
    def __init__(self):
        self.vector_store = get_vector_store()
        self.embedding_model = OpenAIEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    async def ingest_directory_data(self):
        response = requests.get(DIRECTORY_API_URL)
        response.raise_for_status()
        if not response.text:
            data = []
        else:
            try:
                data = response.json()
            except:
                data = [{"content": response.text}]

        if not isinstance(data, list):
            if isinstance(data, dict):
                data = [data]
            else:
                data = [{"content": str(data)}]

        docs: List[Document] = []
        for i, item in enumerate(data):
            if isinstance(item, dict):
                text = self._json_to_text(item)
                source_id = item.get("_id", f"item_{i}")
            else:
                text = str(item)
                source_id = f"item_{i}"

            if not text.strip():
                print(f"Skipping empty text for item {source_id}")
                continue

            docs.append(Document(page_content=text, metadata={
                "source": "directory_api",
                "id": source_id,
                "company_name": item.get("name", "") if isinstance(item, dict) else "",
                "category": item.get("category", "") if isinstance(item, dict) else ""
            }))

        if not docs:
            print("No valid documents found to ingest. Skipping.")
            return

        splits = self.text_splitter.split_documents(docs)
        if not splits:
            raise ValueError("Text splitter returned no chunks — check your content or chunk settings")

        # Validate embeddings on a small sample
        sample_texts = [splits[0].page_content]
        sample_embeds = self.embedding_model.embed_documents(sample_texts)
        if not sample_embeds or not isinstance(sample_embeds[0], list):
            raise ValueError("Embedding model returned empty or invalid embedding vector")

        self.vector_store.add_documents(splits)

    def _json_to_text(self, item: Dict) -> str:
        lines = []
        def add(label, val):
            if val:
                lines.append(f"{label}: {val}")
        add("Company Name", item.get("name"))
        add("Category", item.get("category"))
        add("Short Description", item.get("shortDescription"))
        add("Detailed Description", item.get("detail"))
        add("Website", item.get("url"))
        add("Verification", item.get("verificationStatus"))
        add("Founder Name", item.get("founderName"))
        add("Founder Details", item.get("founderDetails"))
        for plat, url in item.get("socialLinks", {}).items():
            add(f"{plat.capitalize()} Link", url)
        for m in item.get("teamMembers", []):
            add(f"Team Member: {m.get('name')} — {m.get('title')}", "")
            if m.get("linkedinUrl"):
                add("LinkedIn", m.get("linkedinUrl"))
        for k, v in item.get("promotionSettings", {}).items():
            label = k.replace("interestedIn", "Interested in ").replace("has", "Has ").title()
            add(label, v)
        return "\n".join(lines)

    def query(self, question: str) -> Dict:
        prompt_template = """
You are an expert assistant answering questions about verified crypto and blockchain‑related companies.
Use the context below to respond precisely and helpfully.
Only answer from the given context. If the information is not available, respond with "I don't know".

Context:
{context}

Question: {question}

Answer:
"""
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        qa = RetrievalQA.from_chain_type(
            llm=get_llm(),
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )
        result = qa({"query": question})
        return {"answer": result["result"], "sources": [doc.metadata.get("source", "") for doc in result["source_documents"]]}
