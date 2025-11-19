# backend/app/services/retrieval.py

import os
from qdrant_client import QdrantClient, models
from typing import List, AsyncGenerator
from fastapi import Depends

from app.utils.embeddings import get_embedding_model, EmbeddingModel
from app.services.llm import get_llm_provider, LLMProvider

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION_NAME = "knowledge_base"

class RetrievalService:
    """
    Handles the retrieval of relevant document chunks from the Qdrant
    vector database based on a user's query.
    """

    STRICT_PROMPT_TEMPLATE = """
Use the following context to answer the user's question.
If you cannot find the answer in the provided context, simply state that you could not find the information in the provided documents.
Do not use any external knowledge or make up information.

Context:
{context}

Question:
{question}
"""

    HYBRID_PROMPT_TEMPLATE = """
You are a helpful AI assistant. Use the following context from user-provided documents as your primary source of truth to answer the user's question.
You may supplement your answer with your own general knowledge to provide a more complete and helpful response.
If the provided context directly contradicts your general knowledge, you must prioritize the information from the context.

Context:
{context}

Question:
{question}
"""

    def __init__(self, embedding_model: EmbeddingModel, llm_provider: LLMProvider):
        self.embedding_model = embedding_model
        self.llm_provider = llm_provider
        self.qdrant_client = QdrantClient(url=QDRANT_URL)

    def _retrieve_relevant_chunks(self, query: str) -> List[models.ScoredPoint]:
        query_embedding = self.embedding_model.encode_query(query)
        
        # Use query_points instead of search for newer Qdrant client versions
        search_result = self.qdrant_client.query_points(
            collection_name=QDRANT_COLLECTION_NAME,
            query=query_embedding,
            limit=5,
            with_payload=True,
        ).points
        
        return search_result

    async def generate_response(self, query: str, enhance_with_ai: bool = False) -> AsyncGenerator[str, None]:
        print(f"Retrieving chunks for query: '{query}'")
        relevant_chunks = self._retrieve_relevant_chunks(query)
        
        if not relevant_chunks:
            yield "I could not find any relevant information in the uploaded documents to answer your question."
            return

        context_str = "\n\n---\n\n".join([chunk.payload['text'] for chunk in relevant_chunks])
        
        if enhance_with_ai:
            print("Using HYBRID prompt template.")
            prompt = self.HYBRID_PROMPT_TEMPLATE.format(context=context_str, question=query)
        else:
            print("Using STRICT prompt template.")
            prompt = self.STRICT_PROMPT_TEMPLATE.format(context=context_str, question=query)
            
        async for chunk in self.llm_provider.generate(prompt):
            yield chunk

# --- Dependency Injection ---
def get_retrieval_service(
    embedding_model: EmbeddingModel = Depends(get_embedding_model),
    llm_provider: LLMProvider = Depends(get_llm_provider),
) -> RetrievalService:
    return RetrievalService(embedding_model, llm_provider)
