# backend/app/services/retrieval.py

import os
from qdrant_client import QdrantClient, models
from typing import List, AsyncGenerator, Dict
from fastapi import Depends

from app.utils.embeddings import get_embedding_model, EmbeddingModel
from app.services.llm import get_llm_provider, LLMProvider
from app.services.bm25 import get_bm25_service, BM25Service

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION_NAME = "knowledge_base"

class RetrievalService:
    """
    Handles the retrieval of relevant document chunks from the Qdrant
    vector database and BM25 index based on a user's query.
    """
    
    # ... Templates ...

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

    def __init__(self, embedding_model: EmbeddingModel, llm_provider: LLMProvider, bm25_service: BM25Service):
        self.embedding_model = embedding_model
        self.llm_provider = llm_provider
        self.bm25_service = bm25_service
        self.qdrant_client = QdrantClient(url=QDRANT_URL)

    def _reciprocal_rank_fusion(self, vector_results: List[models.ScoredPoint], keyword_results: List[Dict], k: int = 60) -> List[Dict]:
        """
        Combines vector and keyword search results using Reciprocal Rank Fusion (RRF).
        Returns a list of payloads (dicts) sorted by RRF score.
        """
        scores = {} # {doc_id: score}
        doc_map = {} # {doc_id: payload}

        # Process Vector Results
        for rank, point in enumerate(vector_results):
            # Ensure payload exists
            if not point.payload: continue
            doc_id = point.payload.get("document_id")
            # Use chunk text as unique key if ID not sufficient, but ID should be chunk ID in future
            # For now, we rely on payload['text'] as unique identifier if doc_id is just parent doc
            # Let's assume text is unique enough for this MVP fusion
            unique_key = point.payload['text'] 
            
            if unique_key not in scores:
                scores[unique_key] = 0
                doc_map[unique_key] = point.payload
            
            scores[unique_key] += 1 / (k + rank + 1)

        # Process Keyword Results
        for rank, item in enumerate(keyword_results):
            unique_key = item["document"]["text"]
            
            if unique_key not in scores:
                scores[unique_key] = 0
                doc_map[unique_key] = item["document"] # Normalized to match payload structure
            
            scores[unique_key] += 1 / (k + rank + 1)

        # Sort by score
        sorted_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        
        # Return top 5 fused results
        return [doc_map[key] for key in sorted_keys[:5]]

    def _retrieve_relevant_chunks(self, query: str) -> List[Dict]:
        # 1. Vector Search
        query_embedding = self.embedding_model.encode_query(query)
        vector_results = self.qdrant_client.query_points(
            collection_name=QDRANT_COLLECTION_NAME,
            query=query_embedding,
            limit=10, # Fetch more for fusion
            with_payload=True,
        ).points
        
        # 2. Keyword Search
        keyword_results = self.bm25_service.search(query, k=10)
        
        # 3. Fusion
        fused_results = self._reciprocal_rank_fusion(vector_results, keyword_results)
        
        return fused_results

    async def generate_response(self, query: str, enhance_with_ai: bool = False) -> AsyncGenerator[str, None]:
        print(f"Retrieving chunks for query: '{query}'")
        relevant_chunks = self._retrieve_relevant_chunks(query)
        
        if not relevant_chunks:
            yield "I could not find any relevant information in the uploaded documents to answer your question."
            return

        context_str = "\n\n---\n\n".join([chunk['text'] for chunk in relevant_chunks])
        
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
    bm25_service: BM25Service = Depends(get_bm25_service),
) -> RetrievalService:
    return RetrievalService(embedding_model, llm_provider, bm25_service)
