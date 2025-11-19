# backend/app/api/query.py

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict

from app.services.retrieval import RetrievalService
from app.services.llm import LLMProvider
from app.utils.embeddings import get_embedding_model, EmbeddingModel
from app.services.llm import get_llm_provider

router = APIRouter()

class QueryRequest(BaseModel):
    query: str
    enhance_with_ai: bool = False

class QueryResponse(BaseModel):
    text: str
    sources: List[Dict]

from app.services.bm25 import get_bm25_service, BM25Service

def get_retrieval_service(
    embedding_model: EmbeddingModel = Depends(get_embedding_model),
    llm_provider: LLMProvider = Depends(get_llm_provider),
    bm25_service: BM25Service = Depends(get_bm25_service),
) -> RetrievalService:
    """Dependency to create a RetrievalService instance."""
    return RetrievalService(embedding_model, llm_provider, bm25_service)

def _build_prompt(query: str, context_chunks: List[Dict]) -> str:
    """Builds a detailed prompt for the LLM with context and instructions."""
    
    context_str = "\n\n---\n\n".join([chunk["text"] for chunk in context_chunks])
    
    # The main instruction for the LLM
    prompt = f"""
    You are a helpful AI assistant. Based on the CONTEXT provided, answer the USER'S QUESTION.
    Your answer should be concise and directly address the question.
    If the context does not contain the answer, state that you could not find the information in the provided documents.
    Do not use any information outside of the provided context.

    CONTEXT:
    {context_str}

    USER'S QUESTION:
    {query}
    """
    return prompt

@router.post("/api/query")
async def query(
    request: QueryRequest,
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
) -> StreamingResponse:
    """
    Accepts a user's query, retrieves relevant context from the vector DB,
    and streams the LLM's response.
    """
    return StreamingResponse(
        retrieval_service.generate_response(request.query, request.enhance_with_ai), 
        media_type="text/event-stream"
    )
