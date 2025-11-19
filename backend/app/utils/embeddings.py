# backend/app/utils/embeddings.py

import os
import google.generativeai as genai
from typing import List
from dotenv import load_dotenv

load_dotenv()

class EmbeddingModel:
    """
    A wrapper class for the Gemini Embedding API.
    This replaces the local SentenceTransformer to save RAM on the server.
    """
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set in environment variables.")
        genai.configure(api_key=api_key)
        self.model_name = "models/text-embedding-004"

    def encode_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Encodes a list of text documents into vector embeddings using Gemini API.
        """
        print(f"Generating embeddings for {len(texts)} chunks using Gemini API...")
        
        # Gemini API accepts a list of strings directly
        result = genai.embed_content(
            model=self.model_name,
            content=texts,
            task_type="retrieval_document",
            title="Document Chunk" # Optional, helps with quality
        )
        
        # The result is a dictionary with 'embedding' key
        return result['embedding']

    def encode_query(self, text: str) -> List[float]:
        """
        Encodes a single query string into a vector embedding using Gemini API.
        """
        result = genai.embed_content(
            model=self.model_name,
            content=text,
            task_type="retrieval_query"
        )
        return result['embedding']

# Instantiate a single instance to be used as a dependency
try:
    embedding_model = EmbeddingModel()
except ValueError as e:
    print(f"Warning: {e}. Embeddings will fail until GEMINI_API_KEY is set.")
    embedding_model = None

def get_embedding_model():
    """
    Dependency function to get the embedding model instance.
    """
    if embedding_model is None:
         raise ValueError("Embedding model failed to initialize. Check GEMINI_API_KEY.")
    return embedding_model