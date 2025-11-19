# backend/app/utils/embeddings.py

from sentence_transformers import SentenceTransformer
from typing import List

class EmbeddingModel:
    """
    A wrapper class for the SentenceTransformer model to ensure it is loaded
    only once and can be easily used throughout the application.
    """
    _model = None

    @classmethod
    def get_model(cls):
        """
        Singleton pattern to load the model only once.
        """
        if cls._model is None:
            # This is the model specified in the plan. It's small and efficient.
            model_name = 'all-MiniLM-L6-v2'
            print(f"Loading sentence transformer model: {model_name}...")
            cls._model = SentenceTransformer(model_name)
            print("Model loaded successfully.")
        return cls._model

    def encode_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Encodes a list of text documents into vector embeddings.
        """
        model = self.get_model()
        print(f"Creating embeddings for {len(texts)} documents...")
        embeddings = model.encode(texts, convert_to_tensor=False)
        print("Embeddings created.")
        return embeddings.tolist()

    def encode_query(self, text: str) -> List[float]:
        """
        Encodes a single query string into a vector embedding.
        """
        model = self.get_model()
        embedding = model.encode(text, convert_to_tensor=False)
        return embedding.tolist()

# Instantiate a single instance to be used as a dependency
embedding_model = EmbeddingModel()

def get_embedding_model():
    """
    Dependency function to get the embedding model instance.
    """
    return embedding_model
