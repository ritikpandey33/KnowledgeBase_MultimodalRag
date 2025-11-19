import os
import pickle
import re
from typing import List, Dict
from rank_bm25 import BM25Okapi

BM25_INDEX_PATH = "bm25_index.pkl"

class BM25Service:
    """
    Manages the BM25 index for keyword-based search.
    Persists the index and the corresponding documents to a pickle file.
    """
    
    def __init__(self):
        self.bm25 = None
        self.documents = [] # List of Dicts: [{"text": "...", "metadata": {...}}, ...]
        self.load_index()

    def load_index(self):
        """Loads the BM25 index and documents from disk if they exist."""
        if os.path.exists(BM25_INDEX_PATH):
            try:
                with open(BM25_INDEX_PATH, "rb") as f:
                    data = pickle.load(f)
                    self.documents = data.get("documents", [])
                    corpus = [self._tokenize(doc["text"]) for doc in self.documents]
                    if corpus:
                        self.bm25 = BM25Okapi(corpus)
                    print(f"Loaded BM25 index with {len(self.documents)} chunks.")
            except Exception as e:
                print(f"Failed to load BM25 index: {e}")
                self.documents = []
                self.bm25 = None

    def save_index(self):
        """Saves the current documents list to disk (rebuilds index on load)."""
        try:
            with open(BM25_INDEX_PATH, "wb") as f:
                # We only save documents, we rebuild BM25 object on load/update
                pickle.dump({"documents": self.documents}, f)
            print("Saved BM25 index to disk.")
        except Exception as e:
            print(f"Failed to save BM25 index: {e}")

    def add_documents(self, chunks: List[str], metadatas: List[Dict]):
        """
        Adds new documents to the index and rebuilds it.
        NOTE: BM25 is an in-memory index. For large datasets, this is slow.
        For this project scale, rebuilding is acceptable.
        """
        new_docs = [{"text": chunk, "metadata": meta} for chunk, meta in zip(chunks, metadatas)]
        self.documents.extend(new_docs)
        
        # Rebuild index
        tokenized_corpus = [self._tokenize(doc["text"]) for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        self.save_index()

    def delete_documents(self, document_id: str):
        """
        Removes all chunks associated with a specific document_id and rebuilds the index.
        """
        initial_count = len(self.documents)
        # Filter out documents matching the ID
        self.documents = [
            doc for doc in self.documents 
            if doc["metadata"].get("document_id") != document_id
        ]
        
        if len(self.documents) < initial_count:
            print(f"Removed {initial_count - len(self.documents)} chunks from BM25 index for doc {document_id}.")
            
            # Rebuild index
            if self.documents:
                tokenized_corpus = [self._tokenize(doc["text"]) for doc in self.documents]
                self.bm25 = BM25Okapi(tokenized_corpus)
            else:
                self.bm25 = None
                
            self.save_index()
        else:
            print(f"No chunks found in BM25 index for doc {document_id}.")

    def search(self, query: str, k: int = 5) -> List[Dict]:
        """
        Performs keyword search using BM25.
        Returns a list of documents with scores.
        """
        if not self.bm25:
            return []

        tokenized_query = self._tokenize(query)
        # get_scores returns a list of scores for every document in the corpus
        scores = self.bm25.get_scores(tokenized_query)
        
        # Pair scores with documents
        scored_docs = []
        for i, score in enumerate(scores):
            if score > 0: # Filter out zero relevance
                scored_docs.append({
                    "document": self.documents[i],
                    "score": score
                })
        
        # Sort by score desc and take top k
        scored_docs.sort(key=lambda x: x["score"], reverse=True)
        return scored_docs[:k]

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenizer: lowercase and remove punctuation."""
        text = text.lower()
        # Remove special characters
        text = re.sub(r'[^a-z0-9\s]', '', text)
        return text.split()

# Singleton instance
bm25_service = BM25Service()

def get_bm25_service():
    return bm25_service
