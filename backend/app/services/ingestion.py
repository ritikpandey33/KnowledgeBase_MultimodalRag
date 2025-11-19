# backend/app/services/ingestion.py
import os
import uuid
import re
from urllib.parse import urlparse, parse_qs
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy.orm import Session
from fastapi import Depends
from qdrant_client import QdrantClient, models as qdrant_models
from youtube_transcript_api import YouTubeTranscriptApi
import trafilatura

from app.db.models import Document, DocumentStatus, SourceType
from app.db.database import get_db, SessionLocal
from app.utils.embeddings import get_embedding_model, EmbeddingModel
from app.services.bm25 import get_bm25_service

# --- Qdrant Configuration ---
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION_NAME = "knowledge_base"


class IngestionService:
    """
    Handles the processing of documents to extract, chunk, embed, and store
    text in the vector database.
    """

    def __init__(self, db: Session, embedding_model: EmbeddingModel):
        self.db = db
        self.embedding_model = embedding_model
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            is_separator_regex=False,
        )
        self.qdrant_client = QdrantClient(url=QDRANT_URL)
        self._ensure_qdrant_collection_exists()

    def _ensure_qdrant_collection_exists(self):
        try:
            collection_info = self.qdrant_client.get_collection(collection_name=QDRANT_COLLECTION_NAME)
            # Check if vector size matches Gemini's 768
            if collection_info.config.params.vectors.size != 768:
                print(f"Collection '{QDRANT_COLLECTION_NAME}' has incorrect vector size. Recreating...")
                self.qdrant_client.delete_collection(collection_name=QDRANT_COLLECTION_NAME)
                raise Exception("Collection deleted to force recreation.")
        except Exception:
            self.qdrant_client.create_collection(
                collection_name=QDRANT_COLLECTION_NAME,
                vectors_config=qdrant_models.VectorParams(size=768, distance=qdrant_models.Distance.COSINE),
            )

    def _extract_text_from_pdf(self, file_path: str) -> str:
        text = ""
        with open(file_path, "rb") as f:
            reader = PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() or ""
        return text

    def _extract_text_from_youtube(self, video_url: str) -> str:
        parsed_url = urlparse(video_url)
        if "youtube.com" in parsed_url.netloc:
            query_params = parse_qs(parsed_url.query)
            video_id = query_params["v"][0]
        elif "youtu.be" in parsed_url.netloc:
            video_id = parsed_url.path[1:]
        else:
            raise ValueError("Not a valid YouTube URL.")
        transcript_list = YouTubeTranscriptApi().fetch(video_id)
        return " ".join([item.text for item in transcript_list])

    def _extract_text_from_web(self, url: str) -> str:
        """
        Extracts the main text content from a web page using trafilatura.
        This replaces the complex Playwright scraper.
        """
        print(f"Scraping text from web page: {url}")
        try:
            downloaded = trafilatura.fetch_url(url)
            if not downloaded:
                raise ValueError("Failed to fetch URL content.")
            
            text = trafilatura.extract(
                downloaded,
                include_tables=True,
                include_comments=False,
                output_format='markdown'
            )
            
            if not text:
                raise ValueError("Failed to extract text from fetched content.")
                
            print(f"Extracted {len(text)} characters from web page.")
            return text
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            return ""

    def process_document(self, document_id: str):
        document = self.db.query(Document).filter(Document.id == document_id).first()
        if not document:
            return

        try:
            text = ""
            if document.source_type == SourceType.pdf:
                text = self._extract_text_from_pdf(document.saved_file_path)
            elif document.source_type == SourceType.youtube:
                text = self._extract_text_from_youtube(document.source_url)
            elif document.source_type == SourceType.web:
                text = self._extract_text_from_web(document.source_url)
            
            if not text:
                raise ValueError("Text extraction failed or returned no content.")

            chunks = self.text_splitter.split_text(text)
            embeddings = self.embedding_model.encode_documents(chunks)

            # Prepare payloads for Qdrant and metadata for BM25
            payloads = [{"text": chunk, "document_id": str(document.id), "source_filename": document.filename} for chunk in chunks]

            self.qdrant_client.upsert(
                collection_name=QDRANT_COLLECTION_NAME,
                points=qdrant_models.Batch(
                    ids=[str(uuid.uuid4()) for _ in chunks],
                    vectors=embeddings,
                    payloads=payloads
                ),
                wait=True
            )

            # Update BM25 Index
            print("Updating BM25 index...")
            bm25_service = get_bm25_service()
            bm25_service.add_documents(chunks, payloads)
            
            document.chunk_count = len(chunks)
            document.status = DocumentStatus.completed
            self.db.commit()
            print(f"Successfully processed document {document.id}.")

        except Exception as e:
            document.status = DocumentStatus.failed
            self.db.commit()
            print(f"An error occurred during document processing for {document.id}: {e}")

    def delete_document(self, document_id: str):
        """
        Deletes a document's vectors from Qdrant and removes it from the BM25 index.
        """
        print(f"Deleting document {document_id} from vector database...")
        
        # 1. Delete from Qdrant using Filter
        try:
            self.qdrant_client.delete(
                collection_name=QDRANT_COLLECTION_NAME,
                points_selector=qdrant_models.FilterSelector(
                    filter=qdrant_models.Filter(
                        must=[
                            qdrant_models.FieldCondition(
                                key="document_id",
                                match=qdrant_models.MatchValue(value=document_id),
                            )
                        ]
                    )
                ),
            )
            print("Deleted vectors from Qdrant.")
        except Exception as e:
            print(f"Error deleting from Qdrant: {e}")

        # 2. Delete from BM25
        print(f"Deleting document {document_id} from BM25 index...")
        try:
            bm25_service = get_bm25_service()
            bm25_service.delete_documents(document_id)
        except Exception as e:
            print(f"Error deleting from BM25: {e}")


# --- Standalone Background Tasks ---
def process_document_task(document_id: str):
    db = SessionLocal()
    try:
        embedding_model = get_embedding_model()
        service = IngestionService(db, embedding_model)
        service.process_document(document_id)
    finally:
        db.close()

def delete_document_task(document_id: str):
    db = SessionLocal()
    try:
        # We don't strictly need the embedding model for deletion, but the service requires it in init
        # Ideally we'd refactor IngestionService to not require it for deletion, but this works for now.
        embedding_model = get_embedding_model()
        service = IngestionService(db, embedding_model)
        service.delete_document(document_id)
    finally:
        db.close()