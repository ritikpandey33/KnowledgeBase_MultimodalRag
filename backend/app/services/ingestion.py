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
from playwright.sync_api import sync_playwright

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
            self.qdrant_client.get_collection(collection_name=QDRANT_COLLECTION_NAME)
        except Exception:
            self.qdrant_client.create_collection(
                collection_name=QDRANT_COLLECTION_NAME,
                vectors_config=qdrant_models.VectorParams(size=384, distance=qdrant_models.Distance.COSINE),
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
        Extracts the main text content from a web page by first rendering it
        with a headless browser (Playwright) and then extracting content
        with trafilatura.
        """
        print(f"Scraping text from JS-heavy page: {url}")
        html_content = ""
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                page.goto(url, wait_until='networkidle', timeout=15000)
                html_content = page.content()
                browser.close()
        except Exception as e:
            print(f"Playwright failed to render page {url}: {e}")
            return self._extract_text_from_web_fallback(url)

        if not html_content:
            return ""

        print("Page rendered successfully. Extracting content with trafilatura...")
        try:
            text = trafilatura.extract(
                html_content,
                include_tables=True,
                include_comments=False,
                output_format='markdown'
            )
            if not text:
                raise ValueError("Trafilatura failed to extract content from rendered HTML.")
            
            print(f"Extracted {len(text)} characters from web page.")
            return text
        except Exception as e:
            print(f"Error extracting content with trafilatura from {url}: {e}")
            return ""

    def _extract_text_from_web_fallback(self, url: str) -> str:
        """A simple fallback scraper that doesn't use a headless browser."""
        print(f"Falling back to simple scraper for: {url}")
        try:
            downloaded = trafilatura.fetch_url(url)
            if not downloaded: return ""
            text = trafilatura.extract(downloaded, include_tables=True, include_comments=False, output_format='markdown')
            return text or ""
        except Exception as e:
            print(f"Fallback scraper failed for {url}: {e}")
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


# --- Standalone Background Task ---
def process_document_task(document_id: str):
    db = SessionLocal()
    try:
        embedding_model = get_embedding_model()
        service = IngestionService(db, embedding_model)
        service.process_document(document_id)
    finally:
        db.close()