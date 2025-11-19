# backend/app/api/documents.py

import os
import shutil
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, BackgroundTasks
from pydantic import BaseModel, HttpUrl
from sqlalchemy.orm import Session
import uuid

from app.db.database import get_db
from app.db import models
from app.services.ingestion import process_document_task, delete_document_task
from typing import List

router = APIRouter()
UPLOAD_DIRECTORY = "./temp_uploads"
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

# --- Pydantic Models ---
class UrlRequest(BaseModel):
    url: HttpUrl

class DocumentResponse(BaseModel):
    id: uuid.UUID
    filename: str
    source_type: models.SourceType
    status: models.DocumentStatus
    upload_date: str
    chunk_count: int

    class Config:
        orm_mode = True

# --- Endpoints ---

@router.get("/api/documents", response_model=List[DocumentResponse])
async def get_documents(db: Session = Depends(get_db)):
    """Returns a list of all uploaded documents."""
    documents = db.query(models.Document).order_by(models.Document.upload_date.desc()).all()
    # Convert datetime to string for Pydantic
    for doc in documents:
        doc.upload_date = doc.upload_date.isoformat()
    return documents

@router.delete("/api/documents/{document_id}", status_code=204)
async def delete_document(
    document_id: uuid.UUID,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Deletes a document from the database, vector store, and file system.
    """
    document = db.query(models.Document).filter(models.Document.id == document_id).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    # Remove local file if it exists
    if document.source_type == models.SourceType.pdf and document.saved_file_path:
        if os.path.exists(document.saved_file_path):
            try:
                os.remove(document.saved_file_path)
            except OSError as e:
                print(f"Error deleting file {document.saved_file_path}: {e}")

    # Trigger background task to clean up Qdrant and BM25
    background_tasks.add_task(delete_document_task, str(document_id))

    # Delete metadata from Postgres immediately
    db.delete(document)
    db.commit()
    
    return None

@router.post("/api/documents/upload", status_code=202)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...), 
    db: Session = Depends(get_db)
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file name provided.")

    file_extension = os.path.splitext(file.filename)[1]
    saved_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = os.path.join(UPLOAD_DIRECTORY, saved_filename)

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    finally:
        file.file.close()

    new_document = models.Document(
        filename=file.filename,
        saved_file_path=file_path,
        source_type=models.SourceType.pdf,
        status=models.DocumentStatus.processing
    )
    db.add(new_document)
    db.commit()
    db.refresh(new_document)

    background_tasks.add_task(process_document_task, new_document.id)

    return {
        "message": "File accepted and is being processed in the background.",
        "document_id": new_document.id,
        "original_filename": new_document.filename,
    }

@router.post("/api/documents/youtube", status_code=202)
async def add_youtube_document(
    request: UrlRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    url = str(request.url)
    new_document = models.Document(
        filename=f"YouTube Video: {url}",
        source_url=url,
        source_type=models.SourceType.youtube,
        status=models.DocumentStatus.processing,
        saved_file_path=""
    )
    db.add(new_document)
    db.commit()
    db.refresh(new_document)
    background_tasks.add_task(process_document_task, new_document.id)
    return {
        "message": "YouTube URL accepted and is being processed in the background.",
        "document_id": new_document.id,
        "source_url": url,
    }

@router.post("/api/documents/web", status_code=202)
async def add_web_document(
    request: UrlRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Accepts a web page URL, creates a document record, and starts
    background processing to scrape the content.
    """
    url = str(request.url)
    
    new_document = models.Document(
        filename=f"Web Page: {url}",
        source_url=url,
        source_type=models.SourceType.web,
        status=models.DocumentStatus.processing,
        saved_file_path="" # Not applicable
    )
    db.add(new_document)
    db.commit()
    db.refresh(new_document)

    background_tasks.add_task(process_document_task, new_document.id)

    return {
        "message": "Web page URL accepted and is being processed in the background.",
        "document_id": new_document.id,
        "source_url": url,
    }
