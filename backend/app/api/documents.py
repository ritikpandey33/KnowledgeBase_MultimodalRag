# backend/app/api/documents.py

import os
import shutil
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, BackgroundTasks
from pydantic import BaseModel, HttpUrl
from sqlalchemy.orm import Session
import uuid

from app.db.database import get_db
from app.db import models
from app.services.ingestion import process_document_task

router = APIRouter()
UPLOAD_DIRECTORY = "./temp_uploads"
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

# --- Pydantic Models ---
class UrlRequest(BaseModel):
    url: HttpUrl

# --- Endpoints ---

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
