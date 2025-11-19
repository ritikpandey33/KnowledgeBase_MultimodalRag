# backend/app/db/models.py

import uuid
from sqlalchemy import Column, String, DateTime, Integer, Enum, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
import enum

from .database import Base

class SourceType(str, enum.Enum):
    pdf = "pdf"
    youtube = "youtube"
    web = "web"
    text = "text"

class DocumentStatus(str, enum.Enum):
    processing = "processing"
    completed = "completed"
    failed = "failed"

class Document(Base):
    __tablename__ = "documents"

    # Use PostgreSQL's UUID type for the primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    user_id = Column(String, default="default_user", nullable=False)
    filename = Column(String, nullable=False)
    saved_file_path = Column(String, nullable=False)
    
    source_type = Column(Enum(SourceType), nullable=False)
    source_url = Column(String, nullable=True)
    
    upload_date = Column(DateTime(timezone=True), server_default=func.now())
    
    chunk_count = Column(Integer, default=0)
    status = Column(Enum(DocumentStatus), default=DocumentStatus.processing, nullable=False)
    
    # Use JSONB for metadata if using PostgreSQL, otherwise JSON
    # For simplicity, we use the generic JSON type here.
    meta_data = Column(JSON, nullable=True)

    def __repr__(self):
        return f"<Document(id={self.id}, filename='{self.filename}', status='{self.status}')>"
