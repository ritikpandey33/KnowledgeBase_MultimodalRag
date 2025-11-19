# backend/app/db/database.py

import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is not set. Please check your .env file.")

# Create the SQLAlchemy engine
# The connect_args are recommended for SQLite, but generally safe for other DBs.
# For PostgreSQL, they are not strictly necessary.
engine = create_engine(DATABASE_URL)

# Create a configured "Session" class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create a base class for our models to inherit from
Base = declarative_base()

# Dependency to get a DB session
def get_db():
    """
    FastAPI dependency that provides a database session for a single request.
    Ensures the session is always closed after the request.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
