import os
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import our LLM service and the new routers
from app.services.llm import get_llm_provider
from app.api import documents, query

app = FastAPI(
    title="Smart Knowledge Assistant API",
    description="API for the Smart Knowledge Assistant project.",
    version="0.1.0",
)

# Include the routers
app.include_router(documents.router)
app.include_router(query.router)

# Initialize the LLM provider on startup
# This is done at the module level so it runs once when the app starts
try:
    llm_provider = get_llm_provider()
    print(f"Successfully initialized LLM provider: {llm_provider.__class__.__name__}")
except (ValueError, ImportError) as e:
    print(f"Error initializing LLM provider: {e}")
    llm_provider = None

@app.get("/")
def read_root():
    """
    A simple endpoint to confirm the API is running.
    """
    provider_status = os.getenv('LLM_PROVIDER', 'not set')
    return {
        "status": "ok",
        "message": "Welcome to the Smart Knowledge Assistant API!",
        "llm_provider_configured": provider_status,
        "llm_provider_initialized": llm_provider is not None
    }

@app.get("/api/test-llm")
async def test_llm_stream():
    """
    A test endpoint to verify the LLM provider is working.
    Streams a response from the configured provider asynchronously.
    """
    if not llm_provider:
        return {"error": "LLM provider is not initialized. Check your .env configuration and logs."}

    prompt = "Hello! Tell me a short story about a robot."
    
    # The application code calls the async 'generate' method.
    # FastAPI's StreamingResponse can handle the async generator directly.
    response_generator = llm_provider.generate(prompt)
    
    return StreamingResponse(response_generator, media_type="text/plain")

# To run this app:
# 1. Make sure you are in the 'backend' directory.
# 2. Run the command: uvicorn app.main:app --reload