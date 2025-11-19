# backend/app/services/llm.py

import os
from abc import ABC, abstractmethod
from typing import AsyncGenerator

# --- 1. The Abstraction (Interface) ---

class LLMProvider(ABC):
    """
    An abstract base class for all LLM providers.
    It ensures that any provider we use has an async 'generate' method
    that supports streaming.
    """
    @abstractmethod
    async def generate(self, prompt: str, history: list = None) -> AsyncGenerator[str, None]:
        """
        Generates a streaming response from the LLM asynchronously.
        Must be implemented by all concrete provider classes.
        """
        # This is an async generator, so it must be implemented with 'async for'
        # and 'yield'. We add a dummy yield here to satisfy the type checker.
        yield ""

# --- 2. Concrete Implementations ---

class OpenAIProvider(LLMProvider):
    """Concrete implementation for OpenAI's API using its async client."""
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        try:
            import openai
            # Use the AsyncOpenAI client
            self.client = openai.AsyncOpenAI(api_key=api_key)
            self.model = model
        except ImportError:
            raise ImportError("The 'openai' library is required to use the OpenAIProvider. Please install it with 'pip install openai'.")

    async def generate(self, prompt: str, history: list = None) -> AsyncGenerator[str, None]:
        print(f"Using OpenAI ({self.model}) to generate response...")
        messages = [{"role": "user", "content": prompt}]
        
        # Use 'await' for the async call
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
        )
        # Use 'async for' to iterate over the async stream
        async for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                yield content

class GeminiProvider(LLMProvider):
    """Concrete implementation for Google's Gemini API using its async methods."""
    def __init__(self, api_key: str, model: str = "gemini-2.5-pro"):
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model)
        except ImportError:
            raise ImportError("The 'google-generativeai' library is required to use the GeminiProvider. Please install it with 'pip install google-generativeai'.")

    async def generate(self, prompt: str, history: list = None) -> AsyncGenerator[str, None]:
        print(f"Using Gemini ({self.model.model_name}) to generate response...")
        
        response = await self.model.generate_content_async(prompt, stream=True)
        
        async for chunk in response:
            # Safety check: Ensure the chunk has content before accessing .text
            if chunk.parts:
                yield chunk.text

class MockProvider(LLMProvider):
    """A mock provider for testing and development without using real API keys."""
    async def generate(self, prompt: str, history: list = None) -> AsyncGenerator[str, None]:
        import asyncio
        print("Using MockProvider for testing...")
        mock_response = f"This is a mock response to the prompt: '{prompt}'"
        for word in mock_response.split():
            yield word + " "
            await asyncio.sleep(0.05) # Simulate network latency


# --- 3. The Factory Function (remains synchronous) ---

def get_llm_provider() -> LLMProvider:
    """
    Factory function to get the configured LLM provider.
    Reads the LLM_PROVIDER from the environment and returns an
    instantiated provider object, with strict validation.
    """
    provider_name = os.getenv("LLM_PROVIDER", "mock").lower()
    
    if provider_name == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key == "sk-your-openai-key-here":
            raise ValueError(
                "LLM_PROVIDER is set to 'openai' but OPENAI_API_KEY is missing or not set in your .env file."
            )
        return OpenAIProvider(api_key=api_key)
        
    elif provider_name == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key or api_key == "your-gemini-api-key-here":
            raise ValueError(
                "LLM_PROVIDER is set to 'gemini' but GEMINI_API_KEY is missing or not set in your .env file."
            )
        return GeminiProvider(api_key=api_key)
    
    elif provider_name == "mock":
        return MockProvider()
        
    else:
        raise ValueError(f"Unsupported LLM provider: '{provider_name}'. Please check your .env file.")

