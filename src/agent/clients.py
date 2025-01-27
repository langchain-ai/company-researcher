import os
from tavily import AsyncTavilyClient

API_KEY = os.getenv("TAVILY_API_KEY")
if not API_KEY:
    raise EnvironmentError(
        "TAVILY_API_KEY not found. Please set it in your environment variables."
    )

tavily_async_client = AsyncTavilyClient(api_key=API_KEY)
