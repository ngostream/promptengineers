import os
import requests
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")


class WebSearchTool(BaseModel):
    """Search web for a topic and returns the top 5 result snippets and urls."""
    reasoning: str = Field(..., description="Reasoning process before calling tool")
    query: str = Field(..., description="Search query")
    sites: Optional[List[str]] = Field(None, description="Optional list of site domains to restrict search to. Note that we support custom parsing for amazon.com reviews and reddit.com comments, so those sites may yield better data points.")

    def execute(self) -> Dict:
        if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
            raise ValueError("Set GOOGLE_API_KEY and GOOGLE_CSE_ID as environment variables.")

        limit = 5
        search_query = self.query
        if self.sites:
            site_filter = " OR ".join([f"site:{s}" for s in self.sites])
            search_query = f"{search_query} {site_filter}"

        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": GOOGLE_API_KEY,
            "cx": GOOGLE_CSE_ID,
            "q": search_query,
            "num": limit,
        }

        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        out = []
        for item in data.get("items", []):
            out.append({
                "title": item.get("title"),
                "href": item.get("link"),
                "body": item.get("snippet"),
            })

        return {"results": out}

# Example usage:
if __name__ == "__main__":
    tool = WebSearchTool(
        reasoning="Find AI articles from reliable tech sources",
        query="carbon aware AI",
        sites=["theverge.com", "reuters.com"]
    )
    result = tool.execute()
    for r in result["results"]:
        print(r["title"], "-", r["href"])
