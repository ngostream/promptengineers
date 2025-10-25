from typing import List, Dict
from pydantic import BaseModel, Field
from ddgs import DDGS

class WebSearchTool(BaseModel):
    """Search web for a topic and return result snippets and urls."""
    reasoning: str = Field(..., description="Reasoning process before calling tool")
    query: str = Field(..., description="Search query")
    limit: int = Field(10, ge=1, le=25, description="Max results")

    def execute(self) -> Dict:
        results = list(DDGS().text(self.query, max_results=self.limit))
        out = [{
            "title": r.get("title"),
            "href": r.get("href"),
            "body": r.get("body"),
        } for r in results]
        return {"results": out}
