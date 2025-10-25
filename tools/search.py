from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from ddgs import DDGS

class WebSearchTool(BaseModel):
    """Search web for a topic and return result snippets and urls."""
    reasoning: str = Field(..., description="Reasoning process before calling tool")
    query: str = Field(..., description="Search query")
    limit: int = Field(10, ge=1, le=25, description="Max results")
    sites: Optional[List[str]] = Field(None, description="Optional list of site domains to restrict search to")

    def execute(self) -> Dict:
        search_query = self.query
        if self.sites:
            # user-specified site filters
            site_filter = " OR ".join([f"site:{s}" for s in self.sites])
            search_query = f"{search_query} {site_filter}"

        results = list(DDGS().text(search_query, max_results=self.limit))
        out = [{
            "title": r.get("title"),
            "href": r.get("href"),
            "body": r.get("body"),
        } for r in results]
        return {"results": out}

if __name__ == "__main__":
    # user specifies sites
    tool1 = WebSearchTool(
        reasoning="Find AI articles from reliable tech sources",
        query="carbon aware AI",
        sites=["theverge.com", "reuters.com"],
        limit=5
    )
    result1 = tool1.execute()
    print("User-specified sites results:")
    for r in result1["results"]:
        print(r["title"], "-", r["href"])

    # agent chooses sites (sites=None)
    tool2 = WebSearchTool(
        reasoning="Find recent AI news from major sources",
        query="AI regulation",
        limit=5
    )
    result2 = tool2.execute()
    print("\nAgent-chosen sites results:")
    for r in result2["results"]:
        print(r["title"], "-", r["href"])

    # agent chooses sites (sites=None)
    tool3 = WebSearchTool(
        reasoning="Find news about job market",
        query="Job market",
        sites=["reddit.com"],
        limit=5
    )
    result3 = tool3.execute()
    print("\nReddit test:")
    for r in result3["results"]:
        print(r["title"], "-", r["href"])
        