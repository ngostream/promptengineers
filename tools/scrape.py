import re, requests
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field
from typing import List, Dict

class ScrapeUrlsTool(BaseModel):
    """Fetch and clean article text from a list of URLs."""
    urls: List[str] = Field(..., description="List of URLs to fetch")
    timeout: int = Field(12, ge=3, le=60)

    def _extract(self, html: str) -> str:
        soup = BeautifulSoup(html, "lxml")
        for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
            tag.decompose()
        text = re.sub(r"\s+", " ", soup.get_text(" ").strip())
        return text

    def execute(self) -> Dict:
        docs = []
        for u in self.urls:
            try:
                r = requests.get(u, timeout=self.timeout, headers={"User-Agent": "insight-scout/1.0"})
                if r.ok and r.text:
                    docs.append({"url": u, "text": self._extract(r.text)[:20000]})
            except Exception:
                continue
        return {"docs": docs}
