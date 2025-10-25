import re, requests
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field
from typing import List, Dict
import streamlit as st

class ScrapeUrlsTool(BaseModel):
    """Fetch and clean article text from a list of URLs."""
    reasoning: str = Field(..., description="Reasoning process before calling tool")
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
        valid_urls = []
        n_parsed_posts = 0
        for u in self.urls:
            try:
                r = requests.get(u, timeout=self.timeout, headers={"User-Agent": "insight-scout/1.0"})
                if r.ok and r.text:
                    parsed_posts = self._extract(r.text)#[:20000]
                    docs.append({"url": u, "posts": parsed_posts})
                    valid_urls.append(u)
                    n_parsed_posts+=len(parsed_posts)
            except Exception:
                continue
        st.session_state.scraped_data += docs
        return {'valid_scraped_urls': valid_urls, 'number_of_parsed_posts': n_parsed_posts}
