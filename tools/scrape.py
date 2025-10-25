import re, requests
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field
from typing import List, Dict
import streamlit as st
from urllib.parse import urlparse

def check_url_platform(url: str) -> str:
    """
    Returns 'amazon', 'reddit', or 'other' based on the URL.
    """
    try:
        hostname = urlparse(url).hostname or ""
        hostname = hostname.lower()
        if "amazon." in hostname:
            return "amazon"
        elif "reddit." in hostname:
            return "reddit"
        else:
            return "other"
    except Exception:
        return "other"
    
#create custom scrapers for each valid site
def get_reddit_comments(url: str):
    """Return all comments and replies from a Reddit post JSON."""
    if not url.endswith("/.json"):
        url = url.rstrip("/") + "/.json"

    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers)
    r.raise_for_status()
    data = r.json()

    all_comments = []

    def traverse_comments(children):
        for child in children:
            if child["kind"] != "t1":  # t1 = comment
                continue
            body = child["data"].get("body")
            if body:
                all_comments.append(body)
            # Recursively get replies
            replies = child["data"].get("replies")
            if replies and isinstance(replies, dict):
                traverse_comments(replies["data"]["children"])

    # Reddit returns [post info, comments] structure
    top_level_comments = data[1]["data"]["children"]
    traverse_comments(top_level_comments)

    return all_comments

def scrape_amazon_reviews(page_url: str, delay: float = 1.0):
    """
    Scrape paragraphs inside Amazon review containers.
    Example URL: first page of reviews for a product.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }

    reviews = []
    # print(f"Fetching page: {page_url}")

    r = requests.get(page_url, headers=headers)
    if r.status_code != 200:
        # print(f"Failed to load page {page_url} (status {r.status_code})")
        return []

    soup = BeautifulSoup(r.text, "lxml")

    # Look for all elements with class containing 'review-text'
    review_divs = soup.find_all(class_=lambda c: c and "review-text" in c)
    if not review_divs:
        # print("No reviews found on this page — stopping.")
        return []

    for div in review_divs:
        # Get all <p> tags inside the review container
        for p in div.find_all("span"):
            text = p.get_text(strip=True)
            if text and text not in reviews:
                reviews.append(text)

    return reviews

class ScrapeUrlsTool(BaseModel):
    """Fetch and clean article text from a list of URLs."""
    reasoning: str = Field(..., description="Reasoning process before calling tool")
    urls: List[str] = Field(..., description="List of URLs to fetch")
    timeout: int = Field(12, ge=3, le=60)

    def _extract(self, url) -> str:
        if check_url_platform(url) == "reddit":
            comments = get_reddit_comments(url)
            if comments:
                return comments
        elif check_url_platform(url) == "amazon":
            reviews = scrape_amazon_reviews(url)
            if reviews:
                return reviews
        html = url.text
        soup = BeautifulSoup(html, "lxml")
        for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
            tag.decompose()
        chunks = []
        seen_texts = set()

        # Collect all <p> and <span> text
        for el in soup.find_all(["p", "span"]):
            text = el.get_text(" ", strip=True)
            if text and text not in seen_texts:
                seen_texts.add(text)
                chunks.append(text)
        return chunks

    def execute(self) -> Dict:
        valid_urls = []
        n_parsed_posts = 0
        for u in self.urls:
            try:
                r = requests.get(u, timeout=self.timeout, headers={"User-Agent": "insight-scout/1.0"})
                # if r.ok and r.text:
                parsed_posts = self._extract(r)#[:20000]
                st.session_state.scraped_data['texts']+=parsed_posts
                st.session_state.scraped_data['urls']+=[u]*len(parsed_posts)
                valid_urls.append(u)
                n_parsed_posts+=len(parsed_posts)
            except Exception:
                continue
        return {'valid_scraped_urls': valid_urls, 'number_of_scraped_chunks': n_parsed_posts}
