import json, asyncio
from typing import List, Dict, Any

from unwrap_sdk import (
    create_openai_completion, create_embeddings, execute_tool_call,
    GPT5Deployment
)
from tools.search import WebSearchTool
from tools.scrape import ScrapeUrlsTool
from tools.embed_cluster import ClusterFromVectorsTool
from tools.sentiment import SimpleLexSentimentTool
from agent.prompts import SYSTEM_PLANNER, SYSTEM_REPORTER

AVAILABLE = {
    "WebSearchTool": WebSearchTool,
    "ScrapeUrlsTool": ScrapeUrlsTool,
}

async def plan(topic: str) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": SYSTEM_PLANNER},
        {"role": "user", "content": f"Topic: {topic}. Start the plan with a web search (10 results)."},
    ]
    resp = await create_openai_completion(
        messages, model=GPT5Deployment.GPT_5_MINI, tools=[WebSearchTool], tool_choice="auto"
    )
    return {"messages": messages, "resp": resp}

async def act_until_no_tools(messages, resp) -> Dict[str, Any]:
    # Execute any tool calls from model and append results; allow search + scrape
    while True:
        msg = resp.choices[0].message
        if not msg.tool_calls:
            break
        for call in msg.tool_calls:
            result = execute_tool_call(call, AVAILABLE)
            messages.append({"role": "assistant", "tool_calls": [call]})
            messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)})
        resp = await create_openai_completion(
            messages,
            model=GPT5Deployment.GPT_5_MINI,
            tools=[WebSearchTool, ScrapeUrlsTool],
            tool_choice="auto",
        )
    return {"messages": messages, "resp": resp}

async def collect_items(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for m in messages:
        if m.get("role") == "tool":
            try:
                payload = json.loads(m.get("content", "{}"))
            except Exception:
                continue
            data = payload.get("data", {})  # <-- fix: look inside 'data'
            
            # WebSearchTool format
            for r in data.get("results", []):
                items.append({"title": r.get("title"), "body": r.get("body"), "url": r.get("href")})
            
            # ScrapeUrlsTool format
            for d in data.get("docs", []):
                items.append({"title": None, "body": d.get("text"), "url": d.get("url")})
    print(f"[DEBUG] collect_items found {len(items)} total items")
    return items

async def embed_and_cluster(items: List[Dict[str, Any]], min_cluster_size=2) -> Dict[str, Any]:
    # Prefer full text when available, else snippet/title
    texts = [ (i.get("body") or i.get("title") or "").strip() for i in items ]
    urls = [ i.get("url") for i in items ]
    vectors = await create_embeddings(texts)
    clusters = ClusterFromVectorsTool(vectors=vectors, min_cluster_size=min_cluster_size).execute()["groups"]
    return {"texts": texts, "urls": urls, "clusters": clusters}

async def summarize_cluster(texts: List[str], cid: int) -> str:
    # Summarize a cluster with GPT-5-MINI
    joined = "\n".join(t[:500] for t in texts[:12])
    contentString = ""
    if cid == -1:
        contentString = "This is a list of responses that don't categorize into any clusters. \
                        Summarize this list into: Title + 4 bullets + 1-sentence why-it-matters. When summarizing, keep in mind that\
                        these do not belong to any clusters, and may be anomalies."
    else:
        contentString = "You summarize clusters into: Title + 4 bullets + 1-sentence why-it-matters."
    messages = [
        {"role": "system", "content": contentString},
        {"role": "user", "content": f"Summarize these items:\n{joined}"}
    ]
    resp = await create_openai_completion(messages, model=GPT5Deployment.GPT_5_MINI)
    return resp.choices[0].message.content or ""

async def summarize_clusters(texts: List[str], urls: List[str], clusters: Dict[int, List[int]]):
    out = []
    for cid, idxs in clusters.items():
        cluster_texts = [texts[i] for i in idxs]
        sent = SimpleLexSentimentTool(texts=[t[:500] for t in cluster_texts]).execute()["scores"]
        s_avg = (sum(sent)/max(1,len(sent))) if sent else 0
        summary = ""
        summary = await summarize_cluster(cluster_texts, cid)
        # Simple score: size + small boost for positive sentiment
        score = min(100, int(len(idxs) * 6 + max(0, s_avg) * 5))
        srcs = [urls[i] for i in idxs if urls[i]]
        out.append({"cluster_id": cid, "summary": summary, "score": score, "sources": list(dict.fromkeys(srcs))[:5]})
    # Sort themes by score desc
    out.sort(key=lambda x: x["score"], reverse=True)
    return out

async def write_report(topic: str, themes: List[Dict[str, Any]]) -> str:
    bullets = []
    for t in themes:
        src_lines = "\n  ".join([f"- {u}" for u in t['sources']]) or "- (no sources)"
        bullets.append(f"### Theme (Score {t['score']})\n{t['summary']}\n\n**Sources:**\n  {src_lines}\n")
    content = "\n".join(bullets)
    messages = [
        {"role": "system", "content": SYSTEM_REPORTER},
        {"role": "user", "content": f"Topic: {topic}\n\nThemes:\n{content}"}
    ]
    resp = await create_openai_completion(messages, model=GPT5Deployment.GPT_5)
    return resp.choices[0].message.content or ""

async def run_insight_scout(topic: str) -> Dict[str, Any]:
    print(f"[DEBUG] Starting Insight Scout for topic: {topic}")
    
    # 1) Plan → initial search
    ctx = await plan(topic)
    messages, resp = ctx["messages"], ctx["resp"]
    print(f"[DEBUG] Initial plan completed. Tool calls found: {bool(resp.choices[0].message.tool_calls)}")
    
    # Manual search if model didn't call tools
    if not resp.choices[0].message.tool_calls:
        print("[DEBUG] No tool calls from model. Running manual search...")
        search_tool = WebSearchTool(query=topic, max_results=10)
        search_results = search_tool.execute()
        print(f"[DEBUG] Manual search results: {len(search_results.get('results', []))} items")
        messages.append({"role": "tool", "tool_call_id": "manual_search", "content": json.dumps(search_results)})

        hrefs = [r["href"] for r in search_results.get("results", []) if r.get("href")]
        if hrefs:
            print(f"[DEBUG] Running manual scrape on {len(hrefs[:8])} URLs")
            scrape_tool = ScrapeUrlsTool(urls=hrefs[:8])
            scrape_results = scrape_tool.execute()
            print(f"[DEBUG] Manual scrape results: {len(scrape_results.get('docs', []))} docs")
            messages.append({"role": "tool", "tool_call_id": "manual_scrape", "content": json.dumps(scrape_results)})

    # 2) Let the model call tools (search first; it may then ask to scrape)
    step = await act_until_no_tools(messages, resp)
    messages = step["messages"]
    print(f"[DEBUG] Tool execution loop complete. Total messages: {len(messages)}")

    # 3) Collect items (from search + scrape)
    items = await collect_items(messages)
    print(f"[DEBUG] Collected items: {len(items)}")
    if not items:
        print("[DEBUG] No items found after collection!")
        return {"topic": topic, "themes": [], "report": "No items found. Try another topic or broader query."}

    # 4) Embed + cluster
    ec = await embed_and_cluster(items, min_cluster_size=2)
    print(f"[DEBUG] Embedding and clustering complete. Number of clusters: {len(ec['clusters'])}")

    # 5) Summarize clusters + score
    themes = await summarize_clusters(ec["texts"], ec["urls"], ec["clusters"])
    print(f"[DEBUG] Summarization complete. Number of themes: {len(themes)}")

    # 6) Final report
    report = await write_report(topic, themes)
    print(f"[DEBUG] Report generated. Length: {len(report)} characters")

    return {"topic": topic, "themes": themes, "report": report}
