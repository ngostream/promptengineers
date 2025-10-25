import json, asyncio
from typing import List, Dict, Any

from unwrap_sdk import (
    create_openai_completion, create_embeddings, execute_tool_call,
    GPT5Deployment
)
import streamlit as st
from tools.search import WebSearchTool
from tools.scrape import ScrapeUrlsTool
from tools.embed_cluster import ClusterFromVectorsTool
from tools.sentiment import Cluster_Summarize_and_Score
from agent.prompts import SYSTEM_PLANNER, SYSTEM_REPORTER
from unwrap_sdk import HF_MODEL, HF_API_KEY
import numpy as np

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

async def embed_and_cluster(min_cluster_size=2):
    """
    Embeds item texts and clusters the vectors.
    Returns dict with texts, urls, and cluster results.
    """
    items = st.session_state.scraped_data # list of dicts with 'url' and 'texts' keys
    texts = items.get('texts', [])
    urls = items.get('urls', [])
    
    # get embeddings using HF Inference Providers API
    vectors = await create_embeddings(
        inputs=texts,
        model=HF_MODEL,
        api_key=HF_API_KEY
    )
    st.session_state.scraped_embeddings['texts'] += texts
    st.session_state.scraped_embeddings['urls'] += urls
    st.session_state.scraped_embeddings['vectors'] += vectors
    
    # cluster vectors with required reasoning field
    cluster_tool = ClusterFromVectorsTool(
        reasoning="Clustering article embeddings to identify common themes and group similar content",
        vectors=vectors, 
        min_cluster_size=min_cluster_size
    )
    print("Clustering results...")
    cluster_results = cluster_tool.execute()
    
    # cluster_results contains: {"labels": [...], "groups": {cluster_id: [indices]}}
    # return with 'clusters' key for backward compatibility
    print(f"[DEBUG] Embedding output shape: {np.array(vectors).shape}")
    return {
        "texts": texts, 
        "urls": urls, 
        "clusters": cluster_results  # pass the entire cluster_results dict
    }


async def summarize_clusters(texts: List[str], urls: List[str], clusters: Dict[str, Any], relevancy_threshold: float = 0.5):
    """
    Summarize each cluster with sentiment analysis and scoring.
    clusters should be the dict with "groups" and "labels" keys.
    """
    out = []
    # extract the groups dict from clusters
    groups = clusters.get("groups", {})
    
    for cid, idxs in groups.items():
        cluster_texts = [texts[i] for i in idxs]
        
        # debug output
        print(f"[DEBUG] Cluster {cid}: {len(idxs)} items")
        if cluster_texts:
            print(f"[DEBUG] First text sample: {cluster_texts[0][:200]}")
        
        # sent_result = await SimpleLexSentimentTool(
        #     reasoning="Analyzing sentiment of cluster texts to gauge overall tone",
        #     texts= cluster_texts     #[t[:500] for t in cluster_texts]  previous version
        # ).execute()

        summary_result = await Cluster_Summarize_and_Score(
            texts= cluster_texts,
            original_prompt="General web search on the given topic"
        ).execute()
        relevancy = float(summary_result.get("relevancy", 0))
        sentiment = float(summary_result.get("sentiment", 0))
        summary = summary_result.get("summary", "")
        # scores = sent_result.get("scores", [])
        # s_avg = (sum(scores)/max(1, len(scores))) if scores else 0
        
        # summary = await summarize_cluster(cluster_texts, cid)
        
        # using sentiment as score
        # score = s_avg
        srcs = [urls[i] for i in idxs if i < len(urls) and urls[i]]
        out.append({
            "cluster_id": cid, 
            "summary": summary, 
            "score": sentiment, 
            "relevancy": relevancy,
            "sources": list(dict.fromkeys(srcs))[:5]
        })
    
    # sort themes by simple score desc
    out.sort(key=lambda x: x["relevancy"], reverse=True)
    return [o for o in out if o["relevancy"]>relevancy_threshold]

async def summarize_cluster(texts: List[str], cid: int) -> str:
    # Summarize a cluster with GPT-5-MINI

    joined = "\n".join(texts) #summarize all text from the entire cluster
    contentString = ""
    if cid == -1:
        contentString = "This is a list of responses that don't categorize into any clusters. \
                        You summarize clusters into: Title + 4 bullets + 1-sentence why-it-matters. When summarizing, keep in mind that\
                        these do not belong to any clusters, and may be anomalies."
    else:
        contentString = "You summarize clusters into: Title + 4 bullets + 1-sentence why-it-matters."
    messages = [
        {"role": "system", "content": contentString},
        {"role": "user", "content": f"Summarize these items:\n{joined}"}
    ]
    resp = await create_openai_completion(messages, model=GPT5Deployment.GPT_5_MINI)
    return resp.choices[0].message.content or ""

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
        search_tool = WebSearchTool(query=topic, reasoning="Manual search", limit=10)
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

    # 4) Embed + cluster in parallel (max 5 parallel calls allowed)
    print(f"[DEBUG] Number of texts before embedding: {len(st.session_state.scraped_data.get('texts', []))}")
    ec = await embed_and_cluster(min_cluster_size=2)
    print(f"[DEBUG] Embedding and clustering complete. Number of clusters: {len(ec['clusters'])}")

    # 5) Summarize clusters + score
    themes = await summarize_clusters(ec["texts"], ec["urls"], ec["clusters"])
    print(f"[DEBUG] Summarization complete. Number of themes: {len(themes)}")

    # 6) Final report
    report = await write_report(topic, themes)
    print(f"[DEBUG] Report generated. Length: {len(report)} characters")

    return {"topic": topic, "themes": themes, "report": report}
