import json, asyncio
from typing import List, Dict, Any
from datetime import datetime

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

async def act_until_no_tools(messages, resp, log) -> Dict[str, Any]:
    # Execute any tool calls from model and append results; allow search + scrape
    while True:
        msg = resp.choices[0].message
        log(msg)
        if not msg.tool_calls:
            log(messages)
            break
        for call in msg.tool_calls:
            result = execute_tool_call(call, AVAILABLE)
            log(json.dumps(result))
            messages.append({"role": "assistant", "tool_calls": [call]})
            messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)})
        resp = await create_openai_completion(
            messages,
            model=GPT5Deployment.GPT_5_MINI,
            tools=[WebSearchTool, ScrapeUrlsTool],
            tool_choice="auto",
        )
    return {"messages": messages, "resp": resp}

async def embed_and_cluster(min_cluster_size=2, log = print):
    """
    Embeds item texts and clusters the vectors.
    Returns dict with texts, urls, and cluster results.
    """
    items = st.session_state.scraped_data # list of dicts with 'url' and 'texts' keys
    texts = items.get('texts', [])
    urls = items.get('urls', [])
    
    # get embeddings using HF Inference Providers API
    log('Creating embeddings...')
    vectors = await create_embeddings(
        inputs=texts
    )
    log(f'Created {len(vectors)} embeddings.')
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
    log("Clustering results...")
    cluster_results = cluster_tool.execute()
    
    # st.session_state.cluster_data['labels'] = cluster_results['labels']
    # st.session_state.cluster_data['groups'] = cluster_results['groups']

    for label, indexes in cluster_results['groups'].items():
        source_urls = set()
        for idx in indexes:
            if urls[idx] not in source_urls:
                source_urls.add(urls[idx])
        st.session_state.cluster_data['urls'][label] = list(source_urls)
    
    # cluster_results contains: {"labels": [...], "groups": {cluster_id: [indices]}}
    # return with 'clusters' key for backward compatibility
    return {
        "texts": texts, 
        "urls": urls, 
        "clusters": cluster_results  # pass the entire cluster_results dict
    }


async def summarize_clusters(texts: List[str], urls: List[str], clusters: Dict[str, Any], original_prompt: str, relevancy_threshold: float = 0.5, log = print):
    """
    Summarize each cluster with sentiment analysis and scoring.
    clusters should be the dict with "groups" and "labels" keys.
    """
    out = []
    # extract the groups dict from clusters
    groups = clusters.get("groups", {})
    
    batchesOfClusterTexts = []
    sem = asyncio.Semaphore(7)
    
    async def summarizeSingleCluster(cluster_texts:List[str], cid: int, idxs: List[int]):
        async with sem:
        # debug output
            print(f"[DEBUG] Cluster {cid}: {len(idxs)} items")
            log(f"Cluster {cid}: {len(idxs)} items")
            if cluster_texts:
                print(f"[DEBUG] First text sample: {cluster_texts[0][:200]}")
                log(f"First text sample: {cluster_texts[0][:200]}")

            summary_result = await Cluster_Summarize_and_Score(
                texts= cluster_texts,
                original_prompt=original_prompt
            ).execute()
            relevancy = float(summary_result.get("relevancy", 0))
            sentiment = float(summary_result.get("sentiment", 0))
            summary = summary_result.get("summary", "")
            # {'labels': [], 'groups': {}, 'summaries': {}, 'relevancies': {}, 'sentiments': {}, 'urls': {}}
            if relevancy<relevancy_threshold:
                return
            st.session_state.cluster_data['labels'].append(cid)
            st.session_state.cluster_data['groups'][cid] = idxs
            st.session_state.cluster_data['summaries'][cid] = summary
            st.session_state.cluster_data['relevancies'][cid] = relevancy
            st.session_state.cluster_data['sentiments'][cid] = sentiment

            log(f"Cluster {cid} summary: {summary[:200]}... Relevancy: {relevancy}, Sentiment: {sentiment}")
            # scores = sent_result.get("scores", [])
            # s_avg = (sum(scores)/max(1, len(scores))) if scores else 0
            
            # summary = await summarize_cluster(cluster_texts, cid)
            
            # using sentiment as score
            # score = s_avg

            # srcs = [urls[i] for i in idxs if i < len(urls) and urls[i]]
            out.append({
                "cluster_id": cid, 
                "summary": summary, 
                "score": sentiment, 
                "relevancy": relevancy,
                # "sources": list(dict.fromkeys(srcs))[:5]
            })

    for cid, idxs in groups.items(): #{int: list[int]}
        cluster_texts = [texts[i] for i in idxs] #list[list[str]] per cluster
        batchesOfClusterTexts.append((cluster_texts, (cid,idxs)))
    
    tasks = [summarizeSingleCluster(i[0], i[1][0], i[1][1]) for i in batchesOfClusterTexts]
    await asyncio.gather(*tasks)
    
    # sort themes by simple score desc
    out.sort(key=lambda x: x["relevancy"], reverse=True)
    return out

# async def summarize_cluster(texts: List[str], cid: int) -> str:
#     # Summarize a cluster with GPT-5-MINI

#     joined = "\n".join(texts) #summarize all text from the entire cluster
#     contentString = ""
#     if cid == -1:
#         contentString = "This is a list of responses that don't categorize into any clusters. \
#                         You summarize clusters into: Title + 4 bullets + 1-sentence why-it-matters. When summarizing, keep in mind that\
#                         these do not belong to any clusters, and may be anomalies."
#     else:
#         contentString = "You summarize clusters into: Title + 4 bullets + 1-sentence why-it-matters."
#     messages = [
#         {"role": "system", "content": contentString},
#         {"role": "user", "content": f"Summarize these items:\n{joined}"}
#     ]
#     resp = await create_openai_completion(messages, model=GPT5Deployment.GPT_5_MINI)
#     return resp.choices[0].message.content or ""

async def write_report(topic: str, themes: List[Dict[str, Any]]) -> str:
    bullets = []
    for t in themes:
        src_lines = "\n  ".join([f"- {u}" for u in st.session_state.cluster_data['urls'][t['cluster_id']]]) or "- (no sources)"
        bullets.append(f"### Theme (Score {t['score']})\n{t['summary']}\n\n**Sources:**\n {src_lines}\n")
    content = "\n".join(bullets)
    messages = [
        {"role": "system", "content": SYSTEM_REPORTER},
        {"role": "user", "content": f"Topic: {topic}\n\nThemes:\n{content}"}
    ]
    resp = await create_openai_completion(messages, model=GPT5Deployment.GPT_5)
    return resp.choices[0].message.content or ""

async def run_insight_scout(topic: str, log_fn = None) -> Dict[str, Any]:
    def log(msg):
        if log_fn:
            log_fn(msg)
        else:
            print(msg)

    print(f"[DEBUG] Starting Insight Agent for topic: {topic}")
    
    # 1) Plan → initial search
    ctx = await plan(topic)
    messages, resp = ctx["messages"], ctx["resp"]
    print(f"[DEBUG] Initial plan completed. Tool calls found: {bool(resp.choices[0].message.tool_calls)}")
    
    # Manual search if model didn't call tools
    if not resp.choices[0].message.tool_calls:
        print("[DEBUG] No tool calls from model. Running manual search...")
        log('No tool calls from model. Running manual search...')
        search_tool = WebSearchTool(query=topic, reasoning="Manual search", limit=10)
        search_results = search_tool.execute()
        print(f"[DEBUG] Manual search results: {len(search_results.get('results', []))} items")
        log(f"Manual search results: {len(search_results.get('results', []))} items")
        messages.append({"role": "tool", "tool_call_id": "manual_search", "content": json.dumps(search_results)})

        hrefs = [r["href"] for r in search_results.get("results", []) if r.get("href")]
        if hrefs:
            print(f"[DEBUG] Running manual scrape on {len(hrefs[:8])} URLs")
            log(f"Running manual scrape on {len(hrefs[:8])} URLs")
            scrape_tool = ScrapeUrlsTool(urls=hrefs[:8])
            scrape_results = scrape_tool.execute()
            print(f"[DEBUG] Manual scrape results: {len(scrape_results.get('docs', []))} docs")
            log(f"Manual scrape results: {len(scrape_results.get('docs', []))} docs")
            messages.append({"role": "tool", "tool_call_id": "manual_scrape", "content": json.dumps(scrape_results)})

    # 2) Let the model call tools (search first; it may then ask to scrape)
    step = await act_until_no_tools(messages, resp, log)
    messages = step["messages"]
    print(f"[DEBUG] Tool execution loop complete. Total messages: {len(messages)}")

    # 4) Embed + cluster in parallel
    ec = await embed_and_cluster(min_cluster_size=2, log = log)
    print(f"[DEBUG] Embedding and clustering complete. Number of clusters: {len(ec['clusters']['labels'])}")
    log(f"Embedding and clustering complete. Number of clusters: {len(ec['clusters']['labels'])}")

    # 5) Summarize clusters + score
    print("Summarizing Clusters at ", datetime.now())
    themes = await summarize_clusters(ec["texts"], ec["urls"], ec["clusters"], original_prompt = topic, log = log)
    print("Summariziation done at ", datetime.now())
    print(f"[DEBUG] Summarization complete. Number of themes: {len(themes)}")
    log(f"Summarization complete. Number of themes: {len(themes)}")

    # 6) Final report
    print("Generating final report...")
    max_retries = 3
    attempt = 0
    report = ""

    while attempt < max_retries:
        report = await write_report(topic, themes)
        if report and len(report.strip()) > 0:
            break  # success
        attempt += 1
        print(f"[DEBUG] Empty report on attempt {attempt}. Retrying...")
        log(f"Empty report on attempt {attempt}. Retrying...")

    if not report or len(report.strip()) == 0:
        report = "Failed to generate report after multiple attempts. Try rerunning the agent."

    print(f"[DEBUG] Report generated. Length: {len(report)} characters after {attempt+1} attempt(s)")
    log(f"Report generated. Length: {len(report)} characters after {attempt+1} attempt(s)")

    return {"topic": topic, "themes": themes, "report": report}
