import time
import streamlit as st
import asyncio
import numpy as np
import pandas as pd
import json, os

from agent.loop import run_insight_scout
from tools.embed_cluster import ClusterFromVectorsTool
from ui_cluster_viz import visualize_clusters, visualize_family_scores  # make sure ui_cluster_viz.py is in repo root

# --- DEMO MODE TOGGLE ---
DEMO_MODE = True  # flip to False in main branch

def save_demo_results(topic, result):
    os.makedirs("demo_results", exist_ok=True)
    fname = f"demo_results/{topic.replace(' ', '_')}.json"

    with open(fname, "w") as f:
        json.dump({
            "result": result,
            "scraped_data": st.session_state.get("scraped_data", {}),
            "scraped_embeddings": st.session_state.get("scraped_embeddings", {}),
            "cluster_data": st.session_state.get("cluster_data", {}),
            "logs": st.session_state.get("logs", []),
        }, f)

    st.success(f"Saved demo results to {fname}")

def load_demo_results(topic):
    import json, os

    fname = f"demo_results/{topic.replace(' ', '_')}.json"

    if not os.path.exists(fname):
        return None

    with open(fname, "r") as f:
        data = json.load(f)

    # Restore session state
    st.session_state.scraped_data = data.get("scraped_data", {})
    st.session_state.scraped_embeddings = data.get("scraped_embeddings", {})
    st.session_state.cluster_data = data.get("cluster_data", {})

    # Restore logs or add a fallback if missing
    st.session_state.logs = data.get("logs") or [
        {"Time (s)": "0.00", "Event": "System", "Description": f"Loaded cached results for '{topic}'"},
        {"Time (s)": "—", "Event": "Cache", "Description": "Demo mode: no live agent activity"},
    ]

    return data.get("result", {})

# --- Session state initialization ---
if 'scraped_data' not in st.session_state:
    st.session_state.scraped_data = {'texts': [], 'urls': []}
if 'scraped_embeddings' not in st.session_state:
    st.session_state.scraped_embeddings = {'texts': [], 'urls': [], 'vectors': []}
if 'cluster_data' not in st.session_state:
    st.session_state.cluster_data = {'labels': [], 'groups': {}, 'summaries': {}, 'relevancies': {}, 'sentiments': {}, 'urls': {}, 'topics': {}}
if 'logs' not in st.session_state:
    st.session_state.logs = []

st.set_page_config(page_title="ScrapAgent", layout="wide")
col1, col2 = st.columns([2, 9])

with col1:
    st.title("ScrapAgent")

with col2:
    st.markdown(
        """
        <div style='
            position: relative;
            top: 25px;
            font-size: 32px;
            color: gray;
        '>
            Crappy Scraper For Your Scrappy Insights
        </div>
        """,
        unsafe_allow_html=True
    )

# --- UI input ---
q = st.text_input("Topic", value="electric bikes this week")
run = st.button("Run")

# --- Tabs: Results + Live Logs ---
results_tab, logs_tab = st.tabs(["Results", "Logs"])

with logs_tab:
    log_box = st.empty()

# def add_log(msg: str):
#     """Append a log message and re-render logs tab."""
#     st.session_state.logs.append(msg)
#     log_box.markdown(
#         "\n".join(f"- {line}" for line in st.session_state.logs[-200:]) or "_no logs yet_"
#     )

def add_log_table(event_type, description):
    current_time = time.time()
    elapsed = current_time - st.session_state.start_time
    st.session_state.logs.append({
        "Time (s)": f"{elapsed:.2f}",
        "Event": event_type,
        "Description": description
    })
    # st.session_state.logs.append({"Event": event_type, "Description": description})
    df = pd.DataFrame(st.session_state.logs[-200:])  # show only latest 200
    log_box.dataframe(df, width='stretch', hide_index=True)

async def run_agent(topic):
    """Run the agent asynchronously with real-time logging."""
    st.session_state.scraped_data = {'texts': [], 'urls': []}
    st.session_state.scraped_embeddings = {'texts': [], 'urls': [], 'vectors': []}
    st.session_state.cluster_data = {'labels': [], 'groups': {}, 'summaries': {}, 'relevancies': {}, 'sentiments': {}, 'urls': {}, 'topics': {}}
    st.session_state.logs = []
    st.session_state.start_time = time.time()
    # add_log(f"Starting research on '{topic}'...")
    add_log_table("System", f"Starting research on '{topic}'...")

    result = await run_insight_scout(topic, log_fn=add_log_table)

    add_log_table("System", "Agent run complete.")
    return result


# --- Run logic ---
if run:
    with results_tab:
        if DEMO_MODE:
            result = load_demo_results(q)
            if result is None:
                with st.spinner("Running live agent (no demo data found)..."):
                    result = asyncio.run(run_agent(q))
                    save_demo_results(q, result)
            # --- Force log table refresh in demo mode ---
            if st.session_state.logs:
                df = pd.DataFrame(st.session_state.logs)
                with logs_tab:
                    st.dataframe(df, width='stretch', hide_index=True)
            else:
                with logs_tab:
                    st.info("No logs available for this demo.")
        else:
            with st.spinner("Thinking… scraping… clustering… summarizing…"):
                result = asyncio.run(run_agent(q))
                # optionally save after every live run
                save_demo_results(q, result)

        # --- Report ---
        st.subheader("Research Brief")
        st.markdown(result.get("report", "(no report)"))

        # --- Themes ---
        st.subheader("Themes")
        themes = result.get("themes", [])
        if not themes:
            st.info("No themes found.")
        else:
            def sentiment_label(score):
                if score is None:
                    return "Unknown"
                if score == 0:
                    return "Neutral"
                elif score < -0.75:
                    return "Strongly Negative"
                elif score < -0.25:
                    return "Moderately Negative"
                elif score < 0:
                    return "Slightly Negative"
                elif score > 0.75:
                    return "Strongly Positive"
                elif score > 0.25:
                    return "Moderately Positive"
                else:
                    return "Slightly Positive"

            for t in themes:
                score = t.get("score", 0)
                label = sentiment_label(score)

                # get and clean sources
                cluster_id = t.get("cluster_id")
                sources = st.session_state.cluster_data.get("urls", {}).get(cluster_id, [])

                with st.expander(f"{label} (Score {score:.1f}) - {t['topic']}"):
                    st.markdown(t.get("summary", "_No summary available._"))

                    if sources:
                        st.markdown("**All Sources Used:**")
                        for src in sources:
                            st.markdown(f"- [{src}]({src})")
                    else:
                        st.markdown("_No sources listed._")

        # --- Cluster visualization ---
        vectors = st.session_state.scraped_embeddings.get("vectors", [])
        texts = st.session_state.scraped_data.get("texts", [])
        urls = st.session_state.scraped_data.get("urls", [])
        cluster_groups = st.session_state.cluster_data.get("groups", {})
        cluster_relevancies = st.session_state.cluster_data.get("relevancies", {})
        cluster_sentiments = st.session_state.cluster_data.get("sentiments", {})
        clusters = st.session_state.cluster_data

        if clusters:
            
            visualize_family_scores(urls,
                cluster_groups,
                cluster_relevancies,
                "Average Cluster Relevancy by Domain")
            visualize_family_scores(urls,
                cluster_groups,
                cluster_sentiments,
                "Average Cluster Sentiment by Domain")
            visualize_clusters(
                embeddings=np.array(vectors, dtype=np.float32),
                cluster_meta=clusters,
                title="Embedding Clusters",
            )
        else:
            st.info("No embeddings available to visualize yet (or lengths mismatch).")