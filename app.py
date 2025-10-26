import time
import streamlit as st
import asyncio
import numpy as np
import pandas as pd

from agent.loop import run_insight_scout
from tools.embed_cluster import ClusterFromVectorsTool
from ui_cluster_viz import visualize_clusters, visualize_family_scores  # make sure ui_cluster_viz.py is in repo root


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
            The Crappy Scraper For Your Scrappy Insights
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

def add_log_table(event_type, description):
    current_time = time.time()
    elapsed = current_time - st.session_state.start_time
    st.session_state.logs.append({
        "Time (s)": f"{elapsed:.2f}",
        "Event": event_type,
        "Description": description
    })
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
        with st.spinner("Thinking… scraping… clustering… summarizing…"):
            result = asyncio.run(run_agent(q))

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
                sources = st.session_state.cluster_data['urls'][t.get("cluster_id")]
                # sources = list({s.strip() for s in t.get("sources", []) if s.strip()})

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