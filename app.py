import streamlit as st
import asyncio
import numpy as np

from agent.loop import run_insight_scout
from tools.embed_cluster import ClusterFromVectorsTool
from ui_cluster_viz import visualize_clusters  # make sure ui_cluster_viz.py is in repo root


# --- Session state initialization ---
if 'scraped_data' not in st.session_state:
    st.session_state.scraped_data = {'texts': [], 'urls': []}
if 'scraped_embeddings' not in st.session_state:
    st.session_state.scraped_embeddings = {'texts': [], 'urls': [], 'vectors': []}
if 'cluster_data' not in st.session_state:
    st.session_state.cluster_data = {'labels': [], 'groups': {}, 'summaries': {}, 'relevancies': {}, 'sentiments': {}, 'urls': {}}
if 'logs' not in st.session_state:
    st.session_state.logs = []

st.set_page_config(page_title="ScrapAgent", layout="wide")
st.title("ScrapAgent")

# --- UI input ---
q = st.text_input("Topic", value="electric bikes this week")
run = st.button("Run Agent")

# --- Tabs: Results + Live Logs ---
results_tab, logs_tab = st.tabs(["Results", "Logs"])

with logs_tab:
    log_box = st.empty()

def add_log(msg: str):
    """Append a log message and re-render logs tab."""
    st.session_state.logs.append(msg)
    log_box.markdown(
        "\n".join(f"- {line}" for line in st.session_state.logs[-200:]) or "_no logs yet_"
    )

async def run_agent(topic):
    """Run the agent asynchronously with real-time logging."""
    st.session_state.logs.clear()
    add_log(f"Starting research on '{topic}'...")

    result = await run_insight_scout(topic, log_fn=add_log)

    add_log("Agent run complete.")
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

                with st.expander(f"{label} (Score {score:.1f}) - Theme"):
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

        if vectors and texts and len(vectors) == len(texts):
            # clusters = ClusterFromVectorsTool(
            #     reasoning="Visualize clusters for the current run",
            #     vectors=vectors,
            #     min_cluster_size=5
            # ).execute()
            clusters = st.session_state.cluster_data

            visualize_clusters(
                embeddings=np.array(vectors, dtype=np.float32),
                cluster_meta=clusters,
                title="Embedding Clusters",
            )
        else:
            st.info("No embeddings available to visualize yet (or lengths mismatch).")