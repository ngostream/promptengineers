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
if 'logs' not in st.session_state:
    st.session_state.logs = []

st.set_page_config(page_title="Insight Agent", layout="wide")
st.title("Insight Agent")

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
        for t in themes:
            with st.expander(f"Score {t['score']} - Theme"):
                st.markdown(t.get("summary", ""))
                st.markdown("**Sources:**\n" + "\n".join([f"- {u}" for u in t.get("sources", [])]))

        # --- Cluster visualization ---
        st.subheader("Embedding Clusters")
        vectors = st.session_state.scraped_embeddings.get("vectors", [])
        texts = st.session_state.scraped_data.get("texts", [])
        urls = st.session_state.scraped_data.get("urls", [])

        if vectors and texts and len(vectors) == len(texts):
            clusters = ClusterFromVectorsTool(
                reasoning="Visualize clusters for the current run",
                vectors=vectors,
                min_cluster_size=5
            ).execute()

            visualize_clusters(
                embeddings=np.array(vectors, dtype=np.float32),
                labels=clusters["labels"],
                texts=texts,
                urls=urls,
                title="Embedding clusters"
            )
        else:
            st.info("No embeddings available to visualize yet (or lengths mismatch).")
