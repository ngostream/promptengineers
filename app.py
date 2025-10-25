import streamlit as st
import asyncio
import numpy as np

from agent.loop import run_insight_scout
from tools.embed_cluster import ClusterFromVectorsTool
from ui_cluster_viz import visualize_clusters  # make sure ui_cluster_viz.py is in repo root

if 'scraped_data' not in st.session_state:
    st.session_state.scraped_data = {'texts': [], 'urls': []}
if 'scraped_embeddings' not in st.session_state:
    st.session_state.scraped_embeddings = {'texts': [], 'urls': [], 'vectors': []}

st.set_page_config(page_title="Insight Agent", layout="wide")
st.title("Insight Agent - Autonomous Research Agent")

q = st.text_input("Topic", value="electric bikes this week")
run = st.button("Run Agent")

if run:
    with st.spinner("Thinking… scraping… clustering… summarizing…"):
        result = asyncio.run(run_insight_scout(q))

    st.subheader("Research Brief")
    st.markdown(result.get("report", "(no report)"))

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
