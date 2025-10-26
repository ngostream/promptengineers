# ui_cluster_viz.py

import numpy as np
import streamlit as st
from typing import List, Dict, Optional, Any
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from site_score_aggregator import compute_family_scores
import pandas as pd
try:
    import plotly.express as px
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False
    import matplotlib.pyplot as plt


@st.cache_data(show_spinner=False)
def _reduce_embeddings(emb: np.ndarray, method: str, seed: int) -> np.ndarray:
    if method == "PCA (fast)":
        pca = PCA(n_components=2, random_state=seed)
        return pca.fit_transform(emb)
    else:
        tsne = TSNE(n_components=2, learning_rate="auto", init="random",
                    perplexity=30, random_state=seed)
        return tsne.fit_transform(emb)


def visualize_clusters(
    embeddings: np.ndarray,
    cluster_meta: Dict[str, Any],
    title: str = "Embedding clusters",
):
    """
    embeddings: (N, D)
    labels: length-N ints (HDBSCAN: -1 = noise)
    texts:  length-N strings for hover/legend derivation
    urls:   length-N strings for legend links (optional)
    cluster_meta: optional dict:
        {
          <label>: {"description": "...", "link": "https://..."},
          ...
        }
    """
    st.subheader(title)
    idxs = []
    for group in st.session_state.cluster_data['groups'].values():
        idxs+=group
    
    labels = np.zeros(len(embeddings), dtype=int)
    for i, ids in cluster_meta.get("groups", {}).items():
        for idx in ids:
            labels[idx] = int(i)
    # Controls
    # colA, colB, colC = st.columns([1, 1, 1])
    method = "PCA (fast)"
    seed = 42
    # with colA:
    #     method = st.selectbox("Reducer", ["PCA (fast)", "t-SNE (slower)"], index=0)
    # with colB:
    #     seed = st.number_input("Random seed", value=42, step=1)
    # with colC:
    #     sample_n = st.slider(
    #         "Max points to plot",
    #         min_value=200,
    #         max_value=len(embeddings),
    #         value=min(2000, len(embeddings)),
    #         step=100,
    #     )

    # Subsample for speed if needed
    
    # if len(idx) > sample_n:
    #     rng = np.random.default_rng(seed)
    #     idx = rng.choice(idx, size=sample_n, replace=False)

    emb = np.asarray(embeddings)[idxs]
    lab = np.asarray(labels)[idxs]

    # Normalize (cosine-friendly) if not already
    norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9
    emb_norm = emb / norms

    # 2D reduction
    coords = _reduce_embeddings(emb_norm, method, seed)
    x, y = coords[:, 0], coords[:, 1]

    # Cluster sizes (over the *plotted* subset)
    cluster_indices = cluster_meta.get("groups", {})
    # cluster_indices: Dict[int, List[int]] = {}
    # for i, c in enumerate(lab):
    #     cluster_indices.setdefault(int(c), []).append(i)
    cluster_sizes = {c: len(ixs) for c, ixs in cluster_indices.items()}

    # Build legend entries (either from cluster_meta or derived)
    legend_rows = []
    for c, ixs in sorted(cluster_indices.items(), key=lambda kv: (kv[0] == -1, kv[0])):  # put -1 last
        size = cluster_sizes[c]
        desc = cluster_meta.get("summaries", {}).get(c, "")
        relevancy = cluster_meta.get("relevancies", {}).get(c, 0.0)
        sentiment = cluster_meta.get("sentiments", {}).get(c, "")
        links = cluster_meta.get("urls", {}).get(c, [])
        topic = cluster_meta.get("topics", {}).get(c, "")
        legend_rows.append(
            {
                "cluster": c,
                "size": size,
                "description": desc if desc else "(no description)",
                "relevancy": relevancy,
                "sentiment": sentiment,
                "links": links,
                "topic": topic,
            }
        )

    # Prepare plotting data
    # For hover, we only want to show the cluster size.
    point_sizes = np.array([cluster_sizes[int(c)] for c in lab], dtype=int)

    if PLOTLY_OK:
        # Build a minimal data dict
        data = {
            "x": x,
            "y": y,
            "cluster": lab.astype(int),
        }
        fig = px.scatter(
            data,
            x="x",
            y="y",
            color="cluster",
            title=title,
            opacity=0.9,
        )
        # Attach customdata (cluster size) and set hover to ONLY show size
        fig.update_traces(customdata=np.c_[lab], hovertemplate="label: %{customdata[0]}<extra></extra>")
        fig.update_traces(marker=dict(size=6, line=dict(width=0)))
        st.plotly_chart(fig, use_container_width=True)
    else:
        # Matplotlib fallback (hover not interactive here; shows colors only)
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        sc = ax.scatter(x, y, c=lab, s=12, alpha=0.9, cmap="tab20")
        ax.set_title(title)
        ax.set_xlabel("dim-1")
        ax.set_ylabel("dim-2")
        st.pyplot(fig)

    # Legend under the plot
    st.markdown("### Legend")
    for row in legend_rows:
        c = row["cluster"]
        size = row["size"]
        desc = row["description"]
        links = row["links"]
        relevancy = row["relevancy"]
        sentiment = row["sentiment"]
        topic = row["topic"]
        # Noise cluster (-1) callout
        label = "noise (-1)" if c == -1 else f"cluster {c}: {topic}"
        links_str = ", ".join([f"[link {i+1}]({u})" for i, u in enumerate(links)])
        st.markdown(f"- **{label}** — size: {size} - relevancy: {relevancy} - sentiment: {sentiment} — {desc}  \n  ↳ {links_str}")

def visualize_family_scores(
    urls: list[str],
    cluster_groups: dict[int, list[int]],
    cluster_scores: dict[int, float],
    title: str = "Average Cluster Scores by Domain",
    agg: str = 'mean',
):
    """
    Visualize aggregated cluster scores by URL family as a bar chart.

    Parameters
    ----------
    urls : list of str
        List of URLs.
    cluster_groups : list of (int, list[int])
        Mapping of cluster ID → indices of URLs in that cluster.
    cluster_scores : dict[int, float]
        Mapping of cluster ID → score (e.g., sentiment or relevancy).
    title : str
        Plot title.
    agg : {'mean', 'max', 'sum'}
        Aggregation function for per-domain scores.
    """
    st.subheader(title)

    # Compute family-level aggregates
    family_scores = compute_family_scores(urls, cluster_groups, cluster_scores, agg=agg)

    if not family_scores:
        st.warning("No family scores to display.")
        return

    # Create dataframe for plotting
    df = pd.DataFrame(list(family_scores.items()), columns=["Domain", "Score"])
    df = df.sort_values("Score", ascending=False)

    # Plot bar chart
    fig = px.bar(
        df,
        x="Domain",
        y="Score",
        text="Score",
        title=title,
        color="Score",
        color_continuous_scale="Blues",
    )
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside", cliponaxis=False)
    fig.update_layout(
        xaxis_title="Domain (Family)",
        yaxis_title="Aggregated Score",
        xaxis_tickangle=-30,
        plot_bgcolor="black",
    )

    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})