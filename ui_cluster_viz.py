# ui_cluster_viz.py

import numpy as np
import streamlit as st
from typing import List, Dict, Optional
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

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
    labels: List[int],
    texts: List[str],
    urls: Optional[List[str]] = None,
    title: str = "Embedding clusters",
    cluster_meta: Optional[Dict[int, Dict[str, str]]] = None,
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
    assert len(embeddings) == len(labels) == len(texts), "Length mismatch"

    st.subheader(title)

    # Controls
    colA, colB, colC = st.columns([1, 1, 1])
    with colA:
        method = st.selectbox("Reducer", ["PCA (fast)", "t-SNE (slower)"], index=0)
    with colB:
        seed = st.number_input("Random seed", value=42, step=1)
    with colC:
        sample_n = st.slider(
            "Max points to plot",
            min_value=200,
            max_value=len(labels),
            value=min(2000, len(labels)),
            step=100,
        )

    # Subsample for speed if needed
    idx = np.arange(len(labels))
    if len(idx) > sample_n:
        rng = np.random.default_rng(seed)
        idx = rng.choice(idx, size=sample_n, replace=False)

    emb = np.asarray(embeddings)[idx]
    lab = np.asarray(labels)[idx]
    txt = [texts[i] for i in idx]
    url = [urls[i] if urls else "" for i in idx]

    # Normalize (cosine-friendly) if not already
    norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9
    emb_norm = emb / norms

    # 2D reduction
    coords = _reduce_embeddings(emb_norm, method, seed)
    x, y = coords[:, 0], coords[:, 1]

    # Cluster sizes (over the *plotted* subset)
    cluster_indices: Dict[int, List[int]] = {}
    for i, c in enumerate(lab):
        cluster_indices.setdefault(int(c), []).append(i)
    cluster_sizes = {c: len(ixs) for c, ixs in cluster_indices.items()}

    # Build legend entries (either from cluster_meta or derived)
    legend_rows = []
    for c, ixs in sorted(cluster_indices.items(), key=lambda kv: (kv[0] == -1, kv[0])):  # put -1 last
        size = len(ixs)
        if cluster_meta and c in cluster_meta:
            desc = (cluster_meta[c].get("description") or "").strip()
            link = (cluster_meta[c].get("link") or "").strip()
        else:
            # Derive: take a representative preview + a representative url
            # pick the shortest non-empty text as a concise “desc”
            rep_texts = [txt[i] for i in ixs if txt[i]]
            desc = min(rep_texts, key=len) if rep_texts else ""
            desc = (desc[:160] + "…") if len(desc) > 160 else desc
            rep_urls = [url[i] for i in ixs if url[i]]
            link = rep_urls[0] if rep_urls else ""

        legend_rows.append(
            {
                "cluster": c,
                "size": size,
                "description": desc if desc else "(no description)",
                "link": link,
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
        fig.update_traces(customdata=np.c_[point_sizes], hovertemplate="size: %{customdata[0]}<extra></extra>")
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
        link = row["link"]
        # Noise cluster (-1) callout
        label = "noise (-1)" if c == -1 else f"cluster {c}"
        if link:
            st.markdown(f"- **{label}** — size: {size} — {desc}  \n  ↳ [{link}]({link})")
        else:
            st.markdown(f"- **{label}** — size: {size} — {desc}")
