# --- Cluster visualization for Streamlit ---
# deps: numpy, scikit-learn, plotly (optional but nicer), streamlit
import hashlib
from typing import List, Dict, Optional
import numpy as np
import streamlit as st
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
        # TSNE is slower but can separate tangled clusters
        tsne = TSNE(n_components=2, learning_rate="auto", init="random",
                    perplexity=30, random_state=seed)
        return tsne.fit_transform(emb)

def visualize_clusters(
    embeddings: np.ndarray,
    labels: List[int],
    texts: List[str],
    urls: Optional[List[str]] = None,
    title: str = "Embedding clusters"
):
    """
    embeddings: (N, D) float32/float64
    labels: length-N ints from HDBSCAN (noise is -1)
    texts: length-N strings (chunk text)
    urls:  length-N strings (origin URL for each chunk)
    """
    assert len(embeddings) == len(labels) == len(texts), "Length mismatch"

    st.subheader(title)

    # Controls
    colA, colB, colC = st.columns([1,1,1])
    with colA:
        method = st.selectbox("Reducer", ["PCA (fast)", "t-SNE (slower)"], index=0)
    with colB:
        seed = st.number_input("Random seed", value=42, step=1)
    with colC:
        sample_n = st.slider("Max points to plot", min_value=200, max_value=len(labels), value=min(2000, len(labels)), step=100)

    # Optional subsample for speed in big runs
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

    # Reduce
    coords = _reduce_embeddings(emb_norm, method, seed)
    x, y = coords[:,0], coords[:,1]

    # Prepare data frame-like dict (no pandas required)
    data = {
        "x": x, "y": y,
        "cluster": lab.astype(int),
        "preview": [t[:220] + ("…" if len(t) > 220 else "") for t in txt],
        "url": url,
    }

    # Cluster stats
    unique, counts = np.unique(lab, return_counts=True)
    stats = sorted(zip(unique.tolist(), counts.tolist()), key=lambda z: (-z[1], z[0]))
    st.markdown("**Cluster sizes** (label → #points; -1 = noise)")
    st.write(", ".join([f"{c}: {n}" for c, n in stats]))

    # Plot
    if PLOTLY_OK:
        hover = [f"cluster: {{cluster}}<br>{'{preview}'}<br>{'{url}'}"]
        fig = px.scatter(
            data, x="x", y="y", color="cluster",
            hover_name="cluster",
            hover_data={"cluster": True, "x": False, "y": False, "preview": True, "url": True},
            title=title,
            opacity=0.85
        )
        fig.update_traces(marker=dict(size=6, line=dict(width=0)))
        st.plotly_chart(fig, use_container_width=True)
    else:
        # Fallback to matplotlib
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        sc = ax.scatter(data["x"], data["y"], c=data["cluster"], s=10, alpha=0.85, cmap="tab20")
        ax.set_title(title)
        ax.set_xlabel("dim-1")
        ax.set_ylabel("dim-2")
        st.pyplot(fig)

    # Optional: download CSV of the plotted points
    import csv, io
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["x","y","cluster","preview","url"])
    for i in range(len(x)):
        w.writerow([x[i], y[i], int(lab[i]), data["preview"][i], data["url"][i]])
    st.download_button("Download plotted points (CSV)", buf.getvalue(), file_name="cluster_points.csv", mime="text/csv")
