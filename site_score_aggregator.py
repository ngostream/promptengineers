import pandas as pd
import tldextract
from collections import defaultdict

def compute_family_scores(urls, cluster_groups, cluster_scores, agg='mean'):
    """
    Aggregate cluster scores by URL family (domain).
    
    urls: list of str
    clusters: list of int (same length as urls)
    cluster_scores: dict[int, float]
    agg: 'mean', 'max', or 'sum'
    """
    # Extract base domain (e.g., 'reddit.com' from 'https://www.reddit.com/...').
    families = [f"{tldextract.extract(u).domain}.{tldextract.extract(u).suffix}" for u in urls]
    
    # Map family → list of cluster scores
    family_scores = defaultdict(list)
    for cid, idxs in cluster_groups.items():
        if cid not in cluster_scores:
            continue
        score = cluster_scores.get(cid)
        for idx in idxs:
            if 0 <= idx < len(families):
                family_scores[families[idx]].append(score)
    
    # Aggregate per family
    if agg == 'mean':
        aggregated = {fam: sum(scores)/len(scores) for fam, scores in family_scores.items() if scores}
    elif agg == 'max':
        aggregated = {fam: max(scores) for fam, scores in family_scores.items() if scores}
    elif agg == 'sum':
        aggregated = {fam: sum(scores) for fam, scores in family_scores.items() if scores}
    else:
        raise ValueError("agg must be 'mean', 'max', or 'sum'")
    return aggregated
