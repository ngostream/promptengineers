from typing import List, Dict
import numpy as np
from pydantic import BaseModel, Field
from sklearn.preprocessing import normalize
import hdbscan

class ClusterFromVectorsTool(BaseModel):
    """Cluster precomputed vectors and return groups (indices)."""
    vectors: List[List[float]]
    min_cluster_size: int = Field(5, ge=2, le=50)

    def execute(self) -> Dict:
        X = np.array(self.vectors, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X = normalize(X)
        clusterer = hdbscan.HDBSCAN(min_cluster_size=self.min_cluster_size, metric='euclidean')
        labels = clusterer.fit_predict(X)
        groups: Dict[int, List[int]] = {}
        for i, c in enumerate(labels):
            if c == -1:
                continue
            groups.setdefault(int(c), []).append(i)
        return {"labels": labels.tolist(), "groups": groups}
