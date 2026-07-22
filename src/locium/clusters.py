"""Discover clusters within a wing and name them.

A field of unlabelled dots is decoration. Clustering the projected coordinates
and pulling each cluster's most distinguishing terms out with TF-IDF is what
turns the map into something readable.
"""

import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer

_TOKEN_PATTERN = r"(?u)\b[a-zA-Z][a-zA-Z_\-]{2,}\b"


def cluster_labels(
    coords: np.ndarray,
    texts: list[str],
    min_cluster_size: int = 5,
    top_n: int = 3,
) -> tuple[list[dict], np.ndarray]:
    """Cluster 2D coordinates and label each cluster from its members' text."""
    n = len(coords)
    if n < min_cluster_size:
        return [], np.full(n, -1, dtype=int)

    assignments = HDBSCAN(min_cluster_size=min_cluster_size).fit_predict(coords)
    cluster_ids = sorted({int(c) for c in assignments if c >= 0})
    if not cluster_ids:
        return [], assignments

    documents = [
        " ".join(texts[i] for i in range(n) if assignments[i] == cid)
        for cid in cluster_ids
    ]
    vectorizer = TfidfVectorizer(
        stop_words="english", max_features=5000, token_pattern=_TOKEN_PATTERN
    )
    matrix = vectorizer.fit_transform(documents)
    terms = np.array(vectorizer.get_feature_names_out())

    clusters = []
    for row, cid in enumerate(cluster_ids):
        scores = matrix.getrow(row).toarray().ravel()
        label = " ".join(terms[np.argsort(-scores)[:top_n]])
        members = coords[assignments == cid]
        clusters.append(
            {
                "cluster": cid,
                "label": label,
                "centroid": [float(members[:, 0].mean()), float(members[:, 1].mean())],
            }
        )
    return clusters, assignments
