"""Sub-clustering for saturated chambers.

A mined palace lands most of its drawers in a handful of (wing, technical,
technical) chambers -- hundreds of drawers whose room name carries no
information. Clustering their embeddings recovers the structure the taxonomy
lost, and a TF-IDF label per cluster names it. This is layout-and-label only:
clusters subdivide the inside of one chamber, they never change what a
chamber is.

sklearn is imported lazily so the viewer process never pays for it -- only
``locium build`` does.
"""

import numpy as np


def cluster_count(n: int, cap: int) -> int:
    """Cluster count for a chamber of ``n`` drawers: grows slowly, capped.

    sqrt(n/60) puts a 150-drawer chamber at 2 and a ~1,700-drawer one at 5 --
    enough to name the major topics without shredding the room into slivers
    too small to label.
    """
    return max(2, min(cap, round((n / 60) ** 0.5)))


def assign_clusters(vectors: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Cluster ids (0..k-1) per row, deterministic for a given seed."""
    from sklearn.cluster import KMeans

    return KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(vectors)


def label_clusters(texts: list[str], assignments: np.ndarray, k: int) -> list[str]:
    """A short TF-IDF label per cluster, '' where no label can be made.

    Top mean-TF-IDF terms of a cluster's drawers, which surfaces what is
    distinctive about the cluster rather than what is common to the chamber.
    A vocabulary can legitimately come out empty (texts that are all
    stopwords or all symbols), so that case degrades to unlabelled clusters
    rather than failing the build.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectoriser = TfidfVectorizer(max_features=4000, stop_words="english")
    try:
        tfidf = vectoriser.fit_transform(texts)
    except ValueError:
        return [""] * k
    terms = vectoriser.get_feature_names_out()

    labels = []
    for cluster in range(k):
        rows = np.where(assignments == cluster)[0]
        if not len(rows):
            labels.append("")
            continue
        mean = np.asarray(tfidf[rows].mean(axis=0)).ravel()
        top = np.argsort(mean)[::-1][:3]
        labels.append(" · ".join(terms[t] for t in top if mean[t] > 0))
    return labels
