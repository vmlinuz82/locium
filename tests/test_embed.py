import numpy as np

from locium.embed import embed_query


def test_returns_a_384_dimensional_float32_vector():
    vector = embed_query("chromadb corruption")
    assert vector.shape == (384,)
    assert vector.dtype == np.float32


def test_result_is_normalised():
    norm = float(np.linalg.norm(embed_query("docker compose")))
    assert abs(norm - 1.0) < 1e-5


def test_embedding_is_stable_across_calls():
    assert np.allclose(embed_query("same text"), embed_query("same text"))


def test_related_text_scores_higher_than_unrelated():
    probe = embed_query("docker container compose file")
    close = embed_query("docker compose configuration")
    far = embed_query("voucher affiliate discount code")
    assert float(probe @ close) > float(probe @ far)


def test_empty_string_does_not_crash():
    assert embed_query("").shape == (384,)


def test_embed_query_matches_chromas_default_embedding_function():
    # This is the assertion that would catch a model swap: query vectors must
    # land in the same space as the drawer vectors ChromaDB stored, and Chroma
    # embeds drawers with its own DefaultEmbeddingFunction. Comparing against
    # that directly (rather than lexical-overlap proxies) pins the model.
    from chromadb.api.types import DefaultEmbeddingFunction

    text = "docker compose configuration"
    ours = embed_query(text)

    chroma_raw = np.asarray(DefaultEmbeddingFunction()([text])[0], dtype=np.float32)
    chroma_norm = float(np.linalg.norm(chroma_raw))
    theirs = chroma_raw / chroma_norm if chroma_norm > 1e-9 else chroma_raw

    assert np.allclose(ours, theirs, atol=1e-5)
