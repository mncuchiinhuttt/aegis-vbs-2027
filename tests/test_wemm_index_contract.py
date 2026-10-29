"""Qdrant consumer invariants for the pinned WeMM primary embedding."""

import sys
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import numpy as np
import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "inference-code"))

from models.embedding_identity import embedding_fingerprint, wemm_embedding_identity
from preprocessing.indexing.indexer import QdrantIndexer, SHOT_COLLECTION_NAME
from search.hybrid_search import EmbeddingIdentityError, HybridSearcher, VISUAL_COLLECTION_NAME


@pytest.fixture
def qdrant():
    client = QdrantClient(":memory:")
    yield client
    client.close()

def _create_collection(client, collection_name, dimension=3):
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
    )

def _indexer(client, identity):
    indexer = object.__new__(QdrantIndexer)
    indexer.client = client
    indexer.embedding_identity = identity
    indexer.embedding_fingerprint = embedding_fingerprint(identity)
    indexer.secondary_enabled = False
    indexer.batch_size = 8
    indexer._visual_buffer = []
    indexer._audio_buffer = []
    indexer._shot_buffer = []
    return indexer

def _searcher(client, identity):
    searcher = object.__new__(HybridSearcher)
    searcher.client = client
    searcher.embedding_identity = identity
    searcher.embedding_fingerprint = embedding_fingerprint(identity)
    searcher.visual_collection_name = VISUAL_COLLECTION_NAME
    searcher._verified_identity_collections = set()
    return searcher

def _id(label): return str(uuid5(NAMESPACE_URL, label))

def _point(point_id, vector, identity=None, **payload):
    point_payload = dict(payload)
    if identity is not None:
        point_payload["embedding_fingerprint"] = embedding_fingerprint(identity)
    return PointStruct(id=_id(point_id), vector=vector, payload=point_payload)

def test_matching_visual_identity_is_accepted_and_new_points_are_tagged(qdrant):
    identity = wemm_embedding_identity("revision-a", 3)
    _create_collection(qdrant, VISUAL_COLLECTION_NAME)
    qdrant.upsert(
        collection_name=VISUAL_COLLECTION_NAME,
        points=[_point("existing", [1.0, 0.0, 0.0], identity, modality="visual")],
    )

    indexer = _indexer(qdrant, identity)
    indexer._ensure_collection(
        VISUAL_COLLECTION_NAME,
        3,
        identity_fingerprint=indexer.embedding_fingerprint,
    )
    indexer.index_visual_point(_id("new"), np.array([0.0, 1.0, 0.0]), {"modality": "visual"})
    indexer.index_visual_point(_id("speech"), np.array([0.0, 0.0, 1.0]), {"modality": "speech"})
    indexer.flush_visual()

    points = qdrant.retrieve(
        collection_name=VISUAL_COLLECTION_NAME,
        ids=[_id(label) for label in ("existing", "new", "speech")],
        with_payload=True,
    )
    assert {point.payload["embedding_fingerprint"] for point in points} == {indexer.embedding_fingerprint}

def test_same_dimension_foreign_identity_is_rejected_without_deleting_points(qdrant):
    expected = wemm_embedding_identity("revision-current", 3)
    foreign = wemm_embedding_identity("revision-old", 3)
    _create_collection(qdrant, VISUAL_COLLECTION_NAME)
    qdrant.upsert(
        collection_name=VISUAL_COLLECTION_NAME,
        points=[_point("old", [1.0, 0.0, 0.0], foreign, modality="visual")],
    )

    indexer = _indexer(qdrant, expected)
    with pytest.raises(RuntimeError, match="unknown or incompatible embedding identity"):
        indexer._ensure_collection(
            VISUAL_COLLECTION_NAME,
            3,
            identity_fingerprint=indexer.embedding_fingerprint,
        )

    preserved = qdrant.retrieve(collection_name=VISUAL_COLLECTION_NAME, ids=[_id("old")])
    assert len(preserved) == 1
    assert preserved[0].payload["embedding_fingerprint"] == embedding_fingerprint(foreign)


def test_unknown_identity_is_rejected_without_deleting_points(qdrant):
    identity = wemm_embedding_identity("revision-current", 3)
    _create_collection(qdrant, VISUAL_COLLECTION_NAME)
    qdrant.upsert(
        collection_name=VISUAL_COLLECTION_NAME,
        points=[_point("legacy", [1.0, 0.0, 0.0], modality="visual")],
    )

    indexer = _indexer(qdrant, identity)
    with pytest.raises(RuntimeError, match="unknown or incompatible embedding identity"):
        indexer._ensure_collection(
            VISUAL_COLLECTION_NAME,
            3,
            identity_fingerprint=indexer.embedding_fingerprint,
        )

    assert [point.id for point in qdrant.retrieve(collection_name=VISUAL_COLLECTION_NAME, ids=[_id("legacy")])] == [_id("legacy")]


def test_mixed_identity_collection_is_rejected_and_preserved(qdrant):
    expected = wemm_embedding_identity("revision-current", 3)
    foreign = wemm_embedding_identity("revision-old", 3)
    _create_collection(qdrant, VISUAL_COLLECTION_NAME)
    qdrant.upsert(
        collection_name=VISUAL_COLLECTION_NAME,
        points=[
            _point("current", [0.0, 1.0, 0.0], expected, modality="visual"),
            _point("old", [1.0, 0.0, 0.0], foreign, modality="visual"),
        ],
    )

    indexer = _indexer(qdrant, expected)
    with pytest.raises(RuntimeError, match="unknown or incompatible embedding identity"):
        indexer._ensure_collection(
            VISUAL_COLLECTION_NAME,
            3,
            identity_fingerprint=indexer.embedding_fingerprint,
        )

    preserved = qdrant.retrieve(
        collection_name=VISUAL_COLLECTION_NAME,
        ids=[_id(label) for label in ("current", "old")],
        with_payload=True,
    )
    assert {point.id for point in preserved} == {_id("current"), _id("old")}


def test_search_identity_filter_excludes_foreign_vectors_from_dense_and_browse(qdrant):
    identity = wemm_embedding_identity("revision-current", 3)
    _create_collection(qdrant, VISUAL_COLLECTION_NAME)
    qdrant.upsert(
        collection_name=VISUAL_COLLECTION_NAME,
        points=[
            _point("valid", [0.5, 0.8660254, 0.0], identity, modality="visual", source_file="video.mp4"),
        ],
    )
    searcher = _searcher(qdrant, identity)
    searcher._validate_collection_identity(VISUAL_COLLECTION_NAME)

    # Simulate a legacy writer adding a foreign point after startup validation.
    qdrant.upsert(
        collection_name=VISUAL_COLLECTION_NAME,
        points=[_point("foreign", [1.0, 0.0, 0.0], modality="visual", source_file="video.mp4")],
    )

    dense = searcher._dense_search_vector(np.array([1.0, 0.0, 0.0]), top_k=5)
    browsed = searcher.get_all_points_for_video("video.mp4")

    assert [hit["id"] for hit in dense] == [_id("valid")]
    assert [point["id"] for point in browsed] == [_id("valid")]
    with pytest.raises(EmbeddingIdentityError, match="unknown or incompatible embedding identity"):
        searcher.get_point_vector(_id("foreign"))


def test_shot_search_identity_filter_excludes_foreign_shot_vectors(qdrant):
    identity = wemm_embedding_identity("revision-current", 3)
    _create_collection(qdrant, SHOT_COLLECTION_NAME)
    qdrant.upsert(
        collection_name=SHOT_COLLECTION_NAME,
        points=[_point("valid-shot", [0.0, 1.0, 0.0], identity, shot_id="shot-1")],
    )
    searcher = _searcher(qdrant, identity)
    searcher._validate_collection_identity(SHOT_COLLECTION_NAME)
    qdrant.upsert(
        collection_name=SHOT_COLLECTION_NAME,
        points=[_point("foreign-shot", [1.0, 0.0, 0.0], shot_id="shot-2")],
    )

    hits = searcher._shot_search_vector(np.array([0.0, 1.0, 0.0]), top_k=5)

    assert [hit["id"] for hit in hits] == [_id("valid-shot")]
