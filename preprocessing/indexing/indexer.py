import numpy as np
from typing import Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, TextIndexParams, TextIndexType, TokenizerType,
    Filter, FieldCondition, MatchValue, PayloadSchemaType,
)
from preprocessing.config import (
    QDRANT_HOST, QDRANT_PORT, QDRANT_API_KEY, QDRANT_UPSERT_BATCH_SIZE,
    SECONDARY_EMBEDDER_ENABLED, QDRANT_ALLOW_RECREATE, VISUAL_COLLECTION_NAME, SHOT_COLLECTION_NAME,
)
from models.embedding_identity import embedding_fingerprint, wemm_embedding_identity

# Name of the secondary (SigLIP) named vector queried via `using="siglip"`.
SECONDARY_VECTOR_NAME = "siglip"

class QdrantIndexer:
    """
    Manages connection to Qdrant, initializes collections, and pushes indexes.
    """
    def __init__(
        self,
        visual_dim: int = 2560,
        audio_dim: int = 768,
        secondary_dim: Optional[int] = None,
        *,
        embedding_identity: dict,
    ):
        if not isinstance(embedding_identity, dict) or embedding_identity != wemm_embedding_identity(
            embedding_identity.get("revision"), visual_dim
        ):
            raise ValueError("QdrantIndexer requires a WeMM native-embedding identity matching visual_dim.")
        self.embedding_identity = dict(embedding_identity)
        self.embedding_fingerprint = embedding_fingerprint(self.embedding_identity)
        print(f"Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}...")
        self.client = QdrantClient(
            host=QDRANT_HOST,
            port=QDRANT_PORT,
            api_key=QDRANT_API_KEY if QDRANT_API_KEY else None
        )
        self.batch_size = QDRANT_UPSERT_BATCH_SIZE
        self._visual_buffer = []
        self._audio_buffer = []
        self._shot_buffer = []
        self.secondary_enabled = SECONDARY_EMBEDDER_ENABLED and secondary_dim is not None
        self._init_collections(visual_dim, audio_dim, secondary_dim if self.secondary_enabled else None)

    def _validate_collection_identity(self, name: str, expected_fingerprint: str) -> int:
        """Return point count, refusing any nonempty mixed or unidentified collection."""
        total = self.client.count(collection_name=name, exact=True).count
        if total == 0:
            return 0
        identity_filter = Filter(
            must=[
                FieldCondition(
                    key="embedding_fingerprint",
                    match=MatchValue(value=expected_fingerprint),
                )
            ]
        )
        matching = self.client.count(
            collection_name=name,
            count_filter=identity_filter,
            exact=True,
        ).count
        if matching != total:
            raise RuntimeError(
                f"Qdrant collection '{name}' contains {total - matching} point(s) with an "
                "unknown or incompatible embedding identity. Existing points were not changed. "
                "Choose a new collection name and explicitly reindex; do not mix embeddings."
            )
        return total


    def _ensure_collection(
        self,
        name: str,
        dim: int,
        secondary_dim: Optional[int] = None,
        *,
        identity_fingerprint: Optional[str] = None,
    ):
        """
        Create missing collections and validate existing vector schemas.
        Identity-protected visual/shot collections are never deleted when
        populated; schema changes require an explicit new collection/reindex.
        """
        if identity_fingerprint is None and name in (VISUAL_COLLECTION_NAME, SHOT_COLLECTION_NAME):
            identity_fingerprint = self.embedding_fingerprint
        recreate = False
        collection_exists = self.client.collection_exists(name)
        existing_point_count = 0
        if collection_exists:
            if identity_fingerprint is not None:
                existing_point_count = self._validate_collection_identity(name, identity_fingerprint)
            try:
                info = self.client.get_collection(name)
                vectors_config = info.config.params.vectors
                is_named = isinstance(vectors_config, dict) and not hasattr(vectors_config, "size")

                if secondary_dim is not None and not is_named:
                    print(f"Collection '{name}' exists as single-vector but secondary embedder is "
                          f"enabled. Recreating with named vectors...")
                    recreate = True
                elif secondary_dim is None and is_named:
                    print(f"Collection '{name}' exists as named-vector but secondary embedder is "
                          f"disabled. Recreating with a single vector...")
                    recreate = True
                elif is_named:
                    current_size = vectors_config.get("default").size if vectors_config.get("default") else None
                    if current_size != dim:
                        print(f"Collection '{name}' exists but 'default' vector size {current_size} "
                              f"!= {dim}. Recreating...")
                        recreate = True
                else:
                    current_size = vectors_config.size if hasattr(vectors_config, "size") else vectors_config.get("size")
                    if current_size is not None and current_size != dim:
                        print(f"Collection '{name}' exists but has size {current_size} instead of {dim}. Recreating...")
                        recreate = True
            except Exception as e:
                # A timeout/permission error is not evidence of a schema
                # mismatch.  Never destroy an existing collection merely
                # because its metadata could not be read.
                raise RuntimeError(
                    f"Could not verify Qdrant collection '{name}'; refusing to delete it: {e}"
                ) from e
        else:
            recreate = True
        if recreate and identity_fingerprint is not None and existing_point_count:
            raise RuntimeError(
                f"Qdrant collection '{name}' has an incompatible vector schema and contains "
                f"{existing_point_count} verified points. Existing points were not deleted. "
                "Choose a new collection name and explicitly reindex."
            )

        if recreate:
            if collection_exists and not QDRANT_ALLOW_RECREATE:
                raise RuntimeError(
                    f"Qdrant collection '{name}' has an incompatible schema. "
                    "Set QDRANT_ALLOW_RECREATE=true only for an intentional full rebuild."
                )
            if collection_exists:
                self.client.delete_collection(name)
            if secondary_dim is not None:
                print(f"Creating collection '{name}' with named vectors (default={dim}, "
                      f"{SECONDARY_VECTOR_NAME}={secondary_dim})...")
                vectors_config = {
                    "default": VectorParams(size=dim, distance=Distance.COSINE),
                    SECONDARY_VECTOR_NAME: VectorParams(size=secondary_dim, distance=Distance.COSINE),
                }
            else:
                print(f"Creating collection '{name}' with size={dim}...")
                vectors_config = VectorParams(size=dim, distance=Distance.COSINE)
            self.client.create_collection(collection_name=name, vectors_config=vectors_config)

    def _ensure_text_index(self, collection_name: str, field_name: str):
        """Create a multilingual full-text payload index if it doesn't already exist."""
        info = self.client.get_collection(collection_name)
        if field_name in (info.payload_schema or {}):
            return
        print(f"Creating full-text index on '{collection_name}.{field_name}'...")
        self.client.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=TextIndexParams(
                type=TextIndexType.TEXT,
                tokenizer=TokenizerType.MULTILINGUAL,
                min_token_len=2,
                lowercase=True,
            )
        )

    def _ensure_embedding_fingerprint_index(self, collection_name: str) -> None:
        """Index the identity key used by cached guards and query filters."""
        info = self.client.get_collection(collection_name)
        if "embedding_fingerprint" in (info.payload_schema or {}):
            return
        self.client.create_payload_index(
            collection_name=collection_name,
            field_name="embedding_fingerprint",
            field_schema=PayloadSchemaType.KEYWORD,
        )

    def _init_collections(self, visual_dim: int = 2560, audio_dim: int = 768, secondary_dim: Optional[int] = None):
        self._ensure_collection(
            VISUAL_COLLECTION_NAME,
            visual_dim,
            secondary_dim,
            identity_fingerprint=self.embedding_fingerprint,
        )
        self._ensure_collection("audio_env_index", audio_dim)
        # H-EAGLE-lite uses a separate WeMM collection so its coarse stage
        # never changes the live frame-index schema.
        self._ensure_collection(
            SHOT_COLLECTION_NAME,
            visual_dim,
            identity_fingerprint=self.embedding_fingerprint,
        )
        self._ensure_embedding_fingerprint_index(VISUAL_COLLECTION_NAME)
        self._ensure_embedding_fingerprint_index(SHOT_COLLECTION_NAME)
        self._ensure_text_index(VISUAL_COLLECTION_NAME, "text_blob")
        self._ensure_text_index(SHOT_COLLECTION_NAME, "text_blob")

    def index_visual_point(
        self,
        point_id: str,
        vector: np.ndarray,
        payload: Dict[str, Any],
        secondary_vector: Optional[np.ndarray] = None,
    ):
        """
        Upload point to the configured visual collection. Payload format matches
        the point payload specification. When this indexer was constructed with
        secondary_enabled=True, secondary_vector (the SigLIP embedding) is
        stored as the named "siglip" vector alongside the primary one -
        pass None (or omit) to leave that named vector empty for this
        point, e.g. if secondary embedding failed for a single frame.
        """
        if self.secondary_enabled and secondary_vector is not None:
            vector_payload = {
                "default": vector.tolist(),
                SECONDARY_VECTOR_NAME: secondary_vector.tolist(),
            }
        else:
            vector_payload = vector.tolist()

        point = PointStruct(
            id=point_id,
            vector=vector_payload,
            payload={**payload, "embedding_fingerprint": self.embedding_fingerprint},
        )
        self._visual_buffer.append(point)
        if len(self._visual_buffer) >= self.batch_size:
            self.flush_visual()

    def index_audio_point(self, point_id: str, vector: np.ndarray, payload: Dict[str, Any]):
        """
        Upload point to audio_env_index collection.
        """
        point = PointStruct(
            id=point_id,
            vector=vector.tolist(),
            payload=payload
        )
        self._audio_buffer.append(point)
        if len(self._audio_buffer) >= self.batch_size:
            self.flush_audio()

    def index_shot_point(self, point_id: str, vector: np.ndarray, payload: Dict[str, Any]):
        """Buffer one H-EAGLE-lite shot aggregate vector."""
        point = PointStruct(
            id=point_id,
            vector=vector.tolist(),
            payload={**payload, "embedding_fingerprint": self.embedding_fingerprint},
        )
        self._shot_buffer.append(point)
        if len(self._shot_buffer) >= self.batch_size:
            self.flush_shots()

    def delete_shots_for_video(self, video_name: str) -> None:
        """Make a shot rebuild idempotent without touching frame points."""
        self.client.delete(
            collection_name=SHOT_COLLECTION_NAME,
            points_selector=Filter(
                must=[FieldCondition(key="source_file", match=MatchValue(value=video_name))]
            ),
            wait=True,
        )

    def delete_visual_for_video(self, video_name: str) -> None:
        """Explicit opt-in cleanup for a full video reindex."""
        self.client.delete(
            collection_name=VISUAL_COLLECTION_NAME,
            points_selector=Filter(
                must=[FieldCondition(key="source_file", match=MatchValue(value=video_name))]
            ),
            wait=True,
        )

    def flush_visual(self):
        """Upload all buffered visual/speech points in one Qdrant request."""
        if not self._visual_buffer:
            return
        points, self._visual_buffer = self._visual_buffer, []
        self.client.upsert(collection_name=VISUAL_COLLECTION_NAME, points=points)
        print(f"  Flushed {len(points)} visual points to Qdrant.")

    def flush_audio(self):
        """Upload all buffered ambient-audio points in one Qdrant request."""
        if not self._audio_buffer:
            return
        points, self._audio_buffer = self._audio_buffer, []
        self.client.upsert(collection_name="audio_env_index", points=points)
        print(f"  Flushed {len(points)} audio points to Qdrant.")

    def flush_shots(self):
        """Upload buffered H-EAGLE-lite shot aggregates."""
        points, self._shot_buffer = getattr(self, "_shot_buffer", []), []
        if not points:
            return
        self.client.upsert(collection_name=SHOT_COLLECTION_NAME, points=points)
        print(f"  Flushed {len(points)} shot points to Qdrant.")

    def flush(self):
        """Flush every pending collection; call at video and process boundaries."""
        self.flush_visual()
        self.flush_audio()
        self.flush_shots()
