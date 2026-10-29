"""Stable identity metadata for primary WeMM embeddings stored in Qdrant."""

import hashlib
import json
from typing import Any


MODEL_ID = "tencent/WeMM-Embedding-4B"
EMBEDDING_RECIPE = "native-embedding-v1"


def wemm_embedding_identity(revision: str, dimension: int) -> dict[str, Any]:
    """Return the identity contract for one WeMM model revision/output size."""
    return {
        "model": MODEL_ID,
        "revision": revision,
        "dimension": dimension,
        "recipe": EMBEDDING_RECIPE,
    }


def embedding_fingerprint(identity: dict[str, Any]) -> str:
    """Hash identity as canonical JSON so equivalent dictionaries match."""
    canonical_identity = json.dumps(
        identity,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical_identity).hexdigest()
