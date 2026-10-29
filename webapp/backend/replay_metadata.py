"""Describe the loaded VQA path without loading models or exposing credentials."""


def _identifier(value):
    return value if isinstance(value, str) and value else None


def build_runtime_metadata(searcher, reranker, config, request, *, empty=False):
    embedder = getattr(searcher, "embedder", None)
    identity = getattr(embedder, "embedding_identity", None)
    identity = identity if isinstance(identity, dict) else {}
    collection = getattr(searcher, "visual_collection_name", None)
    vlm = getattr(reranker, "vlm", None)
    model_id = getattr(vlm, "model_name", None) or getattr(vlm, "model_id", None)
    return {
        "schema_version": 1,
        "embedding": {
            key: identity.get(key) for key in ("model", "revision", "dimension", "recipe")
        },
        "index": {
            "collection_name": _identifier(collection),
            "expected_embedding_fingerprint": _identifier(
                getattr(searcher, "embedding_fingerprint", None)
            ),
            "identity_check_completed": collection in getattr(
                searcher, "_verified_identity_collections", set()
            ),
        },
        "vlm": {
            "backend": type(vlm).__name__ if vlm is not None else None,
            "model": _identifier(model_id),
            "revision": _identifier(getattr(vlm, "model_revision", None)),
            "min_pixels": getattr(vlm, "min_pixels", None),
            "max_pixels": getattr(vlm, "max_pixels", None),
        },
        "vqa": {
            "worker_limit_requested": getattr(reranker, "vqa_max_workers", None),
            "worker_limit_used": 0 if empty else getattr(reranker, "last_vqa_workers", None),
            "verification_default": config.VERIFICATION_RERANK_ENABLED,
            "detector_active": getattr(reranker, "detector", None) is not None,
        },
        "retrieval": {
            "rrf_constant": config.RRF_CONSTANT,
            "exact_default": config.QDRANT_EXACT_SEARCH,
            "inner_weights": [config.RRF_DENSE_WEIGHT, config.RRF_LEXICAL_WEIGHT],
            "outer_weights": [
                config.RRF_QUERY_WEIGHT, config.RRF_HYDE_WEIGHT, config.RRF_SECONDARY_WEIGHT
            ],
            "secondary_embedder_active": getattr(searcher, "secondary_embedder", None) is not None,
        },
        "request": {
            "exact": request.exact,
            "hnsw_ef": request.hnsw_ef,
            "verify": request.verify,
            "reset_session": request.reset_session,
        },
    }
