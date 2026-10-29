"""Official Tencent WeMM-Embedding-4B multimodal inference wrapper."""

import hashlib
import os
import re
import threading
from pathlib import Path
from typing import Union

import numpy as np
import torch
from PIL import Image

from config import EMBEDDING_MRL_DIM, WEMM_MODEL_REVISION
from .embedding_identity import wemm_embedding_identity


MODEL_ID = "tencent/WeMM-Embedding-4B"
DEFAULT_REVISION = "41a4fafdd26252ddff05c025413af7e61e26f08e"
_OFFICIAL_HELPER_SHA256 = "ac255e1fad459cc3e68891d6c3327f4486922aed02fb3c5c13fb53277ba8e94f"


class WeMMEmbedding4BEmbedder:
    """Encode text and images in WeMM's shared, normalized embedding space."""

    def __init__(
        self,
        revision: str = WEMM_MODEL_REVISION,
        mrl_dim: int | None = EMBEDDING_MRL_DIM,
        batch_size: int = 4,
    ) -> None:
        if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
            raise ValueError("WEMM_MODEL_REVISION must be a pinned 40-character commit SHA")
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")

        self.model_id = MODEL_ID
        self.model_revision = revision
        self.batch_size = batch_size
        self.device, dtype = self._select_device_and_dtype()
        model_path = self._model_source(revision)

        from transformers import AutoConfig, AutoModel, AutoProcessor

        config = AutoConfig.from_pretrained(model_path, revision=revision, trust_remote_code=False)
        native_dim = int(config.text_config.hidden_size)
        allowed_dims = tuple(int(dim) for dim in config.matryoshka_dimensions)
        if native_dim != 2560 or native_dim not in allowed_dims:
            raise RuntimeError(
                f"Unexpected WeMM model embedding config: native={native_dim}, allowed={allowed_dims}"
            )
        if mrl_dim is None:
            mrl_dim = native_dim
        if not isinstance(mrl_dim, int) or isinstance(mrl_dim, bool) or mrl_dim not in allowed_dims:
            raise ValueError(
                f"Unsupported WeMM MRL dimension {mrl_dim!r}; model allows {allowed_dims}"
            )
        self.dimension = mrl_dim

        self.processor = AutoProcessor.from_pretrained(
            model_path,
            revision=revision,
            trust_remote_code=True,
        )
        # The official last-attended-token pooling assumes right padding.
        self.processor.tokenizer.padding_side = "right"
        self.model = AutoModel.from_pretrained(
            model_path,
            revision=revision,
            code_revision=revision,
            trust_remote_code=True,
            dtype=dtype,
        ).to(self.device).eval()
        if int(self.model.config.text_config.hidden_size) != native_dim:
            raise RuntimeError("Loaded WeMM model dimensions differ from its validated config")

        self.embedding_identity = wemm_embedding_identity(self.model_revision, self.dimension)
        self._inference_lock = threading.RLock()

    @staticmethod
    def _select_device_and_dtype() -> tuple[str, torch.dtype]:
        if torch.cuda.is_available():
            # Native bf16 only: pre-Ampere GPUs (e.g. V100) report emulated support,
            # which runs several times slower than fp16 for this model.
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16
            return "cuda", dtype
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            return "mps", torch.float16
        return "cpu", torch.bfloat16

    @staticmethod
    def _model_source(revision: str) -> str:
        local_path = Path(__file__).resolve().parents[1] / "weights" / "WeMM-Embedding-4B"
        if not local_path.is_dir():
            return MODEL_ID
        if revision != DEFAULT_REVISION:
            raise RuntimeError(
                "Local WeMM weights are verified for the pinned default revision only; "
                "remove or update weights/WeMM-Embedding-4B for the configured revision"
            )
        helper_path = local_path / "modeling_wemm_embedding.py"
        if not helper_path.is_file():
            raise RuntimeError(f"Local WeMM checkpoint is missing {helper_path.name}")
        digest = hashlib.sha256(helper_path.read_bytes()).hexdigest()
        if digest != _OFFICIAL_HELPER_SHA256:
            raise RuntimeError("Local WeMM helper does not match the verified pinned upstream code")
        return os.fspath(local_path)

    @staticmethod
    def _image_messages(image: Union[Image.Image, np.ndarray]) -> list[dict]:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        if not isinstance(image, Image.Image):
            raise TypeError("Images must be PIL.Image.Image or numpy.ndarray values")
        return [{"role": "user", "content": [{"type": "image", "image": image.convert("RGB")}]}]

    def _image_batch(self, batch: list[Union[Image.Image, np.ndarray]]) -> list[np.ndarray]:
        from qwen_vl_utils import process_vision_info

        prompts = []
        images = []
        for image in batch:
            messages = self._image_messages(image)
            prompts.append(
                self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
            )
            image_inputs, _, _ = process_vision_info(
                messages,
                image_patch_size=16,
                return_video_kwargs=True,
                return_video_metadata=True,
            )
            if image_inputs is None:
                raise RuntimeError("WeMM image preprocessing produced no image input")
            images.extend(image_inputs if isinstance(image_inputs, list) else [image_inputs])

        inputs = self.processor(text=prompts, images=images, padding=True, return_tensors="pt")
        return self._encode(inputs, len(batch))

    def _encode(self, inputs, expected_count: int) -> list[np.ndarray]:
        inputs = inputs.to(self.device)
        with self._inference_lock, torch.inference_mode():
            vectors = self.model.embedding(**inputs)
        if vectors.ndim != 2 or vectors.shape[0] != expected_count or vectors.shape[1] < self.dimension:
            raise RuntimeError(f"Unexpected WeMM embedding output shape: {tuple(vectors.shape)}")

        rows = vectors.float().cpu().numpy()
        result = []
        for row in rows:
            vector = row[: self.dimension]
            norm = np.linalg.norm(vector)
            if not np.isfinite(norm) or norm <= 0:
                raise RuntimeError("WeMM returned a non-finite or zero embedding")
            result.append(vector / norm)
        return result

    def embed_images_batch(
        self, images: list[Union[Image.Image, np.ndarray]]
    ) -> list[np.ndarray]:
        """Encode images in bounded, real model batches while preserving order."""
        if not images:
            return []
        embeddings = []
        for start in range(0, len(images), self.batch_size):
            embeddings.extend(self._image_batch(images[start : start + self.batch_size]))
        return embeddings

    def embed_image(self, image: Union[Image.Image, np.ndarray]) -> np.ndarray:
        return self.embed_images_batch([image])[0]

    def embed_text(self, text: str) -> np.ndarray:
        if not isinstance(text, str):
            raise TypeError("Text embeddings require a string")
        messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        inputs = self.processor(text=prompt, return_tensors="pt", padding=True)
        return self._encode(inputs, 1)[0]
