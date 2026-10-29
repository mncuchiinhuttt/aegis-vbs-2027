import os
import torch
import numpy as np
from config import M2D_CLAP_MODEL_ID

from .wemm_embedding import WeMMEmbedding4BEmbedder


class M2DClapEmbedder:
    """
    M2D-CLAP sound embedding wrapper generating 768d vectors.
    """
    def __init__(self, model_path: str = M2D_CLAP_MODEL_ID):
        self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        
        # Check if local weights path exists under global weights/
        root_dir = os.path.dirname(os.path.dirname(__file__))
        if model_path is None or model_path == "":
            model_path = os.path.join(root_dir, "weights", "m2d_clap_vit_base-80x1001p16x16p16kpBpTI-2025", "checkpoint-30.pth")
            if not os.path.exists(model_path):
                model_path = os.path.join(root_dir, "weights", "checkpoint-30.pth")
        
        # Resolve model_path if relative
        if not os.path.isabs(model_path):
            abs_path = os.path.join(root_dir, model_path)
            if os.path.exists(abs_path):
                model_path = abs_path

        print(f"Loading local M2D-CLAP model from: {model_path}...")
        try:
            from .portable_m2d import PortableM2D
        except ImportError as exc:
            raise RuntimeError(
                "The M2D portable runtime (models/portable_m2d.py) is not installed. It is licensed "
                "by NTT for evaluation use and is not redistributed here. Read NOTICE.md, then run "
                "`python download_assets.py --accept-m2d-license` to fetch the required assets."
            ) from exc
        # flat_features=True is required for the CLAP head (audio_proj/sem_token):
        # without it, PortableM2D returns the "stacked" 768*5=3840-dim patch
        # representation instead of the 768-dim one sem_token expects, and the
        # concat inside AudioSemanticProj.forward crashes on every input
        self.model = PortableM2D(model_path, flat_features=True)
        self.model.to(self.device)
        self.dim = self.model.cfg.feature_d
        print("Local M2D-CLAP model loaded successfully.")

    def embed_text(self, text: str) -> np.ndarray:
        import torch
        with torch.no_grad():
            text_features = self.model.encode_clap_text([text])
        embedding = text_features[0].float().cpu().numpy()
            
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding

    def embed_audio(self, audio_data: np.ndarray, sampling_rate: int = 48000) -> np.ndarray:
        import torch
        import librosa
        # M2D CLAP model expects 16kHz audio input
        if sampling_rate != 16000:
            audio_data = librosa.resample(audio_data, orig_sr=sampling_rate, target_sr=16000)
        
        # shape must be [B, T]
        audio_tensor = torch.from_numpy(audio_data).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            audio_features = self.model.encode_clap_audio(audio_tensor)
        embedding = audio_features[0].float().cpu().numpy()
            
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding
