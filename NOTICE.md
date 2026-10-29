# Third-party notices

The code in this repository is MIT licensed (see `LICENSE`). It uses, adapts or downloads the components below,
which keep their own licenses. Licenses were checked against the upstream model cards, repositories and package
metadata when this file was written; check upstream before redistributing or using anything commercially.

## Terms that need your attention

| Component | Why it matters |
| --- | --- |
| **NTT M2D / M2D-CLAP** (`models/portable_m2d.py`, `weights/m2d_clap_*`) | NTT's evaluation agreement restricts use to internal, non-commercial evaluation and prohibits redistribution. The runtime is **not included** here. Read the [upstream agreement](https://github.com/nttcslab/m2d/blob/master/LICENSE.pdf) yourself; the full downloader requires `--accept-m2d-license` before fetching any assets. It acquires the unchanged runtime from a pinned NTT commit and verifies SHA-256. Full preprocessing needs it; text/visual retrieval does not. |
| **Ultralytics / YOLOE** (`ultralytics` package, `yoloe-26x-seg.pt`) | AGPL-3.0 (or an Ultralytics enterprise license). It is an optional runtime dependency used for object crops and is not bundled. Set `DETECTOR_OPTION=none` to run VQA on full frames without it. If you modify and serve it over a network, AGPL source-offer duties apply to you. |
| **SAM3** (`facebook/sam3`) | Gated Hugging Face repository with its own license; accept its terms and set `HF_TOKEN`. The full preprocessing entrypoint requires this model and setup fails if access is unavailable. Retrieval and the lightweight corpus indexer do not load SAM3. |
| **V3C** (Vimeo Creative Commons Collection) | Videos and keyframes are distributed by NIST under a data agreement and are **not** part of this repository. `evaluation/data/` contains text derived from V3C metadata (video titles) and V3C video ids only. See <https://videobrowsershowdown.org/about-vbs/existing-data-and-tools/>. |
| **Hosted VLM APIs** | `OPENAI_BASE_URL`/`OPENAI_API_KEY` point at a provider you choose; its terms, rate limits and data handling apply. No key is included. |

## Models and libraries (downloaded or installed by you)

| Component | Role | License (upstream) |
| --- | --- | --- |
| WeMM-Embedding-4B (`tencent/WeMM-Embedding-4B`) | Primary text/image encoder | Apache-2.0 (model card; third-party parts keep their own licenses) |
| Qwen2.5-VL-7B-Instruct | Optional local VLM | Apache-2.0 |
| SigLIP so400m (`google/siglip-so400m-patch14-384`) | Optional secondary encoder | Apache-2.0 |
| Whisper large-v3-turbo (CTranslate2 conversion) + faster-whisper | ASR | MIT |
| SmolVLM2-500M-Video-Instruct | Low-confidence OCR fallback | Apache-2.0 |
| PaddleOCR (PP-OCR) | OCR | Apache-2.0 |
| Real-ESRGAN | Small-crop upscaling | BSD-3-Clause |
| OpenAI CLIP | Scene-variance estimation | MIT |
| Qdrant (server and `qdrant-client`) | Vector database | Apache-2.0 |
| PyTorch, Transformers, OpenCV, FastAPI, OpenAI Python SDK | Runtime | See each package (permissive licenses) |

## Code adapted from other projects

- `preprocessing/video/transnetv2_pytorch.py` is adapted from the PyTorch inference model of
  [TransNetV2](https://github.com/soCzech/TransNetV2) (MIT, Copyright (c) 2020 Tomáš Souček).
- `webapp/frontend/src/components/ui/*` follow the [shadcn/ui](https://github.com/shadcn-ui/ui) component recipes
  (MIT, Copyright (c) 2023 shadcn).

MIT License text for the two entries above:

```
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of
the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
IN THE SOFTWARE.
```
