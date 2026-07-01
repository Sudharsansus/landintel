"""OCR over an FMB page — PaddleOCR (default) or Google Cloud Vision.

Two engine paths share the same output contract (:class:`OCRDetection`):

* **paddle** (default): render the full page at ``zoom`` magnification, run
  PaddleOCR PP-OCRv5, then augment with per-glyph isolation for any
  vector-encoded dimension numbers the page-level pass missed.

* **vision**: render the full page to PNG, POST it to the Google Cloud Vision
  TEXT_DETECTION endpoint, parse word-level bounding boxes.  Vision handles
  rotated and small raster text natively — the right choice for production FMB
  PDFs where numbers are rasterized images.  Falls back to PaddleOCR if Vision
  is unavailable (unconfigured credentials, quota exceeded, etc.).

Select the engine via the ``OCR_ENGINE`` environment variable / ``Settings``
field.  All positions are in **PDF page coordinates** (origin top-left, y down)
so ``anchor.py`` needs no changes regardless of which engine runs.

No cleanup happens here: "44,2" → 44.2 normalisation is ``validator.py``'s job.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
import numpy as np

from ...core.exceptions import OCRFailure
from ...core.models import Point

_log = logging.getLogger(__name__)

__all__ = [
    "OCRDetection",
    "FmbHeader",
    "extract_text",
    "parse_header",
    "DEFAULT_DET_MODEL",
    "DEFAULT_REC_MODEL",
    "DEFAULT_ZOOM",
]

# Y boundary (PDF points) between the header band and the drawing body.
# FMB A4 sheets (841 pt tall): survey metadata, scale, and area occupy the top
# ~130 pt. Below that is the drawing body where dimension numbers are colored
# vector fills — read cleanly by the glyph pass, not the page-level pass.
_HEADER_LIMIT_Y: float = 130.0

# --- NVIDIA DLL bootstrap (runs at import time) --------------------------------
# Pip-installed nvidia-* packages place CUDA DLLs under
# site-packages/nvidia/<pkg>/bin/.  The Windows C++ DLL loader needs these in
# os.environ["PATH"] before any onnxruntime C extension tries to load them.
# We inject them early (module level) so the first ort.InferenceSession call
# always finds cublasLt64_12.dll, cufft64_11.dll, etc.
try:
    import nvidia as _nvidia  # noqa: PLC0415
    _nv_root = Path(_nvidia.__file__).parent
    _extra = [
        str(_pkg / "bin")
        for _pkg in _nv_root.iterdir()
        if (_pkg / "bin").is_dir()
    ]
    if _extra:
        _cur_path = os.environ.get("PATH", "")
        _new = [p for p in _extra if p not in _cur_path]
        if _new:
            os.environ["PATH"] = os.pathsep.join(_new) + os.pathsep + _cur_path
except Exception:  # noqa: BLE001
    pass
# -------------------------------------------------------------------------------

DEFAULT_DET_MODEL = "PP-OCRv5_mobile_det"
DEFAULT_REC_MODEL = "en_PP-OCRv5_mobile_rec"
SERVER_DET_MODEL = "PP-OCRv5_server_det"
SERVER_REC_MODEL = "en_PP-OCRv5_server_rec"
_MOBILE_DET_MODEL = "PP-OCRv5_mobile_det"
DEFAULT_ZOOM = 4.0

# Max proximity (PDF points) between a glyph detection center and a page-level
# detection center for the glyph to be considered a duplicate and suppressed.
_GLYPH_DEDUP_RADIUS: float = 8.0


@dataclass(frozen=True)
class OCRDetection:
    """One recognized text box from any OCR engine.

    Attributes:
        text: The raw recognized string, untouched (commas, noise and all).
        confidence: Recognition confidence in ``[0, 1]``.
        polygon: The text box corners in PDF page coordinates.
        angle_deg: Undirected orientation of the text's longest edge in
            ``[0, 180°)``, or None.  Uses the same ``atan2(dy, dx) % 180``
            convention as ``anchor._orientation`` and
            ``pdf_glyphs._compute_angle``, so it is directly comparable with
            line orientations in ``anchor._angle_diff``.  Set by the paddlevl
            path; ``None`` on the paddle/glyph path (anchor infers orientation
            from polygon corner order instead).
        kind: Token classification set by the VL engine:
            ``"decimal_measurement"`` / ``"integer_marker"`` from the blue pass;
            ``"corner_label"`` / ``"chain_point"`` / ``"red_token"`` from the red pass;
            ``"unknown"`` for PP-OCRv5 and Vision paths.
    """

    text: str
    confidence: float
    polygon: tuple[Point, ...]
    angle_deg: float | None = None
    kind: str = "unknown"

    @property
    def center(self) -> Point:
        """Centroid of the box in PDF page coordinates (for anchoring)."""
        xs = [p[0] for p in self.polygon]
        ys = [p[1] for p in self.polygon]
        return (sum(xs) / len(xs), sum(ys) / len(ys))


def _setup_onnx_gpu() -> bool:
    """Add NVIDIA pip-package DLL dirs to PATH and check CUDAExecutionProvider.

    Pip-installed nvidia-* packages keep their DLLs under
    ``site-packages/nvidia/<pkg>/bin/``.  Windows' C++ DLL loader uses the
    process PATH (os.environ["PATH"]), not Python's add_dll_directory, so we
    prepend each bin dir to PATH.  Idempotent — safe to call many times.
    """
    try:
        import nvidia  # noqa: PLC0415
        nv_root = Path(nvidia.__file__).parent
        extra_paths = []
        for pkg_dir in nv_root.iterdir():
            bin_dir = pkg_dir / "bin"
            if bin_dir.is_dir():
                s = str(bin_dir)
                if s not in os.environ.get("PATH", ""):
                    extra_paths.append(s)
        if extra_paths:
            os.environ["PATH"] = os.pathsep.join(extra_paths) + os.pathsep + os.environ.get("PATH", "")
        import onnxruntime as ort  # noqa: PLC0415
        return "CUDAExecutionProvider" in ort.get_available_providers()
    except Exception:  # noqa: BLE001
        return False


_onnx_gpu_available: bool | None = None  # lazy probe, None = not yet checked


def _onnx_gpu_ready() -> bool:
    global _onnx_gpu_available  # noqa: PLW0603
    if _onnx_gpu_available is None:
        _onnx_gpu_available = _setup_onnx_gpu()
    return _onnx_gpu_available


def onnx_trt_available() -> bool:
    """Whether onnxruntime exposes the TensorRT execution provider on this host."""
    try:
        import onnxruntime as ort  # noqa: PLC0415
        return "TensorrtExecutionProvider" in ort.get_available_providers()
    except Exception:  # noqa: BLE001
        return False


def _configure_trt() -> bool:
    """Enable TensorRT for the ONNX server-det/rec engines (batch throughput).

    Gated by ``LANDINTEL_OCR_TRT`` (default "0" = OFF; set "1" to opt in). DEFAULT OFF
    because the TensorRT EP's first-build engine plan can hang/fail during PaddleOCR
    construction on this SM_120 host, which silently drops the whole ONNX path to the
    mobile-det CPU fallback (measured 2026-07-01). Plain CUDAExecutionProvider needs no
    build step, loads in ~0.4 s, and runs the server-det at ~19 ms/inference on GPU -- so
    TRT buys nothing here and only adds a failure surface. When enabled AND available, it
    sets the engine cache + FP16 env so the (slow) first build is reused across runs.
    Returns True when TRT is configured; no-op + False otherwise (the CUDA path is
    unchanged).
    """
    if os.environ.get("LANDINTEL_OCR_TRT", "0") == "0":
        return False
    if not onnx_trt_available():
        return False
    cache = os.environ.get("LANDINTEL_TRT_CACHE",
                           str(Path(os.environ.get("PADDLE_HOME",
                               str(Path.home() / ".paddlex"))) / "trt_cache"))
    Path(cache).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("ORT_TENSORRT_ENGINE_CACHE_ENABLE", "1")
    os.environ.setdefault("ORT_TENSORRT_CACHE_PATH", cache)
    os.environ.setdefault("ORT_TENSORRT_FP16_ENABLE", "1")
    _log.info("TensorRT EP enabled for ONNX OCR (FP16, engine cache at %s)", cache)
    return True


def _force_onnx_cuda_providers() -> None:
    """Force PaddleX's ONNX Runtime runner onto CUDAExecutionProvider.

    PaddleX derives the ONNX session's execution provider from ``device_type``, which it
    resolves via ``parse_device`` -> ``paddle.device.is_compiled_with_cuda()``. The active
    paddle here is a CPU build (there is no paddle-gpu wheel for Blackwell SM_120), so paddle
    reports "no CUDA" and PaddleX silently downgrades the ONNX session to CPUExecutionProvider
    -- even though onnxruntime-CUDA is fully working on this card (measured ~19 ms/inference,
    GPU util 92%). onnxruntime's CUDA EP is INDEPENDENT of paddle's build, so we patch the
    runner's provider default to CUDA (falling back to CPU) whenever onnxruntime-GPU is ready.
    Idempotent; only touches the ONNX runner, never paddle's own device logic."""
    try:
        from paddlex.inference.models.runners.onnxruntime_runner import (  # noqa: PLC0415
            ONNXRuntimeRunner,
        )
    except Exception as exc:  # noqa: BLE001
        _log.warning("cannot patch ONNX runner providers (%s); ONNX may run on CPU", exc)
        return
    if getattr(ONNXRuntimeRunner, "_landintel_cuda_forced", False):
        return

    def _default_providers(self):  # noqa: ANN001, ANN202
        device_id = self._config.get("device_id") or 0
        return (["CUDAExecutionProvider", "CPUExecutionProvider"],
                [{"device_id": device_id}, {}])

    ONNXRuntimeRunner._default_providers = _default_providers
    ONNXRuntimeRunner._landintel_cuda_forced = True
    _log.info("Patched PaddleX ONNX runner -> CUDAExecutionProvider (bypasses paddle-CPU downgrade)")


def _paddlex_model_dir(model_name: str) -> Path:
    """Return the paddlex official_models directory for ``model_name``."""
    home = Path(os.environ.get("PADDLE_HOME", str(Path.home() / ".paddlex")))
    return home / "official_models" / model_name


# --- HuggingFace Qwen2.5-VL GPU engine (PyTorch CUDA 12.8 / SM_120 safe) ----

# Real Qwen2.5-VL model (the old "0.9B" name does not exist on the Hub). The 3B-Instruct is the
# smallest official Qwen2.5-VL and fits easily on a 48 GB card. Override with LANDINTEL_VL_MODEL.
_HF_VL_MODEL_NAME = os.environ.get("LANDINTEL_VL_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")
# None = not yet attempted; Exception = failed (don't retry); tuple = loaded OK
_hf_florence_engine: tuple[Any, Any] | Exception | None = None


def _build_hf_qwen_engine() -> tuple[Any, Any]:
    """Load Qwen2.5-VL-0.9B on GPU via HuggingFace Transformers.

    Qwen2.5-VL reads rotated, small measurement numbers reliably.
    PyTorch 2.11+cu128 supports Blackwell SM_120 natively — no PaddlePaddle GPU needed.
    Downloads ~2 GB on first call (cached in ~/.cache/huggingface/hub/).
    """
    global _hf_florence_engine  # noqa: PLW0603
    if isinstance(_hf_florence_engine, Exception):
        raise _hf_florence_engine
    if _hf_florence_engine is not None:
        return _hf_florence_engine

    import torch  # noqa: PLC0415
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor  # noqa: PLC0415

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # HARD GPU SAFETY CEILING: cap this process at a fraction of VRAM so a large/dense plot's
    # batch can never drive the card to the OOM edge (an OOM there would crash and waste the
    # whole resumable run). Default 0.80 leaves comfortable headroom on top of the model +
    # activations; override with LANDINTEL_VL_MEM_FRAC. Pair with a modest LANDINTEL_VL_BATCH.
    if device == "cuda":
        try:
            frac = float(os.environ.get("LANDINTEL_VL_MEM_FRAC", "0.80"))
            torch.cuda.set_per_process_memory_fraction(max(0.1, min(frac, 1.0)), 0)
            _log.info("GPU memory ceiling set to %.0f%% of VRAM", frac * 100)
        except Exception as exc:  # noqa: BLE001
            _log.warning("could not set GPU memory ceiling: %s", exc)
    _log.info("Loading %s on %s...", _HF_VL_MODEL_NAME, device)

    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            _HF_VL_MODEL_NAME,
            torch_dtype=torch.float16,
            device_map=device,
        )
        model.eval()
        processor = AutoProcessor.from_pretrained(
            _HF_VL_MODEL_NAME,
            min_pixels=256 * 28 * 28,
            max_pixels=1280 * 28 * 28,
        )
    except Exception as exc:  # noqa: BLE001
        _hf_florence_engine = exc
        raise

    _log.info("%s loaded on %s", _HF_VL_MODEL_NAME, device)
    _hf_florence_engine = (model, processor)
    return _hf_florence_engine


def _ocr_canvas_hf_vl(img_gray: np.ndarray) -> tuple[str, float] | None:
    """OCR a de-rotated glyph canvas with Qwen2.5-VL-0.9B on GPU.

    Prompted specifically for FMB measurement numbers; validates the response
    against the measurement pattern before returning.
    Returns (text, confidence) or None.
    """
    import torch  # noqa: PLC0415

    try:
        model, processor = _build_hf_qwen_engine()
    except Exception as exc:  # noqa: BLE001
        _log.warning("Qwen2.5-VL not available (%s); skipping glyph", exc)
        return None

    from PIL import Image  # noqa: PLC0415
    from qwen_vl_utils import process_vision_info  # noqa: PLC0415

    img_rgb = np.stack([img_gray, img_gray, img_gray], axis=-1)
    pil_img = Image.fromarray(img_rgb.astype(np.uint8))

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": pil_img},
            {"type": "text", "text": (
                "This image contains a single land survey measurement number "
                "from a Tamil Nadu government FMB document. "
                "Read the number exactly as written. "
                "Output ONLY the number, nothing else. "
                "Examples of valid outputs: 69.0  34.6  22.8  (160.2)  78.0  13.4"
            )},
        ],
    }]

    text_prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, _ = process_vision_info(messages)
    device = next(model.parameters()).device
    inputs = processor(
        text=[text_prompt],
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=20, temperature=0.1, do_sample=False)

    out = processor.batch_decode(
        [generated_ids[0][inputs.input_ids.shape[1]:]],
        skip_special_tokens=True,
    )[0].strip()

    # Validate: must look like a measurement number
    clean = out.replace(" ", "").replace(",", ".")
    if _MEAS_RE.match(clean):
        return (clean, 0.95)
    return None


_HF_VL_PROMPT = (
    "This image contains a single land survey measurement number "
    "from a Tamil Nadu government FMB document. "
    "Read the number exactly as written. "
    "Output ONLY the number, nothing else. "
    "Examples of valid outputs: 69.0  34.6  22.8  (160.2)  78.0  13.4"
)


def _ocr_canvases_hf_vl_batch(canvases: list[np.ndarray], batch_size: int = 16
                              ) -> list[tuple[str, float] | None]:
    """Batched Qwen2.5-VL OCR of many glyph canvases -- IDENTICAL model/prompt/params to the
    per-canvas path, just N images per GPU call so the fixed generate() overhead is amortised
    (major speedup on dense plots). Returns one (text, conf)|None per input canvas, in order.
    """
    import torch  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415
    from qwen_vl_utils import process_vision_info  # noqa: PLC0415

    try:
        model, processor = _build_hf_qwen_engine()
    except Exception as exc:  # noqa: BLE001
        _log.warning("Qwen2.5-VL not available (%s); skipping glyph batch", exc)
        return [None] * len(canvases)
    device = next(model.parameters()).device
    bs = int(os.environ.get("LANDINTEL_VL_BATCH", str(batch_size)))
    results: list[tuple[str, float] | None] = [None] * len(canvases)

    for start in range(0, len(canvases), bs):
        chunk = canvases[start:start + bs]
        msgs = []
        for g in chunk:
            pil = Image.fromarray(np.stack([g, g, g], axis=-1).astype(np.uint8))
            msgs.append([{"role": "user", "content": [
                {"type": "image", "image": pil}, {"type": "text", "text": _HF_VL_PROMPT}]}])
        texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                 for m in msgs]
        imgs = []
        for m in msgs:
            ii, _ = process_vision_info(m)
            imgs.extend(ii)
        inputs = processor(text=texts, images=imgs, padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        trimmed = gen[:, inputs.input_ids.shape[1]:]
        outs = processor.batch_decode(trimmed, skip_special_tokens=True)
        for k, out in enumerate(outs):
            clean = out.strip().replace(" ", "").replace(",", ".")
            if _MEAS_RE.match(clean):
                results[start + k] = (clean, 0.95)
    return results


def _extract_glyph_detections_hf_vl(
    pdf_path: Path,
    page_number: int,
) -> list["OCRDetection"]:
    """Glyph extraction with HF Qwen2.5-VL for per-canvas OCR.

    Same position logic as ``_extract_glyph_detections`` (vector geometry is
    unchanged) but replaces PP-OCRv5 with Qwen2.5-VL for each glyph canvas.
    Dramatically improves recall on small, rotated measurement numbers.
    """
    from .pdf_glyphs import extract_glyph_groups, render_glyph_group, glyph_polygon  # noqa: PLC0415

    detections: list[OCRDetection] = []
    try:
        with fitz.open(pdf_path) as doc:
            page = doc[page_number]
            groups = extract_glyph_groups(page)
    except Exception:  # noqa: BLE001
        return detections

    # Pre-check: try to load Florence-2 once before the loop so we don't log
    # a per-glyph warning when the engine is unavailable.
    try:
        _build_hf_qwen_engine()
    except Exception as exc:  # noqa: BLE001
        _log.warning("Florence-2 unavailable (%s); skipping glyph body", exc)
        return detections

    _log.info("HF VL glyph pass: %d glyph groups on %s", len(groups), pdf_path.name)

    # Render every canvas first (upright + 180-flip per group), then OCR them all in ONE
    # batched pass -- same VLM/prompt as the per-canvas path, just amortised GPU calls.
    rendered: list[tuple[Any, int]] = []      # (group, base_index_into_canvases)
    canvases: list[np.ndarray] = []
    for group in groups:
        try:
            img1 = render_glyph_group(group)
            img2 = render_glyph_group(group, flip=True)
        except Exception:  # noqa: BLE001
            continue
        rendered.append((group, len(canvases)))
        canvases.extend((img1, img2))

    batch = _ocr_canvases_hf_vl_batch(canvases)

    for group, base in rendered:
        r1, r2 = batch[base], batch[base + 1]
        if r1 is None and r2 is None:
            continue
        if r1 is None:
            text, conf = r2
        elif r2 is None:
            text, conf = r1
        else:
            text, conf = r1 if r1[1] >= r2[1] else r2
        try:
            polygon = glyph_polygon(group)
            detections.append(OCRDetection(text=text, confidence=conf, polygon=polygon))
        except Exception:  # noqa: BLE001
            continue

    _log.info("HF VL glyph pass: %d/%d groups read", len(detections), len(groups))
    return detections


@lru_cache(maxsize=4)
def _build_engine(det_model: str, rec_model: str, use_mkldnn: bool) -> Any:
    """Construct and cache a PaddleOCR engine (construction is expensive).

    Cached by configuration so repeated calls in a worker reuse one engine.
    Document-orientation/unwarping/textline-orientation are disabled: FMB sheets
    are already upright, so those stages only add latency and failure surface.

    When det_model is SERVER_DET_MODEL, attempts ONNX Runtime GPU inference
    (Blackwell RTX PRO 5000 / SM_120 compatible via nvidia pip packages).
    Falls back to mobile_det on CPU if ONNX GPU is unavailable.
    """
    from paddleocr import PaddleOCR  # noqa: PLC0415

    # --- ONNX Runtime GPU path for server_det (bypasses paddle SM_120 limit) ---
    if det_model == SERVER_DET_MODEL:
        det_onnx = _paddlex_model_dir(det_model) / "inference.onnx"
        rec_onnx = _paddlex_model_dir(rec_model) / "inference.onnx"
        if det_onnx.exists() and rec_onnx.exists() and _onnx_gpu_ready():
            try:
                import paddleocr._common_args as _ca  # noqa: PLC0415
                if "onnxruntime" not in _ca.SUPPORTED_INFERENCE_ENGINE_LIST:
                    _ca.SUPPORTED_INFERENCE_ENGINE_LIST.append("onnxruntime")
                _trt = _configure_trt()
                _force_onnx_cuda_providers()   # bypass paddle-CPU's bogus gpu->cpu downgrade
                _log.info("Using server_det via ONNX Runtime GPU (%s)",
                          "TensorRT+CUDA" if _trt else "CUDAExecutionProvider")
                rec_dir = _paddlex_model_dir(rec_model)
                # rec batch size: bigger = better GPU utilisation on dense pages (more text
                # crops per inference call). Default 32; raise via LANDINTEL_REC_BATCH on the
                # 48 GB card for batch throughput. CPU path keeps 32 (no GPU to feed).
                _rec_batch = int(os.environ.get("LANDINTEL_REC_BATCH", "32"))
                return PaddleOCR(
                    text_detection_model_name=det_model,
                    text_recognition_model_name=rec_model,
                    text_recognition_model_dir=str(rec_dir),
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                    # CAP the detection input side. An FMB page rendered at zoom 4 is ~4000 px;
                    # feeding that full-res to the server-det ONNX graph on CUDA allocates the
                    # whole card (~23 GB) and stalls. Limiting the max side keeps GPU memory
                    # bounded so the CUDA session actually runs. Tunable via LANDINTEL_DET_SIDE.
                    text_det_limit_side_len=int(os.environ.get("LANDINTEL_DET_SIDE", "1920")),
                    text_det_limit_type="max",
                    text_det_thresh=0.3,
                    text_det_box_thresh=0.6,
                    text_det_unclip_ratio=1.5,
                    text_recognition_batch_size=_rec_batch,
                    device="gpu",
                    engine="onnxruntime",
                )
            except Exception as exc:  # noqa: BLE001
                _log.warning(
                    "server_det ONNX GPU engine failed (%s); falling back to mobile_det CPU",
                    exc,
                )
        else:
            _log.warning(
                "server_det ONNX GPU unavailable (onnx=%s rec_onnx=%s gpu=%s); "
                "falling back to mobile_det CPU",
                det_onnx.exists(), rec_onnx.exists(), _onnx_gpu_ready(),
            )
        det_model = _MOBILE_DET_MODEL  # safe CPU fallback

    # --- CPU path (mobile_det or any non-server model) ---
    return PaddleOCR(
        text_detection_model_name=det_model,
        text_recognition_model_name=rec_model,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        text_det_thresh=0.2,
        text_det_box_thresh=0.4,
        text_det_unclip_ratio=2.0,
        text_recognition_batch_size=32,
        device="cpu",
        enable_mkldnn=use_mkldnn,
    )


def _body_ocr_engine(use_mkldnn: bool = False) -> Any:
    """Engine for the BODY / header / glyph OCR passes. Prefers the server-det ONNX engine
    on CUDA (GPU) so these passes -- the bulk of M1 OCR time -- run on the GPU instead of the
    mobile CPU detector. ``_build_engine`` transparently falls back to mobile-det CPU when the
    ONNX/GPU path is unavailable, so this is safe on CPU-only hosts. Historically these passes
    were hard-pinned to DEFAULT_DET_MODEL (mobile CPU), which left the GPU idle even when
    server_det was requested -- fixed 2026-07-01. Force mobile with LANDINTEL_BODY_GPU=0."""
    if os.environ.get("LANDINTEL_BODY_GPU", "1") == "1":
        return _build_engine(SERVER_DET_MODEL, DEFAULT_REC_MODEL, use_mkldnn)
    return _build_engine(DEFAULT_DET_MODEL, DEFAULT_REC_MODEL, use_mkldnn)


def _render_page(pdf_path: Path, page_number: int, zoom: float) -> np.ndarray:
    """Rasterize one PDF page to an RGB ndarray at ``zoom`` magnification."""
    with fitz.open(pdf_path) as doc:
        page = doc[page_number]
        matrix = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        image = np.frombuffer(pix.samples, dtype=np.uint8)
        return image.reshape(pix.height, pix.width, pix.n).copy()


def _parse_result(raw: Any, zoom: float) -> list[OCRDetection]:
    """Turn PaddleOCR's predict() output into :class:`OCRDetection` objects.

    PP-OCRv5 (paddleocr 3.x) returns a list with one result per input image;
    each result is dict-like with parallel ``rec_texts`` / ``rec_scores`` and a
    polygon list. Pixel polygons are divided by ``zoom`` back to PDF page space.
    """
    if not raw:
        return []
    page_result = raw[0]
    texts = page_result["rec_texts"]
    scores = page_result["rec_scores"]
    polys = page_result.get("rec_polys")
    if polys is None:
        polys = page_result.get("dt_polys", [])

    detections: list[OCRDetection] = []
    for text, score, poly in zip(texts, scores, polys):
        # Skip empty/whitespace recognitions: PP-OCRv5 sometimes detects a box
        # but recognizes no characters. That is a non-result, not a measurement.
        # The recognized text itself is left untouched (no comma/decimal cleanup).
        if not str(text).strip():
            continue
        corners = tuple((float(x) / zoom, float(y) / zoom) for x, y in poly)
        detections.append(
            OCRDetection(text=str(text), confidence=float(score), polygon=corners)
        )
    return detections


def _ocr_canvas(img_gray: np.ndarray, engine: Any) -> tuple[str, float] | None:
    """OCR a grayscale glyph canvas; return (text, confidence) or None.

    The canvas is expected to contain one isolated measurement number. We take
    the highest-confidence detection and return it verbatim (no normalisation).
    """
    img_rgb = np.stack([img_gray, img_gray, img_gray], axis=-1)
    try:
        raw = engine.predict(img_rgb)
    except Exception:  # noqa: BLE001
        return None
    if not raw:
        return None
    result = raw[0]
    texts = result.get("rec_texts", [])
    scores = result.get("rec_scores", [])
    if not texts:
        return None
    best_text, best_conf = max(zip(texts, scores), key=lambda ts: ts[1])
    text = str(best_text).strip()
    return (text, float(best_conf)) if text else None


def _ocr_canvas_all(
    img_gray: np.ndarray,
    engine: Any,
) -> list[tuple[str, float, tuple[float, float]]]:
    """OCR a canvas; return ALL detected text regions as (text, conf, canvas_center).

    Unlike _ocr_canvas which returns only the single best result, this captures
    every text box detected by the engine.  Used for multi-measurement cluster
    canvases where several adjacent fills from different measurements were grouped
    together — the detection step finds each text region independently.
    """
    img_rgb = np.stack([img_gray, img_gray, img_gray], axis=-1)
    try:
        raw = engine.predict(img_rgb)
    except Exception:  # noqa: BLE001
        return []
    if not raw:
        return []
    result = raw[0]
    texts = result.get("rec_texts", [])
    scores = result.get("rec_scores", [])
    boxes = result.get("dt_polys", [])
    out: list[tuple[str, float, tuple[float, float]]] = []
    for i, (text, score) in enumerate(zip(texts, scores)):
        text = str(text).strip()
        if not text:
            continue
        if i < len(boxes) and boxes[i] is not None:
            pts = np.asarray(boxes[i], dtype=float)
            cx = float(pts[:, 0].mean())
            cy = float(pts[:, 1].mean())
        else:
            cx, cy = float(img_gray.shape[1]) / 2.0, float(img_gray.shape[0]) / 2.0
        out.append((text, float(score), (cx, cy)))
    return out


_MEAS_RE = re.compile(r"^\(?\d{1,4}[.,]\d{1,2}\)?$")
# Perchar: scan a concatenated string for multiple embedded measurement tokens.
_PERCHAR_MEAS_RE = re.compile(r"\(?\d{1,4}[.,]\d{1,2}\)?")
_MULTIANGLE_CONF_THRESHOLD = 0.60  # retry with extra rotations below this
_MIN_GLYPH_CONF = 0.75


def _find_nearest_line(
    center: tuple[float, float],
    segments: list[Any],
    max_dist: float = 50.0,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Return the nearest segment (start, end) to ``center``, or None.

    Uses point-to-segment distance so a long diagonal line registers as close
    even when the glyph sits beside its middle rather than near an endpoint.
    Ignores segments more than ``max_dist`` PDF points away.
    """
    cx, cy = center
    best_dist = max_dist
    best_seg: tuple[tuple[float, float], tuple[float, float]] | None = None
    for seg in segments:
        x1, y1 = seg.start
        x2, y2 = seg.end
        dx, dy = x2 - x1, y2 - y1
        dlen2 = dx * dx + dy * dy
        if dlen2 < 1e-6:
            dist = math.hypot(cx - x1, cy - y1)
        else:
            t = max(0.0, min(1.0, ((cx - x1) * dx + (cy - y1) * dy) / dlen2))
            dist = math.hypot(cx - (x1 + t * dx), cy - (y1 + t * dy))
        if dist < best_dist:
            best_dist = dist
            best_seg = (seg.start, seg.end)
    return best_seg


def _seg_angle_deg(seg: tuple[tuple[float, float], tuple[float, float]]) -> float:
    """Undirected angle [0, 180) of a segment — same convention as anchor._orientation."""
    (x1, y1), (x2, y2) = seg
    return math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180.0


def _preprocess_canvas(canvas_gray: np.ndarray) -> np.ndarray:
    """Prepare a glyph canvas for PP-OCRv5: scale, threshold, pad.

    PP-OCRv5 is calibrated for ~32-64px text height. FMB fills at 4× zoom are
    ~28-40px, so we scale to 64px height. Adaptive thresholding sharpens edges
    blurred by the fitz renderer's anti-aliasing. 20px white padding prevents
    the recogniser from clipping the first/last character.
    """
    import cv2  # noqa: PLC0415
    gray = canvas_gray.copy()
    h, w = gray.shape
    if h < 64:
        scale_f = 64.0 / h
        gray = cv2.resize(gray, (int(w * scale_f), 64), interpolation=cv2.INTER_CUBIC)
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 15, 4,
    )
    binary = cv2.medianBlur(binary, 3)
    return cv2.copyMakeBorder(binary, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)


def _ocr_canvas_multiangle(
    canvas: np.ndarray,
    engine: Any,
    extra_angles: tuple[float, ...] = (45.0, 90.0, 135.0, 180.0, 270.0),
) -> tuple[str, float] | None:
    """OCR with additional cv2 rotations; keep best measurement-like result.

    The canvas is already de-rotated to what the render pass believes is
    horizontal.  If that was wrong, rotating by 45 / 90 / 135° corrects it.
    Only accepts tokens matching the FMB measurement pattern (d{1,4}[.,]d{1,2}).
    """
    import cv2  # noqa: PLC0415
    h, w = canvas.shape[:2]
    best_text, best_conf = None, 0.0

    for angle in (0.0,) + extra_angles:
        if angle == 0.0:
            rotated = canvas
        else:
            cx_r, cy_r = w / 2.0, h / 2.0
            M = cv2.getRotationMatrix2D((cx_r, cy_r), angle, 1.0)
            cos_a, sin_a = abs(M[0, 0]), abs(M[0, 1])
            nw = int(h * sin_a + w * cos_a) + 20
            nh = int(h * cos_a + w * sin_a) + 20
            M[0, 2] += (nw - w) / 2.0
            M[1, 2] += (nh - h) / 2.0
            rotated = cv2.warpAffine(
                canvas, M, (nw, nh), flags=cv2.INTER_CUBIC, borderValue=255
            )

        # Use _ocr_canvas_all so that measurement tokens spanning multiple
        # detection boxes (e.g. "7.2" split into "7" + ".2" by the det model)
        # are still considered — we only keep measurement-pattern matches.
        all_results = _ocr_canvas_all(rotated, engine)
        for text, conf, _pos in all_results:
            if _MEAS_RE.match(text) and conf > best_conf:
                best_text, best_conf = text, conf

    return (best_text, best_conf) if best_text else None


def _expand_glyph_groups(groups: list[Any]) -> list[Any]:
    """Gap-split glyph groups then replace overlapping multi-fill groups.

    Two stages:
    1. ``split_glyph_group_by_gap_2d`` separates clusters whose fills have a
       measurable intra-cluster gap (adjacent measurement labels proximity-
       clustered by pdf_glyphs).
    2. Overlapping replacement: when a group's fills have intersecting bounding
       boxes (two distinct labels at the same corner position, zero inter-fill
       gap), each fill is treated as an independent measurement.  The combined
       group is *removed* and replaced with per-fill sub-groups at angle 0°.
       This prevents the combined render from producing a misread (e.g. "29.8"
       instead of "59.8" + "20.4") and stops the combined detection from
       deduplicating the correct sub-group detections.
    """
    from .pdf_glyphs import (  # noqa: PLC0415
        GlyphGroup as _GlyphGroup,
        split_glyph_group_by_gap_2d,
    )

    # Stage 1 — gap splitting
    gap_split: list[Any] = []
    for g in groups:
        gap_split.extend(split_glyph_group_by_gap_2d(g))

    # Stage 2 — overlapping fill replacement
    final: list[Any] = []
    for g in gap_split:
        if len(g.drawings) < 2:
            final.append(g)
            continue
        rects = [d["rect"] for d in g.drawings]
        overlapping = any(
            rects[i].intersects(rects[j])
            for i in range(len(rects))
            for j in range(i + 1, len(rects))
        )
        if not overlapping:
            final.append(g)
            continue
        for d in g.drawings:
            r = d["rect"]
            final.append(_GlyphGroup(
                drawings=[d],
                bbox=fitz.Rect(r.x0, r.y0, r.x1, r.y1),
                center=((r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2),
                angle_deg=0.0,
                color=g.color,
                kind=g.kind,
            ))
    return final


def _extract_glyph_detections(
    pdf_path: Path,
    page_number: int,
    engine: Any,
    *,
    debug_dir: Path | None = None,
) -> list[OCRDetection]:
    """Per-element OCR: one PaddleOCR call per isolated glyph group.

    Strategy (in order):
    1. Render with the NEAREST VECTOR LINE's angle (more reliable than the
       glyph-shape-derived angle for small/sparse clusters on diagonal lines).
    2. Try both flip orientations as before.
    3. If best confidence < _MULTIANGLE_CONF_THRESHOLD, try 3 extra cv2 rotations
       and keep any measurement-pattern match with higher confidence.
    4. Apply preprocessing (scale to 64px, adaptive threshold, pad) before every
       OCR call to sharpen glyph edges that fitz anti-aliasing softens.

    debug_dir: when set, each processed canvas is saved as a PNG with angle /
    OCR result encoded in the filename — useful for diagnosing remaining gaps.
    """
    from .pdf_glyphs import (  # noqa: PLC0415
        canvas_to_pdf as _canvas_to_pdf,
        extract_glyph_groups,
        render_glyph_group,
    )
    from .pdf_vectors import _classify_glyph_groups, extract_vectors  # noqa: PLC0415

    detections: list[OCRDetection] = []
    all_segments: list[Any] = []

    try:
        with fitz.open(pdf_path) as doc:
            page = doc[page_number]
            pw = page.rect.width
            ph = page.rect.height
            groups = extract_glyph_groups(page)
            _classify_glyph_groups(groups, pw, ph)
    except Exception:  # noqa: BLE001
        return detections

    # Collect all vector line segments for line-guided angle lookup.
    try:
        vecs = extract_vectors(pdf_path)
        all_segments = list(vecs.boundary) + list(vecs.internal) + list(vecs.chain)
    except Exception:  # noqa: BLE001
        pass  # fall back to glyph-shape angles

    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)

    by_color: dict[str, int] = {"blue": 0, "red": 0, "black": 0}
    for g in groups:
        by_color[g.color] = by_color.get(g.color, 0) + 1
    _log.info(
        "glyph OCR [%s p%d]: %d groups (blue=%d red=%d black=%d)",
        pdf_path.name, page_number, len(groups),
        by_color.get("blue", 0), by_color.get("red", 0), by_color.get("black", 0),
    )

    # Expand: gap-split then replace overlapping-fill groups with per-fill
    # sub-groups.  See _expand_glyph_groups for full rationale.
    expanded_groups: list[Any] = _expand_glyph_groups(groups)

    for idx, group in enumerate(expanded_groups):
        try:
            # --- angle: prefer nearest line, fall back to glyph shape ----------
            nearest_seg = _find_nearest_line(group.center, all_segments)
            line_angle: float | None = (
                _seg_angle_deg(nearest_seg) if nearest_seg is not None else None
            )
            angle_for_render = line_angle  # None → render_glyph_group uses group.angle_deg

            # --- render both flips with line-guided (or shape) angle ----------
            img1 = render_glyph_group(group, flip=False, angle_override_deg=angle_for_render)
            img2 = render_glyph_group(group, flip=True,  angle_override_deg=angle_for_render)
            pre1 = _preprocess_canvas(img1)
            pre2 = _preprocess_canvas(img2)

            all1 = _ocr_canvas_all(pre1, engine)
            all2 = _ocr_canvas_all(pre2, engine)

            # Merge: keep highest-confidence reading for each unique text.
            merged: dict[str, tuple[str, float, tuple[float, float], bool]] = {}
            for text, conf, cpos in all1:
                if conf < _MIN_GLYPH_CONF:
                    continue
                if text not in merged or conf > merged[text][1]:
                    merged[text] = (text, conf, cpos, False)
            for text, conf, cpos in all2:
                if conf < _MIN_GLYPH_CONF:
                    continue
                if text not in merged or conf > merged[text][1]:
                    merged[text] = (text, conf, cpos, True)

            # --- multi-angle fallback for low/no confidence ------------------
            best_conf_so_far = max((v[1] for v in merged.values()), default=0.0)
            best_text_so_far = (
                max(merged.values(), key=lambda v: v[1])[0] if merged else ""
            )
            # Also retry when the primary read doesn't look like a measurement
            # (e.g. a single digit "2" from a "7.2" cluster rendered at wrong angle).
            _no_meas_match = merged and not _MEAS_RE.match(best_text_so_far)
            if best_conf_so_far < _MULTIANGLE_CONF_THRESHOLD or _no_meas_match:
                # Try extra cv2 rotations on the better of the two canvases.
                ma_canvas = pre1 if len(all1) >= len(all2) else pre2
                ma_result = _ocr_canvas_multiangle(ma_canvas, engine)
                if ma_result:
                    ma_text, ma_conf = ma_result
                    if ma_conf >= _MIN_GLYPH_CONF and (
                        ma_text not in merged or ma_conf > merged[ma_text][1]
                    ):
                        # Position: centre of the canvas (single-text result)
                        from .pdf_glyphs import CANVAS_W, CANVAS_H  # noqa: PLC0415
                        cpos = (CANVAS_W / 2.0, CANVAS_H / 2.0)
                        merged[ma_text] = (ma_text, ma_conf, cpos, False)

            # --- perpendicular-render fallback --------------------------------
            # Some labels are printed perpendicular to their annotation line
            # (e.g. "7.2" at ~90° on a nearly-horizontal 10° line). If the
            # primary read is still not measurement-like, try re-renders at
            # +90°, +180°, and +270° relative to the line angle so that any
            # rotation of the text baseline is covered.
            best_text_so_far = (
                max(merged.values(), key=lambda v: v[1])[0] if merged else ""
            )
            if merged and not _MEAS_RE.match(best_text_so_far) and angle_for_render is not None:
                for _perp_offset in (90.0, 180.0, 270.0):
                    if _MEAS_RE.match(
                        max(merged.values(), key=lambda v: v[1])[0]
                    ):
                        break  # already found a measurement, stop
                    perp_angle = angle_for_render + _perp_offset
                    imgp = render_glyph_group(
                        group, flip=False, angle_override_deg=perp_angle
                    )
                    prep = _preprocess_canvas(imgp)
                    allp = _ocr_canvas_all(prep, engine)
                    for text, conf, cpos in allp:
                        if conf >= _MIN_GLYPH_CONF and _MEAS_RE.match(text):
                            if text not in merged or conf > merged[text][1]:
                                merged[text] = (text, conf, cpos, False)

            if not merged:
                if debug_dir is not None:
                    _save_debug(debug_dir, idx, group, line_angle, pre1, "NONE", 0.0)
                continue

            kind_tag = group.kind if group.kind != "survey_number" else ""

            for text, conf, (cx_c, cy_c), used_flip in merged.values():
                px, py = _canvas_to_pdf(
                    group, cx_c, cy_c,
                    flip=used_flip,
                    angle_override_deg=angle_for_render,
                )
                hw, hh = group.bbox.width / 2, group.bbox.height / 2
                effective_angle = (
                    angle_for_render if angle_for_render is not None else group.angle_deg
                )
                ar = math.radians(effective_angle)
                ca, sa = math.cos(ar), math.sin(ar)

                def _rot(
                    ldx: float, ldy: float,
                    _ca: float = ca, _sa: float = sa,
                    _px: float = px, _py: float = py,
                ) -> tuple[float, float]:
                    return (_px + _ca * ldx - _sa * ldy, _py + _sa * ldx + _ca * ldy)

                poly = (_rot(-hw, -hh), _rot(hw, -hh), _rot(hw, hh), _rot(-hw, hh))
                # Chain dims have parens rendered as thin bezier strokes OCR strips.
                # Detect by fill width per char: a plain decimal reading N chars from
                # a fill >5 pt/char wide is a chain dim with stripped parens.
                emit_text = text
                if (
                    re.match(r"^\d{1,4}[.,]\d{1,2}$", text)
                    and len(text) > 0
                    and group.bbox.width / len(text) > 5.0
                ):
                    emit_text = f"({text})"
                detections.append(
                    OCRDetection(text=emit_text, confidence=conf, polygon=poly, kind=kind_tag)
                )

            if debug_dir is not None:
                best = max(merged.values(), key=lambda v: v[1])
                _save_debug(debug_dir, idx, group, line_angle, pre1, best[0], best[1])

        except Exception:  # noqa: BLE001
            continue

    _log.info(
        "glyph OCR [%s p%d]: %d/%d groups read (text extracted)",
        pdf_path.name, page_number, len(detections), len(groups),
    )

    # Inject a sentinel detection marking the largest blue glyph's centre.
    # The survey number (e.g. "100") is drawn as the biggest blue glyph at the
    # plot centroid.  build_plot reads this sentinel to place the SURVEY NUMBER
    # text at the drawn position rather than the geometric centroid.
    try:
        from .pdf_glyphs import largest_blue_glyph, glyph_polygon as _glyph_poly  # noqa: PLC0415
        sg = largest_blue_glyph(groups)
        if sg is not None:
            poly = _glyph_poly(sg)
            detections.append(
                OCRDetection(
                    text="__survey_glyph__",
                    confidence=1.0,
                    polygon=poly,
                    kind="survey_number_glyph",
                )
            )
            _log.info(
                "glyph OCR: survey-number glyph sentinel injected at (%.1f, %.1f)",
                sg.center[0], sg.center[1],
            )
    except Exception as _exc:  # noqa: BLE001
        _log.debug("survey glyph sentinel skipped: %s", _exc)

    return detections


def _save_debug(
    debug_dir: Path,
    idx: int,
    group: Any,
    line_angle: float | None,
    canvas: np.ndarray,
    text: str,
    conf: float,
) -> None:
    """Save a preprocessed glyph canvas as PNG for inspection."""
    try:
        import cv2  # noqa: PLC0415
        la_str = f"{line_angle:.0f}" if line_angle is not None else "NA"
        safe_text = re.sub(r"[^\w.,()-]", "_", text)
        name = (
            f"{idx:03d}_g{group.angle_deg:.0f}_l{la_str}"
            f"_{group.kind}_{safe_text}_c{conf:.2f}.png"
        )
        cv2.imwrite(str(debug_dir / name), canvas)
    except Exception:  # noqa: BLE001
        pass


def _merge_glyph_detections(
    page_dets: list[OCRDetection],
    glyph_dets: list[OCRDetection],
) -> list[OCRDetection]:
    """Append glyph detections that are NOT already covered by page-level OCR.

    A glyph detection is 'covered' when its centre is within _GLYPH_DEDUP_RADIUS
    of any page-level detection's centre — meaning the page-level pass already
    found that measurement. Only genuinely new positions (missed by page-level
    OCR) are added, so no duplicate measurements appear in the result.
    """
    if not glyph_dets:
        return page_dets

    page_centers = [d.center for d in page_dets]
    result = list(page_dets)

    for gd in glyph_dets:
        gx, gy = gd.center
        covered = any(
            math.hypot(gx - px, gy - py) < _GLYPH_DEDUP_RADIUS
            for px, py in page_centers
        )
        if not covered:
            result.append(gd)
            page_centers.append(gd.center)  # prevent future duplicates against itself

    return result


# --- PaddleOCR-VL-1.6 engine (HSV color isolation + recursive token harvest) -

# Regex patterns from the client's ocr_paddle_vl.py.
# DECIMAL_RE: matches "44.2", "105.6", "74·0" (dot or middle-dot decimal).
# INTEGER_RE: matches standalone integers — lookbehind/ahead exclude sub-parts
#             of decimals (e.g. "44" inside "44.2" is NOT matched).
_DECIMAL_RE = re.compile(r"(?<!\d)\d{1,4}\s*[.·]\s*\d{1,2}(?!\d)")
_INTEGER_RE = re.compile(r"(?<![\d.])\d{1,4}(?![\d.])")

# HSV isolation ranges (client's fmb.yaml + preprocess.py).
# Blue pixels = measurement numbers (boundary/chain/subdivision distances).
_BLUE_HSV_LOW: list[int] = [95, 60, 60]
_BLUE_HSV_HIGH: list[int] = [135, 255, 255]
# Red pixels = corner stone labels (A–H) and chain point numbers.
# Red hue wraps around 0°/360° → two ranges OR-combined.
_RED_HSV_RANGES: list[tuple[list[int], list[int]]] = [
    ([0, 30, 20], [12, 255, 255]),
    ([165, 30, 20], [179, 255, 255]),
]

_paddlevl_cache: dict[str, list[OCRDetection]] = {}


@lru_cache(maxsize=1)
def _build_paddlevl_engine() -> Any:
    """Construct and cache the PaddleOCRVL-1.6 pipeline.

    ``use_layout_detection=False`` matches the client's fmb.yaml and gives
    per-word spotting polygons directly rather than block-level bboxes.
    ``use_ocr_for_image_block=True`` ensures image regions within the page
    are also OCR'd (rasterized dimension glyphs in FMB PDFs are image blocks).

    bfloat16 guard: Paddle 3.1.0 CPU's tensor.set() rejects numpy bfloat16
    arrays even though ml_dtypes is installed. The model weights are pre-
    converted to float32 by convert_vl_bf16_to_fp32.py.  We also set Paddle's
    default dtype to float32 here as a belt-and-suspenders guard.
    """
    # Import paddleocr (and its torch dependency) BEFORE paddle to avoid a
    # Windows DLL load-order conflict: when paddle loads its native DLLs first,
    # torch's shm.dll subsequently fails with WinError 127.  Importing PaddleOCRVL
    # first establishes the correct torch→paddle DLL ordering.
    from paddleocr import PaddleOCRVL  # noqa: PLC0415

    # Register bfloat16 with numpy dtype registry (ml_dtypes) and tell Paddle
    # to default-allocate new tensors in float32 rather than bfloat16.
    try:
        import ml_dtypes  # noqa: F401, PLC0415
    except ImportError:
        pass
    try:
        import paddle as _paddle  # noqa: PLC0415
        _paddle.set_default_dtype("float32")
    except Exception:  # noqa: BLE001
        pass

    return PaddleOCRVL(
        pipeline_version="v1.6",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_layout_detection=False,
        use_chart_recognition=False,
        use_seal_recognition=False,
        use_ocr_for_image_block=True,
        device="cpu",
    )


def _render_page_bgr(pdf_path: Path, page_number: int, zoom: float) -> np.ndarray:
    """Render one PDF page to a BGR numpy array (input for HSV isolation)."""
    import cv2  # noqa: PLC0415

    png_bytes = _render_page_png(pdf_path, page_number, zoom)
    arr = np.frombuffer(png_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR H×W×3


def _hsv_isolate(bgr: np.ndarray, low: list[int], high: list[int]) -> np.ndarray:
    """Return white-background BGR image with only pixels in the HSV range kept."""
    import cv2  # noqa: PLC0415

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(low, np.uint8), np.array(high, np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
    out = np.full_like(bgr, 255)
    out[mask > 0] = bgr[mask > 0]
    return out


def _hsv_isolate_multi(
    bgr: np.ndarray, ranges: list[tuple[list[int], list[int]]]
) -> np.ndarray:
    """HSV isolation with multiple ranges OR-combined (handles red hue wrap)."""
    import cv2  # noqa: PLC0415

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    combined = np.zeros(bgr.shape[:2], dtype=np.uint8)
    for low, high in ranges:
        combined |= cv2.inRange(hsv, np.array(low, np.uint8), np.array(high, np.uint8))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
    out = np.full_like(bgr, 255)
    out[combined > 0] = bgr[combined > 0]
    return out


def _bgr_to_png_bytes(bgr: np.ndarray) -> bytes:
    import cv2  # noqa: PLC0415

    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return buf.tobytes()


def _poly_angle_deg(raw: Any) -> float | None:
    """Undirected orientation [0, 180°) of the longest polygon edge.

    Uses ``atan2(dy, dx) % 180`` — the same convention as
    ``anchor._orientation`` and ``pdf_glyphs._compute_angle`` — so the value
    can be passed directly to ``anchor._angle_diff`` without conversion.
    Both traversal directions of the same undirected line give the same result:
    atan2(-dy, -dx) % 180 == atan2(dy, dx) % 180 for any (dx, dy).
    """
    try:
        pts = np.asarray(raw, dtype=float)
        if pts.ndim == 1 and pts.size == 4:
            return 0.0
        if pts.shape[0] < 2:
            return None
        edges = np.roll(pts, -1, axis=0) - pts
        lengths = np.linalg.norm(edges, axis=1)
        edge = edges[int(np.argmax(lengths))]
        return float(math.degrees(math.atan2(edge[1], edge[0]))) % 180.0
    except Exception:  # noqa: BLE001
        return None


def _poly_to_pdf(raw: Any, zoom: float) -> tuple[Point, ...]:
    """Convert a raw pixel polygon or bbox to PDF page-space tuple[Point]."""
    _fallback: tuple[Point, ...] = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
    if raw is None:
        return _fallback
    try:
        pts = np.asarray(raw, dtype=float)
        if pts.ndim == 1 and pts.size == 4:
            x0, y0, x1, y1 = pts / zoom
            return ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
        if pts.ndim == 2 and pts.shape[1] >= 2:
            return tuple((float(p[0]) / zoom, float(p[1]) / zoom) for p in pts)
    except Exception:  # noqa: BLE001
        pass
    return _fallback


def _to_builtin_vl(obj: Any) -> Any:
    """Recursively convert PaddleOCRVL result objects to plain Python types."""
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _to_builtin_vl(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin_vl(v) for v in obj]
    if hasattr(obj, "json"):
        try:
            return json.loads(obj.json())
        except Exception:  # noqa: BLE001
            pass
    if hasattr(obj, "to_dict"):
        try:
            return _to_builtin_vl(obj.to_dict())
        except Exception:  # noqa: BLE001
            pass
    if hasattr(obj, "__dict__"):
        return _to_builtin_vl(vars(obj))
    return repr(obj)


def _records_vl(obj: Any) -> list[dict[str, Any]]:
    """Recursively harvest all {text, score, poly} records from a VL result tree.

    PaddleOCRVL result structure varies across versions and settings. Walking
    the entire tree and collecting everything with a text field is the client's
    robust approach (ocr_paddle_vl.py ``_records()``).
    """
    records: list[dict[str, Any]] = []
    if isinstance(obj, dict):
        text = obj.get("text") or obj.get("rec_text") or obj.get("content")
        poly = (
            obj.get("rec_poly") or obj.get("poly") or obj.get("box")
            or obj.get("dt_poly") or obj.get("bbox")
        )
        score = obj.get("score") or obj.get("rec_score")
        if isinstance(text, str) and text.strip():
            records.append({"text": text.strip(), "score": score, "poly": poly})
        # Parallel list fields: rec_texts / rec_scores / rec_polys
        for key in ("rec_texts", "texts"):
            if isinstance(obj.get(key), list):
                texts_list = obj[key]
                scores_list = obj.get("rec_scores", [])
                polys_list = (
                    obj.get("rec_polys") or obj.get("dt_polys")
                    or obj.get("polys") or []
                )
                for i, t in enumerate(texts_list):
                    if not isinstance(t, str) or not t.strip():
                        continue
                    records.append({
                        "text": t.strip(),
                        "score": scores_list[i] if i < len(scores_list) else None,
                        "poly": polys_list[i] if i < len(polys_list) else None,
                    })
        for value in obj.values():
            if isinstance(value, (dict, list)):
                records.extend(_records_vl(value))
    elif isinstance(obj, list):
        for item in obj:
            records.extend(_records_vl(item))
    return records


def _dedup_detections(dets: list[OCRDetection]) -> list[OCRDetection]:
    """Remove duplicate and fragmented tokens from a VL OCR detection list.

    Two passes:
    1. Exact text match within 10.0 pt → keep higher confidence (handles "28 28",
       "31 31" — same number detected twice from slightly offset boxes).
    2. One text is a prefix of the other AND within 6.0 pt → keep the longer/
       more-confident one (handles "2020"/"21", "108"/"100" merges where one
       detection bleeds into an adjacent number).

    Returns a new list; does not mutate the input.
    """
    if len(dets) <= 1:
        return list(dets)

    to_remove: set[int] = set()
    n = len(dets)

    # Pass 1: exact duplicate (same normalised text, close center)
    for i in range(n):
        if i in to_remove:
            continue
        ti = dets[i].text.strip().lower()
        ci = dets[i].center
        for j in range(i + 1, n):
            if j in to_remove:
                continue
            tj = dets[j].text.strip().lower()
            cj = dets[j].center
            if ti == tj and math.hypot(ci[0] - cj[0], ci[1] - cj[1]) < 10.0:
                if dets[i].confidence >= dets[j].confidence:
                    to_remove.add(j)
                else:
                    to_remove.add(i)
                    break

    # Pass 2: one is a digit-prefix of the other AND very close → keep longer
    survivors = [d for i, d in enumerate(dets) if i not in to_remove]
    to_remove2: set[int] = set()
    m = len(survivors)
    for i in range(m):
        if i in to_remove2:
            continue
        ti = survivors[i].text.strip()
        ci = survivors[i].center
        for j in range(i + 1, m):
            if j in to_remove2:
                continue
            tj = survivors[j].text.strip()
            cj = survivors[j].center
            if math.hypot(ci[0] - cj[0], ci[1] - cj[1]) >= 6.0:
                continue
            # Check digit-prefix overlap
            if ti.isdigit() and tj.isdigit():
                if ti in tj or tj in ti:
                    # Keep longer; if equal length keep higher confidence
                    if len(ti) >= len(tj):
                        to_remove2.add(j)
                    else:
                        to_remove2.add(i)
                        break

    return [d for i, d in enumerate(survivors) if i not in to_remove2]


def _run_vl_pass(
    bgr_isolated: np.ndarray,
    engine: Any,
    zoom: float,
    pass_kind: str,
    min_confidence: float = 0.30,
) -> list[OCRDetection]:
    """OCR one HSV-isolated BGR image and return typed OCRDetection tokens.

    ``pass_kind="measurement"`` (blue image): extracts decimal measurements
    (DECIMAL_RE) and integer markers (INTEGER_RE).

    ``pass_kind="label"`` (red image): collects every word token as-is;
    corner labels A-H and chain points 11-19 are tagged by kind so the anchor
    step's ``_NON_MEASUREMENT_RE`` can route them to ``unanchored_measurements``
    without a false-positive anchor.
    """
    import tempfile  # noqa: PLC0415

    png_bytes = _bgr_to_png_bytes(bgr_isolated)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp.write(png_bytes)
        tmp_path = tmp.name
    try:
        raw_result = list(engine.predict(tmp_path))
    except Exception as exc:  # noqa: BLE001
        _log.warning("PaddleOCR-VL predict failed (%s pass): %s", pass_kind, exc)
        return []
    finally:
        os.unlink(tmp_path)

    records = _records_vl(_to_builtin_vl(raw_result))
    detections: list[OCRDetection] = []
    seen: list[tuple[float, float]] = []  # centers of already-emitted tokens

    for rec in records:
        score = rec.get("score")
        if score is not None:
            try:
                if float(score) < min_confidence:
                    continue
            except (TypeError, ValueError):
                pass

        text = rec.get("text", "")
        poly_raw = rec.get("poly")
        polygon = _poly_to_pdf(poly_raw, zoom)
        conf = float(score) if score is not None else 1.0
        angle = _poly_angle_deg(poly_raw)
        cx = sum(p[0] for p in polygon) / len(polygon)
        cy = sum(p[1] for p in polygon) / len(polygon)

        def _emit(tok: str, kind_tag: str) -> None:
            if any(math.hypot(cx - sx, cy - sy) < 4.0 for sx, sy in seen):
                return
            seen.append((cx, cy))
            detections.append(
                OCRDetection(text=tok, confidence=conf, polygon=polygon,
                             angle_deg=angle, kind=kind_tag)
            )

        if pass_kind == "measurement":
            for m in _DECIMAL_RE.finditer(text):
                _emit(re.sub(r"\s+", "", m.group()).replace("·", "."),
                      "decimal_measurement")
            for m in _INTEGER_RE.finditer(text):
                _emit(m.group(), "integer_marker")
        else:  # label pass
            for word in text.split():
                word = word.strip()
                if not word:
                    continue
                if re.match(r"^[A-H]$", word):
                    kind_tag = "corner_label"
                elif re.match(r"^1[1-9]$", word):
                    kind_tag = "chain_point"
                else:
                    kind_tag = "red_token"
                _emit(word, kind_tag)

    return _dedup_detections(detections)


def _extract_paddlevl(
    pdf_path: Path, page_number: int, zoom: float
) -> list[OCRDetection]:
    """OCR one FMB page via PaddleOCR-VL-1.6 with HSV color isolation.

    Key insight from the client's pipeline (preprocess.py + ocr_paddle_vl.py):
    running OCR on the raw full-color page hallucinated noise because the model
    sees black boundary lines, colored glyphs, and text all at once. Isolating
    by color channel before OCR gives PaddleOCR-VL a clean white-background
    image containing only the target color, which dramatically improves recall
    on rotated measurement labels.

    Pass 1 — blue-isolated image:
        FMB measurement numbers (boundary/chain/subdivision distances) are
        rendered as blue filled vector glyphs.  DECIMAL_RE + INTEGER_RE extract
        typed tokens from the VL result.

    Pass 2 — red-isolated image:
        Corner stone labels (A–H) and chain point numbers are rendered in red.
        All word tokens are collected; the anchor step's _NON_MEASUREMENT_RE
        routes them to unanchored_measurements (correct behavior).

    Our vector line geometry (pdf_vectors.py get_drawings()) is kept unchanged.
    """
    png_bytes = _render_page_png(pdf_path, page_number, zoom)
    cache_key = hashlib.sha256(png_bytes).hexdigest()
    if cache_key in _paddlevl_cache:
        return _paddlevl_cache[cache_key]

    engine = _build_paddlevl_engine()
    bgr = _render_page_bgr(pdf_path, page_number, zoom)

    # Pass 0 — header band via PP-OCRv5.
    # The header block (District, Taluk, Village, Scale, Area) is black text on
    # white — it falls into neither the blue nor the red HSV range. Without this
    # pass, parse_header() gets no detections and scale_denominator stays None,
    # which causes build_plot to raise GeometryError on every paddlevl run.
    # PP-OCRv5 reads horizontal black text reliably and produces multi-word tokens
    # ("Scale : 1 : 2021") that parse_header's regexes require intact.
    header_px = int(_HEADER_LIMIT_Y * zoom)
    header_crop = bgr[:header_px, :]
    header_rgb = header_crop[:, :, ::-1].copy()  # BGR → RGB for PaddleOCR
    header_dets: list[OCRDetection] = []
    try:
        paddle_engine = _body_ocr_engine(False)
        raw_header = paddle_engine.predict(header_rgb)
        header_dets = [
            d for d in _parse_result(raw_header, zoom)
            if d.center[1] < _HEADER_LIMIT_Y
        ]
    except Exception as exc:  # noqa: BLE001
        _log.warning("VL header pass (PP-OCRv5) failed (%s); header fields may be missing", exc)
    _log.info(
        "VL header pass: %d detections on %s page %d",
        len(header_dets), pdf_path.name, page_number,
    )

    blue_bgr = _hsv_isolate(bgr, _BLUE_HSV_LOW, _BLUE_HSV_HIGH)
    blue_dets = _run_vl_pass(blue_bgr, engine, zoom, "measurement")
    _log.info(
        "VL blue pass: %d measurement tokens on %s page %d",
        len(blue_dets), pdf_path.name, page_number,
    )

    red_bgr = _hsv_isolate_multi(bgr, _RED_HSV_RANGES)
    red_dets = _run_vl_pass(red_bgr, engine, zoom, "label")
    _log.info(
        "VL red pass: %d label tokens on %s page %d",
        len(red_dets), pdf_path.name, page_number,
    )

    # Merge passes; suppress red tokens that coincide with a blue detection.
    blue_centers = [d.center for d in blue_dets]
    combined = list(blue_dets)
    for det in red_dets:
        cx, cy = det.center
        if not any(math.hypot(cx - bx, cy - by) < 4.0 for bx, by in blue_centers):
            combined.append(det)

    # Cross-pass dedup on the merged body (Bug 2: remove "28 28", "31 31", etc.)
    combined = _dedup_detections(combined)

    # Header detections prepended so parse_header() finds them first.
    result = header_dets + combined
    _log.info(
        "VL total: %d detections on %s page %d (header=%d blue=%d red=%d)",
        len(result), pdf_path.name, page_number,
        len(header_dets), len(blue_dets), len(red_dets),
    )
    _paddlevl_cache[cache_key] = result
    return result


# --- HuggingFace VL extraction path -----------------------------------------

def _extract_hf_vl(
    pdf_path: Path,
    page_number: int,
    zoom: float,
    det_model: str,
    rec_model: str,
) -> list[OCRDetection]:
    """OCR via HuggingFace Qwen2.5-VL on GPU + vector glyph positions.

    Architecture:
    * Header band: PP-OCRv5 server_det (exact same as paddle path — multi-word
      phrases like "Scale : 1 : 2021" require this).
    * Body glyphs: vector glyph extraction for POSITIONS (pdf_glyphs.py) with
      Qwen2.5-VL for per-canvas TEXT (replaces PP-OCRv5's ~24% recall path).
      Each glyph group is rendered to a clean 520x150 de-rotated canvas and
      VL is asked "what number is this?" — ideal input for a VLM.
    """
    # --- Header band: PP-OCRv5 -----------------------------------------------
    image = _render_page(pdf_path, page_number, zoom)
    header_engine = _build_engine(det_model, rec_model, False)
    try:
        raw = header_engine.predict(image)
        header_dets = [
            d for d in _parse_result(raw, zoom)
            if d.center[1] < _HEADER_LIMIT_Y
        ]
    except Exception as exc:  # noqa: BLE001
        _log.warning("HF VL header pass failed (%s); header fields will be missing", exc)
        header_dets = []
    _log.info("HF VL header pass: %d detections on %s p%d", len(header_dets), pdf_path.name, page_number)

    # --- Body: glyph positions + HF VL text ----------------------------------
    glyph_dets = _extract_glyph_detections_hf_vl(pdf_path, page_number)
    body_glyph_dets = [d for d in glyph_dets if d.center[1] >= _HEADER_LIMIT_Y]
    _log.info(
        "HF VL body: %d glyph detections (%d in body) on %s p%d",
        len(glyph_dets), len(body_glyph_dets), pdf_path.name, page_number,
    )

    result = header_dets + body_glyph_dets
    _log.info(
        "HF VL total: %d detections (header=%d body=%d)",
        len(result), len(header_dets), len(body_glyph_dets),
    )
    return result


# --- Google Cloud Vision engine ----------------------------------------------

# In-memory cache keyed by SHA-256 of the rendered PNG bytes.  Avoids re-OCRing
# the same page when the worker retries a job or runs tests against fixed PDFs.
_vision_cache: dict[str, list[OCRDetection]] = {}

# Per-canvas cache keyed by MD5 of the canvas bytes.  Each glyph canvas is
# small and deterministic — if the same glyph appears in two surveys, skip the
# API round-trip.  None = Vision returned no text for this canvas.
_vision_canvas_cache: dict[str, tuple[str, float] | None] = {}


def _get_vision_credentials() -> Any:
    """Build Google Vision credentials from GOOGLE_CREDENTIALS_JSON env var.

    Uses ``from_service_account_info()`` directly — no temp file, no filesystem
    permissions issues.  ``json.loads()`` handles the ``\\n`` escaping inside the
    private_key field automatically, so the minified single-line JSON pasted into
    the Render dashboard env var field works as-is.

    Returns a ``google.oauth2.service_account.Credentials`` object, or ``None``
    when no credentials are configured (caller falls back to ADC).

    Raises:
        OCRFailure: If ``GOOGLE_CREDENTIALS_JSON`` is set but is not valid JSON.
    """
    creds_json = ""
    try:
        from ...config import get_settings

        creds_json = get_settings().google_credentials_json
    except Exception:  # noqa: BLE001
        pass
    if not creds_json:
        creds_json = os.getenv("GOOGLE_CREDENTIALS_JSON", "")
    if not creds_json:
        return None  # no credentials configured; caller uses ADC or fails

    try:
        creds_dict = json.loads(creds_json)
    except json.JSONDecodeError as exc:
        raise OCRFailure(
            "GOOGLE_CREDENTIALS_JSON is not valid JSON. "
            "Paste the full service-account JSON as one value in Render dashboard "
            "(use ConvertTo-Json -Compress to get a single line).",
        ) from exc

    from google.oauth2 import service_account  # noqa: PLC0415

    return service_account.Credentials.from_service_account_info(
        creds_dict,
        scopes=["https://www.googleapis.com/auth/cloud-vision"],
    )


def _render_page_png(pdf_path: Path, page_number: int, zoom: float) -> bytes:
    """Render one PDF page to PNG bytes for the Vision API."""
    with fitz.open(pdf_path) as doc:
        page = doc[page_number]
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        return pix.tobytes("png")


def _ocr_canvas_vision(canvas: np.ndarray) -> tuple[str, float] | None:
    """Call Google Cloud Vision on a single de-rotated glyph canvas.

    Auth (tried in order):
    1. GOOGLE_CREDENTIALS_JSON env var → service-account SDK path.
    2. GOOGLE_API_KEY env var → REST endpoint (simpler, no JSON file needed).

    Results cached by MD5 of canvas bytes — identical glyphs skip the call.
    Returns (raw_text, 0.90) or None on auth failure / empty result.
    """
    import cv2  # noqa: PLC0415

    canvas_bgr = (
        cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
        if canvas.ndim == 2
        else canvas
    )
    img_hash = hashlib.md5(canvas_bgr.tobytes()).hexdigest()
    if img_hash in _vision_canvas_cache:
        return _vision_canvas_cache[img_hash]

    ok, buf = cv2.imencode(".png", canvas_bgr)
    if not ok:
        _vision_canvas_cache[img_hash] = None
        return None
    png_bytes = buf.tobytes()

    def _set(result: tuple[str, float] | None) -> tuple[str, float] | None:
        _vision_canvas_cache[img_hash] = result
        return result

    # ── Path 1: service-account SDK ──────────────────────────────────────────
    try:
        creds = _get_vision_credentials()
        if creds is not None:
            from google.cloud import vision  # noqa: PLC0415
            client = vision.ImageAnnotatorClient(credentials=creds)
            resp = client.text_detection(image=vision.Image(content=png_bytes))
            anns = resp.text_annotations
            if not anns:
                return _set(None)
            text = anns[0].description.strip().replace("\n", " ").strip()
            return _set((text, 0.90) if text else None)
    except Exception as exc:  # noqa: BLE001
        _log.debug("Vision SDK canvas call failed: %s", exc)

    # ── Path 2: API key REST ──────────────────────────────────────────────────
    import base64  # noqa: PLC0415

    import requests  # noqa: PLC0415

    api_key = os.getenv("GOOGLE_API_KEY", "")
    if not api_key:
        return _set(None)

    img_b64 = base64.b64encode(png_bytes).decode()
    url = f"https://vision.googleapis.com/v1/images:annotate?key={api_key}"
    payload = {
        "requests": [{
            "image": {"content": img_b64},
            "features": [{"type": "TEXT_DETECTION", "maxResults": 5}],
            "imageContext": {"languageHints": ["en"]},
        }]
    }
    try:
        r = requests.post(url, json=payload, timeout=10)
        r.raise_for_status()
        anns = r.json()["responses"][0].get("textAnnotations", [])
        if not anns:
            return _set(None)
        text = anns[0]["description"].strip().replace("\n", " ").strip()
        return _set((text, 0.90) if text else None)
    except Exception as exc:  # noqa: BLE001
        _log.debug("Vision REST canvas call failed: %s", exc)
        return _set(None)


def _extract_vision(
    pdf_path: Path, page_number: int, zoom: float
) -> list[OCRDetection]:
    """OCR one FMB page via Google Cloud Vision, per-glyph canvas approach.

    Architecture:
    - Pass 0: PP-OCRv5 on header band only (scale, district, village).
    - Pass 1: Extract colored glyph clusters via pdf_glyphs (vector positions).
    - Pass 2: Vision REST API on each de-rotated canvas (handles rotated text).

    Vision is the PRIMARY OCR engine for the body — not a fallback.
    PP-OCRv5 handles only the header band where it reads reliably.

    Auth: GOOGLE_API_KEY env var.  Raises OCRFailure if unset.
    """
    # Must have at least one auth method configured.
    has_sa = bool(os.getenv("GOOGLE_CREDENTIALS_JSON", ""))
    try:
        from ...config import get_settings as _gs  # noqa: PLC0415
        has_sa = has_sa or bool(_gs().google_credentials_json)
    except Exception:  # noqa: BLE001
        pass
    has_api_key = bool(os.getenv("GOOGLE_API_KEY", ""))
    if not has_sa and not has_api_key:
        raise OCRFailure(
            "No Google Vision credentials configured. Set either:\n"
            "  GOOGLE_CREDENTIALS_JSON=<service-account JSON string>  (recommended)\n"
            "  GOOGLE_API_KEY=<key>  (API key restricted to Cloud Vision API)\n"
            "Get credentials at GCP Console → APIs & Services → Credentials.",
            pdf=str(pdf_path),
            page=page_number,
        )

    from .pdf_glyphs import (  # noqa: PLC0415
        extract_glyph_groups,
        glyph_polygon,
        largest_blue_glyph,
        render_glyph_group,
    )
    from .pdf_vectors import _classify_glyph_groups, extract_vectors  # noqa: PLC0415

    detections: list[OCRDetection] = []

    # ── Pass 0: header band via PP-OCRv5 ─────────────────────────────────────
    bgr = _render_page_bgr(pdf_path, page_number, zoom)
    header_px = int(_HEADER_LIMIT_Y * zoom)
    header_rgb = bgr[:header_px, :, ::-1].copy()  # BGR → RGB
    header_dets: list[OCRDetection] = []
    try:
        paddle_engine = _body_ocr_engine(False)
        raw_header = paddle_engine.predict(header_rgb)
        header_dets = [
            d for d in _parse_result(raw_header, zoom)
            if d.center[1] < _HEADER_LIMIT_Y
        ]
    except Exception as exc:  # noqa: BLE001
        _log.warning("Vision Pass 0 (header) failed: %s", exc)
    _log.info(
        "Vision Pass 0 (header): %d detections on %s", len(header_dets), pdf_path.name
    )
    detections.extend(header_dets)

    # ── Pass 1: glyph cluster extraction (vector positions) ──────────────────
    try:
        with fitz.open(pdf_path) as doc:
            page = doc[page_number]
            pw = page.rect.width
            ph = page.rect.height
            groups = extract_glyph_groups(page)
            _classify_glyph_groups(groups, pw, ph)
    except Exception as exc:  # noqa: BLE001
        _log.warning("Vision Pass 1 (glyph extraction) failed: %s", exc)
        return detections

    # Gap-split and replace overlapping-fill groups — same expansion as the
    # paddle path.  Without this, merged clusters produce misreads and fills
    # at the same corner position (e.g. 59.8 + 20.4) render as one garbled
    # glyph instead of two distinct measurements.
    expanded_groups = _expand_glyph_groups(groups)

    try:
        vecs = extract_vectors(pdf_path)
        all_segments: list[Any] = list(vecs.boundary) + list(vecs.internal) + list(vecs.chain)
    except Exception:  # noqa: BLE001
        all_segments = []

    by_color: dict[str, int] = {}
    for g in groups:
        by_color[g.color] = by_color.get(g.color, 0) + 1
    _log.info(
        "Vision Pass 1 (glyphs): %d raw groups → %d expanded "
        "(blue=%d red=%d black=%d) on %s",
        len(groups), len(expanded_groups),
        by_color.get("blue", 0), by_color.get("red", 0), by_color.get("black", 0),
        pdf_path.name,
    )

    # Inject survey-number glyph sentinel (same as paddle path).
    try:
        sg = largest_blue_glyph(groups)
        if sg is not None:
            detections.append(OCRDetection(
                text="__survey_glyph__", confidence=1.0,
                polygon=glyph_polygon(sg), kind="survey_number_glyph",
            ))
    except Exception as _exc:  # noqa: BLE001
        _log.debug("Vision survey glyph sentinel skipped: %s", _exc)

    # ── Pass 2: Vision OCR per de-rotated glyph canvas ───────────────────────
    vision_ok = 0
    for group in expanded_groups:
        try:
            nearest_seg = _find_nearest_line(group.center, all_segments)
            angle_for_render = (
                _seg_angle_deg(nearest_seg) if nearest_seg is not None else None
            )
            img1 = render_glyph_group(group, flip=False, angle_override_deg=angle_for_render)
            img2 = render_glyph_group(group, flip=True,  angle_override_deg=angle_for_render)
            pre1 = _preprocess_canvas(img1)
            pre2 = _preprocess_canvas(img2)

            r1 = _ocr_canvas_vision(pre1)
            r2 = _ocr_canvas_vision(pre2)

            if r1 is None and r2 is None:
                continue
            text, conf = (r1 if (r2 is None or (r1 is not None and r1[1] >= r2[1])) else r2)

            # Chain-dim paren detection: a bare decimal from a fill whose bbox
            # is wide relative to the character count is a chain dimension with
            # parens stripped by the rendering (they are strokes, not fills).
            if (
                re.match(r"^\d{1,4}[.,]\d{1,2}$", text)
                and group.bbox.width / max(len(text), 1) > 5.0
            ):
                text = f"({text})"

            vision_ok += 1
            kind_tag = group.kind if group.kind != "survey_number" else ""
            detections.append(OCRDetection(
                text=text, confidence=conf,
                polygon=glyph_polygon(group),
                angle_deg=group.angle_deg,
                kind=kind_tag,
            ))
        except Exception:  # noqa: BLE001
            continue

    _log.info(
        "Vision Pass 2: %d/%d expanded groups read; %d total detections on %s "
        "(canvas cache size: %d)",
        vision_ok, len(expanded_groups), len(detections), pdf_path.name,
        len(_vision_canvas_cache),
    )
    return detections


# --- Multi-angle full-page OCR -----------------------------------------------

_MULTIANGLE_ANGLES = (0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165)

# Reduced angle set for the AUGMENT path: the glyph pass already de-rotates each
# isolated cluster, so the page-level augment only needs to catch numbers the glyph
# grouping missed entirely. 30 deg spacing over [0,180) covers every orientation.
_AUGMENT_ANGLES = (0.0, 30.0, 60.0, 90.0, 120.0, 150.0)


def multiangle_augment_enabled() -> bool:
    """Whether to augment the default glyph body with a few-angle page pass.

    The documented path from ~24% single-pass recall toward usable: rotate the
    whole page through a few angles, OCR each, and merge the measurement-shaped
    tokens the glyph pass missed. Off by default (``LANDINTEL_OCR_MULTIANGLE_AUGMENT``
    unset / "0") so existing behaviour and test timing are unchanged; turn on in
    production where the extra passes are worth the recall.
    """
    return os.environ.get("LANDINTEL_OCR_MULTIANGLE_AUGMENT", "0") == "1"

# Tokens accepted by the multiangle body pass.  Anything not matching is
# dropped before dedup so garbage OCR noise never reaches anchor.py.
_MA_MEAS_RE = re.compile(r"^\(?\d{1,4}[.,]\d{1,2}\)?$")  # e.g. 24.8  (160.2)
_MA_SUBPLOT_RE = re.compile(r"^\d[A-Z]\d?$")             # e.g. 2A  5B1
_MA_STONE_RE   = re.compile(r"^([A-H]|[1-9]|[12]\d|3[01])$")  # A-H or 1-31


def _classify_ma_kind(text: str) -> str:
    if _MA_STONE_RE.match(text):
        return "corner_label" if re.match(r"^[A-H]$", text) else "stone_id"
    if _MA_SUBPLOT_RE.match(text):
        return "parcel_label"
    return "dimension"


def _extract_multiangle(
    pdf_path: Path,
    page_number: int,
    zoom: float,
    angles: tuple[float, ...] = _MULTIANGLE_ANGLES,
) -> list[OCRDetection]:
    """Full-page multi-angle OCR: render once, rotate N times, merge results.

    Bypasses glyph-grouping entirely.  At each rotation angle, diagonal
    measurements that were unreadable at 0° become near-horizontal and OCR
    picks them up.  12 angles at 15° spacing covers all orientations.

    Only measurement-shaped tokens (decimals, sub-plot labels, stone IDs) are
    kept; everything else is dropped before dedup to limit anchor.py noise.
    Position is approximate (rotated image → inverse-transform back to PDF
    space) rather than the exact vector position used by the glyph pass, but
    close enough for distance-based anchoring.

    Slow: 12× PP-OCRv5 passes on the full page.  Acceptable locally; for
    cloud deployment limit to angles where gaps remain after the paddle pass.
    """
    import cv2  # noqa: PLC0415

    engine = _body_ocr_engine(False)

    # Render full page once; all rotation ops work on this array.
    full_img = _render_page(pdf_path, page_number, zoom)  # RGB H×W×3
    h, w = full_img.shape[:2]
    cx_img, cy_img = w / 2.0, h / 2.0

    # Body-only band: skip the header strip so PP-OCRv5 doesn't re-read header
    # text at every rotation and pollute the measurement token list.
    header_px = int(_HEADER_LIMIT_Y * zoom)

    seen: list[tuple[float, float, str]] = []  # (x_pdf, y_pdf, text) of accepted tokens
    body_dets: list[OCRDetection] = []
    new_per_angle: dict[float, int] = {}

    for angle in angles:
        if angle == 0.0:
            rotated = full_img
            M_inv: Any = None
        else:
            M = cv2.getRotationMatrix2D((cx_img, cy_img), float(angle), 1.0)
            cos_a, sin_a = abs(M[0, 0]), abs(M[0, 1])
            new_w = int(h * sin_a + w * cos_a)
            new_h = int(h * cos_a + w * sin_a)
            M[0, 2] += (new_w - w) / 2.0
            M[1, 2] += (new_h - h) / 2.0
            rotated = cv2.warpAffine(
                full_img, M, (new_w, new_h),
                flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255),
            )
            M_inv = cv2.invertAffineTransform(M)

        try:
            raw = engine.predict(rotated)
            angle_dets = _parse_result(raw, zoom)
        except Exception as exc:  # noqa: BLE001
            _log.debug("Multiangle OCR angle=%s failed: %s", angle, exc)
            new_per_angle[angle] = 0
            continue

        new_this_angle = 0
        for det in angle_dets:
            # Map polygon corners back to original image space.
            if M_inv is not None:
                pts = np.array([[p[0], p[1]] for p in det.polygon], dtype=np.float32)
                ones = np.ones((len(pts), 1), dtype=np.float32)
                orig_pts = (M_inv @ np.hstack([pts, ones]).T).T
                poly_pdf = tuple(
                    (float(p[0]) / zoom, float(p[1]) / zoom) for p in orig_pts
                )
            else:
                poly_pdf = det.polygon

            cx_pdf = sum(p[0] for p in poly_pdf) / len(poly_pdf)
            cy_pdf = sum(p[1] for p in poly_pdf) / len(poly_pdf)

            # Skip header band — paddle pass handles it.
            if cy_pdf < _HEADER_LIMIT_Y:
                continue

            text = det.text.strip()

            # Token filter: only measurement-shaped strings.
            if not (_MA_MEAS_RE.match(text) or _MA_SUBPLOT_RE.match(text)
                    or _MA_STONE_RE.match(text)):
                continue

            # Dedup: same text within 8 PDF points of an existing detection.
            if any(
                text == s_text and math.hypot(cx_pdf - sx, cy_pdf - sy) < 8.0
                for sx, sy, s_text in seen
            ):
                continue

            seen.append((cx_pdf, cy_pdf, text))
            body_dets.append(OCRDetection(
                text=text,
                confidence=det.confidence,
                polygon=poly_pdf,
                angle_deg=float(angle),
                kind=_classify_ma_kind(text),
            ))
            new_this_angle += 1

        new_per_angle[angle] = new_this_angle

    _log.info(
        "Multiangle body: %d unique tokens across %d angles — per-angle new: %s",
        len(body_dets), len(angles),
        {int(a): n for a, n in new_per_angle.items()},
    )
    return body_dets


# --- Native header extraction (no OCR model needed) --------------------------


def _extract_header_native(pdf_path: Path, page_number: int) -> list[OCRDetection]:
    """Read header text from the PDF's native text stream (no OCR model).

    The FMB header block (District, Taluk, Village, Scale, Area, Survey No)
    is typeset as native PDF text — not as vector glyph fills.  PyMuPDF reads
    it instantly with ``get_text("text")``, producing multi-word phrases like
    ``"Scale : 1 : 2021"`` that ``parse_header``'s regexes require intact.

    Returns one OCRDetection per non-empty line, all with confidence=1.0.
    The polygon is a dummy header-band rectangle — parse_header ignores
    coordinates; it reads text only.
    """
    with fitz.open(pdf_path) as doc:
        page = doc[page_number]
        raw = page.get_text("text")
    _dummy_poly = ((0.0, 0.0), (400.0, 0.0), (400.0, 20.0), (0.0, 20.0))
    dets: list[OCRDetection] = []
    for line in raw.split("\n"):
        line = line.strip()
        if line:
            dets.append(OCRDetection(text=line, confidence=1.0, polygon=_dummy_poly))
    _log.info(
        "native header [%s p%d]: %d lines of native text",
        pdf_path.name, page_number, len(dets),
    )
    return dets


# --- Public entry point -------------------------------------------------------


def extract_text(
    pdf_path: Path | str,
    *,
    page_number: int = 0,
    zoom: float = DEFAULT_ZOOM,
    det_model: str = DEFAULT_DET_MODEL,
    rec_model: str = DEFAULT_REC_MODEL,
    use_mkldnn: bool = False,
) -> list[OCRDetection]:
    """Render an FMB page and OCR it, routing to the configured OCR engine.

    Engine selection (``OCR_ENGINE`` env/setting, default ``paddle``):

    **paddlevl** (``OCR_ENGINE=paddlevl``): runs PaddleOCR-VL-1.6 locally.
    Document VLM (~0.9B params) with native irregular/rotated-text support.
    Returns per-word rotated polygons from ``spotting_res``; falls back to
    block-level tokens from ``parsing_res_list``.  Requires PaddlePaddle ≥ 3.1.

    **vision** (``OCR_ENGINE=vision``): posts the rendered page to Google Cloud
    Vision TEXT_DETECTION.  Falls back to PaddleOCR when credentials are absent
    or billing is disabled.

    **paddle** (``OCR_ENGINE=paddle``, default):

    * Stage 1 (page-level): rasterise the full page at ``zoom`` and run
      PP-OCRv5.  Reliably reads the header block and near-horizontal numbers.
    * Stage 2 (per-glyph): extract each dimension number's filled vector paths,
      render in isolation on a clean 520×150 canvas (de-rotated, scaled to 70 px
      height), OCR individually.  Adds only positions the page-level pass missed.

    All positions are returned in **PDF page coordinates** (origin top-left,
    y down), so ``anchor.py`` and ``build_plot.py`` need no changes regardless
    of which engine ran.

    Args:
        pdf_path: Path to the FMB PDF.
        page_number: Zero-based page index (FMB sheets are single-page).
        zoom: Render magnification before OCR (3.0 ≈ 216-300 DPI).
        det_model: PP-OCRv5 detection model name (PaddleOCR path only).
        rec_model: PP-OCRv5 recognition model name (PaddleOCR path only).
        use_mkldnn: Enable oneDNN/mkldnn (PaddleOCR path only; leave off if
            the host CPU is unstable with it).

    Returns:
        Raw detections, one per text box. Empty when the page has no text.

    Raises:
        OCRFailure: If the selected engine errors out and no fallback succeeded.
    """
    path = Path(pdf_path)

    # Log vector-vs-raster encoding diagnostic so every processed PDF reports
    # whether its dimension numbers are vector (glyph extraction → 100%) or
    # raster (OCR required). This is the answer to "what's the recall ceiling?"
    try:
        from .pdf_glyphs import diagnose_encoding  # noqa: PLC0415
        with fitz.open(path) as _doc:
            _diag = diagnose_encoding(_doc[page_number])
        _log.info("PDF encoding [%s p%d]: %s", path.name, page_number, _diag.summary())
    except Exception:  # noqa: BLE001
        pass  # diagnostic is informational; never block extraction

    try:
        from ...config import get_settings
        engine_name = get_settings().ocr_engine.lower()
    except Exception:  # noqa: BLE001
        engine_name = os.getenv("OCR_ENGINE", "paddle").lower()

    if engine_name == "hf_vl":
        try:
            return _extract_hf_vl(path, page_number, zoom, det_model, rec_model)
        except Exception as exc:  # noqa: BLE001
            _log.warning(
                "HF Qwen2.5-VL failed (%s); falling back to PaddleOCR paddle path.", exc
            )

    if engine_name == "paddlevl":
        try:
            return _extract_paddlevl(path, page_number, zoom)
        except Exception as exc:  # noqa: BLE001
            _log.warning(
                "PaddleOCR-VL-1.6 failed (%s); falling back to PaddleOCR paddle path.",
                exc,
            )

    if engine_name == "vision":
        try:
            return _extract_vision(path, page_number, zoom)
        except Exception as exc:  # noqa: BLE001
            _log.warning(
                "Google Vision OCR failed (%s); falling back to PaddleOCR. "
                "Check GOOGLE_CREDENTIALS_JSON in your environment.",
                exc,
            )

    if engine_name == "multiangle":
        # Header: PP-OCRv5 on full page (reads header band reliably at 0°).
        # Body: 12-angle full-page OCR merged with dedup.
        image = _render_page(path, page_number, zoom)
        engine = _build_engine(det_model, rec_model, use_mkldnn)
        try:
            raw = engine.predict(image)
        except Exception as exc:  # noqa: BLE001
            raise OCRFailure(
                "PP-OCRv5 header prediction failed (multiangle)", pdf=str(pdf_path), page=page_number
            ) from exc
        header_dets = [
            d for d in _parse_result(raw, zoom) if d.center[1] < _HEADER_LIMIT_Y
        ]
        _log.info(
            "Multiangle header: %d detections on %s", len(header_dets), path.name
        )
        body_dets = _extract_multiangle(path, page_number, zoom)
        _log.info(
            "Multiangle total: %d header + %d body = %d on %s",
            len(header_dets), len(body_dets), len(header_dets) + len(body_dets), path.name,
        )
        return header_dets + body_dets

    # Vector-first path (default for paddle + paddlevl fallback):
    #
    #   Header — PP-OCRv5 on the header crop only.  The FMB header block
    #     ("Scale : 1 : 2021", "District : Sivagangai", …) is rendered/rasterized
    #     in the PDF — native text stream is empty — so OCR is required.  The
    #     header band is black text on white, reads cleanly and reliably.
    #
    #   Body — per-glyph OCR: each blue/red/black cluster of filled vector paths
    #     is rendered in isolation on a clean 520x150 canvas (de-rotated, scaled
    #     to 70 px height) and PP-OCRv5 reads its VALUE only.  Position comes from
    #     the vector bbox (exact), not from OCR detection position (noisy).
    #     No page-level OCR on the body — glyph extraction replaces it entirely.
    image = _render_page(path, page_number, zoom)
    engine = _build_engine(det_model, rec_model, use_mkldnn)
    try:
        raw = engine.predict(image)
    except Exception as exc:  # noqa: BLE001
        raise OCRFailure(
            "PP-OCRv5 header prediction failed", pdf=str(pdf_path), page=page_number
        ) from exc
    all_page_dets = _parse_result(raw, zoom)
    header_dets = [d for d in all_page_dets if d.center[1] < _HEADER_LIMIT_Y]
    _log.info(
        "OCR header [%s p%d]: %d page-level → %d header-band kept",
        path.name, page_number, len(all_page_dets), len(header_dets),
    )

    if engine_name == "perchar":
        body_perchar = _extract_perchar_detections(path, page_number, engine)
        body_perchar = [d for d in body_perchar if d.center[1] >= _HEADER_LIMIT_Y]
        _log.info(
            "OCR result [%s p%d perchar]: %d header + %d body = %d total",
            path.name, page_number, len(header_dets), len(body_perchar),
            len(header_dets) + len(body_perchar),
        )
        return header_dets + body_perchar

    if engine_name == "shapematch":
        body_shape = _extract_shapematch(path, page_number, engine)
        _log.info(
            "shapematch result [%s p%d]: %d header + %d body = %d total",
            path.name, page_number, len(header_dets), len(body_shape),
            len(header_dets) + len(body_shape),
        )
        return header_dets + body_shape

    glyph_dets = _extract_glyph_detections(path, page_number, engine)
    body_glyph_dets = [d for d in glyph_dets if d.center[1] >= _HEADER_LIMIT_Y]

    # Optional multi-angle augmentation: recover measurement numbers the glyph
    # grouping missed entirely (steeply-rotated tokens) by OCRing the page at a few
    # rotations and merging only the NEW positions (proximity-dedup against the
    # glyph body, so a number found by both is not duplicated). See
    # multiangle_augment_enabled. Vector glyph positions stay authoritative; the
    # augment only ADDS misses, never overrides an existing reading.
    if multiangle_augment_enabled():
        try:
            ma_body = _extract_multiangle(path, page_number, zoom, angles=_AUGMENT_ANGLES)
            ma_body = [d for d in ma_body if d.center[1] >= _HEADER_LIMIT_Y]
            before = len(body_glyph_dets)
            body_glyph_dets = _merge_glyph_detections(body_glyph_dets, ma_body)
            _log.info("OCR multiangle augment [%s p%d]: +%d new tokens (glyph %d -> %d)",
                      path.name, page_number, len(body_glyph_dets) - before,
                      before, len(body_glyph_dets))
        except Exception as exc:  # noqa: BLE001
            _log.warning("multiangle augment failed (%s); using glyph body only", exc)

    _log.info(
        "OCR result [%s p%d]: %d header + %d body glyphs = %d total",
        path.name, page_number, len(header_dets), len(body_glyph_dets),
        len(header_dets) + len(body_glyph_dets),
    )

    return header_dets + body_glyph_dets


def _extract_perchar_detections(
    pdf_path: Path,
    page_number: int,
    engine: Any,
) -> list[OCRDetection]:
    """Per-character OCR: render each digit glyph individually, then concatenate.

    Instead of rendering the whole cluster (asking OCR to find AND read a
    multi-digit number), this renders each member path of a GlyphGroup in
    isolation on a clean 200×200 canvas (de-rotated to upright, scaled to
    ~120 px) and reads exactly ONE character per call.  The results are
    concatenated in PCA (left-to-right-along-baseline) order to form the
    number string.

    Advantages over cluster-level OCR:
    - Single-digit OCR on a clean, large canvas → near-100% accuracy.
    - No dependence on the number string layout within the cluster canvas.
    - The comma separator "," and parentheses "(" ")" still rendered from
      their own vector paths.

    Falls back to the existing cluster rendering for groups whose members
    lack vector items (e.g. compound paths that are already a single fitz
    drawing encoding all digits).
    """
    from .pdf_glyphs import (  # noqa: PLC0415
        extract_glyph_groups,
        glyph_polygon,
        member_pca_order,
        render_member_glyph,
    )
    from .pdf_vectors import _classify_glyph_groups, extract_vectors  # noqa: PLC0415

    detections: list[OCRDetection] = []
    all_segments: list[Any] = []

    try:
        with fitz.open(pdf_path) as doc:
            page = doc[page_number]
            pw = page.rect.width
            ph = page.rect.height
            groups = extract_glyph_groups(page)
            _classify_glyph_groups(groups, pw, ph)
    except Exception:  # noqa: BLE001
        return detections

    try:
        vecs = extract_vectors(pdf_path)
        all_segments = list(vecs.boundary) + list(vecs.internal) + list(vecs.chain)
    except Exception:  # noqa: BLE001
        pass

    survey_glyph_group = None

    for group in groups:
        try:
            if group.kind == "survey_number":
                survey_glyph_group = group
                continue

            # Angle: prefer nearest vector line, fall back to PCA shape angle.
            nearest_seg = _find_nearest_line(group.center, all_segments)
            angle_deg = (
                _seg_angle_deg(nearest_seg)
                if nearest_seg is not None
                else group.angle_deg
            )

            ordered = member_pca_order(group)

            # Per-character rendering: one OCR call per member drawing.
            # For compound paths (single drawing = whole number), fall back to
            # the cluster canvas approach.
            if len(ordered) == 1:
                # Either a genuinely single-character group or a compound path —
                # use the cluster canvas (renders the whole path, already centred).
                from .pdf_glyphs import render_glyph_group  # noqa: PLC0415
                img1 = render_glyph_group(group, flip=False, angle_override_deg=angle_deg)
                img2 = render_glyph_group(group, flip=True,  angle_override_deg=angle_deg)
                pre1 = _preprocess_canvas(img1)
                pre2 = _preprocess_canvas(img2)
                r1 = _ocr_canvas(pre1, engine)
                r2 = _ocr_canvas(pre2, engine)
                if r1 and r2:
                    result = r1 if r1[1] >= r2[1] else r2
                else:
                    result = r1 or r2
                if not result:
                    continue
                text, conf = result
            else:
                # Multi-member group: render each character individually.
                chars: list[tuple[str, float]] = []

                for drawing in ordered:
                    # Use each character's own bbox center as the rotation pivot
                    # so it lands in the middle of the 200×200 canvas, not off-edge.
                    r = drawing["rect"]
                    char_center = ((r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2)
                    img1 = render_member_glyph(drawing, angle_deg, char_center, flip=False)
                    img2 = render_member_glyph(drawing, angle_deg, char_center, flip=True)
                    pre1 = _preprocess_canvas(img1)
                    pre2 = _preprocess_canvas(img2)
                    r1 = _ocr_canvas(pre1, engine)
                    r2 = _ocr_canvas(pre2, engine)
                    if r1 and r2:
                        best = r1 if r1[1] >= r2[1] else r2
                    else:
                        best = r1 or r2
                    if best:
                        chars.append(best)

                if not chars:
                    continue
                text = "".join(c for c, _ in chars)
                conf = min(c for _, c in chars)

                # A merged group may contain digits from two adjacent measurement
                # labels (e.g. "28.453.2" = "28.4" + "53.2").  Split by finding
                # all non-overlapping measurement-shaped tokens so each anchors
                # independently against its nearest line.
                splits = _PERCHAR_MEAS_RE.findall(text)
                if len(splits) > 1:
                    poly = glyph_polygon(group)
                    for token in splits:
                        detections.append(OCRDetection(
                            text=token,
                            confidence=conf,
                            polygon=poly,
                            angle_deg=group.angle_deg,
                            kind=group.kind,
                        ))
                    continue  # skip the single-token path below

            if not text:
                continue

            poly = glyph_polygon(group)
            det = OCRDetection(
                text=text,
                confidence=conf,
                polygon=poly,
                angle_deg=group.angle_deg,
                kind=group.kind,
            )
            detections.append(det)

        except Exception:  # noqa: BLE001
            _log.debug("perchar OCR failed for group at %s", group.center, exc_info=True)

    # Survey number glyph sentinel.
    if survey_glyph_group is not None:
        cx = (survey_glyph_group.bbox.x0 + survey_glyph_group.bbox.x1) / 2
        cy = (survey_glyph_group.bbox.y0 + survey_glyph_group.bbox.y1) / 2
        detections.append(OCRDetection(
            text="survey_number_glyph_sentinel",
            confidence=1.0,
            polygon=((cx, cy), (cx, cy), (cx, cy), (cx, cy)),
            angle_deg=0.0,
            kind="survey_number_glyph",
        ))

    return detections


def _extract_shapematch(
    pdf_path: Path,
    page_number: int,
    engine: Any,
) -> list[OCRDetection]:
    """Shape-matching engine: Hu-moment fingerprinting on raw bezier paths.

    No rasterization.  Each digit glyph path is converted to an OpenCV contour,
    7 log-transformed Hu moments are computed (rotation/scale/translation
    invariant), and the path is matched against templates built from
    paddle-confirmed measurements on the same page.

    Pipeline:
      1. Run ``_extract_glyph_detections`` (paddle body) to obtain confirmed
         measurements with known text and PDF-space center coordinates.
      2. For each confirmed measurement, find the nearest dimension glyph group,
         verify member count == char count, and label each member path.
      3. Match ALL dimension groups against the template library.
      4. Validate each assembled string with ``_MEAS_RE``; emit OCRDetection.

    Falls back to paddle body when fewer than 5 measurements can be confirmed
    (too little signal to build reliable templates).
    """
    from .pdf_glyphs import (  # noqa: PLC0415
        build_digit_templates,
        extract_glyph_groups,
        fix_6_9,
        glyph_polygon,
        member_pca_order,
        recognize_digit_by_shape,
    )
    from .pdf_vectors import _classify_glyph_groups  # noqa: PLC0415

    detections: list[OCRDetection] = []

    # ── Step 1: paddle body for bootstrap text ──────────────────────────────
    paddle_body = _extract_glyph_detections(pdf_path, page_number, engine)

    # ── Step 2: glyph groups (same call the paddle body already did) ─────────
    try:
        with fitz.open(pdf_path) as doc:
            page = doc[page_number]
            pw = page.rect.width
            ph = page.rect.height
            groups = extract_glyph_groups(page)
            _classify_glyph_groups(groups, pw, ph)
    except Exception:  # noqa: BLE001
        return paddle_body

    # Collect (text, center) pairs from paddle detections that look like
    # valid measurements (kind=="dimension", text matches _MEAS_RE, conf > 0.7).
    known: list[tuple[str, tuple[float, float]]] = []
    for det in paddle_body:
        if det.kind != "dimension":
            continue
        text = det.text.strip()
        if _MEAS_RE.match(text) and det.confidence >= 0.7:
            known.append((text, det.center))

    if len(known) < 5:
        _log.info(
            "shapematch [%s p%d]: only %d confirmed measurements — "
            "insufficient for templates, falling back to paddle",
            pdf_path.name, page_number, len(known),
        )
        return paddle_body

    # ── Step 3: build templates ──────────────────────────────────────────────
    templates = build_digit_templates(groups, known)
    n_templates = sum(len(v) for v in templates.values())
    _log.info(
        "shapematch [%s p%d]: %d paddle seeds → templates for %d chars "
        "(%d total entries): %s",
        pdf_path.name, page_number, len(known),
        len(templates), n_templates,
        {k: len(v) for k, v in sorted(templates.items())},
    )

    if not templates:
        # This PDF encodes decimal measurements as compound paths (whole number
        # as one fill, or digit-group pairs like "28." + "6"), so individual
        # digit paths are not separately addressable for Hu moment templates.
        # Fall back to paddle which handles compound paths via canvas rendering.
        _log.info(
            "shapematch [%s p%d]: no individual-digit templates — compound path "
            "encoding detected; falling back to paddle",
            pdf_path.name, page_number,
        )
        return paddle_body

    # ── Step 4: match all dimension groups ───────────────────────────────────
    # Track which groups the paddle body already handled so we don't double-emit.
    used_centers: set[tuple[float, float]] = set()
    for det in paddle_body:
        used_centers.add((round(det.center[0], 1), round(det.center[1], 1)))

    shape_matched = 0
    for group in groups:
        if group.kind != "dimension":
            continue

        ordered = member_pca_order(group)
        char_results: list[tuple[str, float, Any]] = []

        for drawing in ordered:
            result = recognize_digit_by_shape(drawing, templates)
            if result is None:
                break
            char, conf = result
            char_results.append((char, conf, drawing["rect"]))

        if len(char_results) != len(ordered):
            continue  # at least one path had no match → skip this group

        # Disambiguate 6 vs 9 using cluster baseline orientation.
        fixed_chars = []
        for char, conf, rect in char_results:
            fixed = fix_6_9(char, rect, group.angle_deg, group.bbox)
            fixed_chars.append((fixed, conf))

        text = "".join(c for c, _ in fixed_chars)

        # Validate: the assembled string must look like a measurement.
        if not _MEAS_RE.match(text):
            continue

        conf = min(c for _, c in fixed_chars)
        key = (round(group.center[0], 1), round(group.center[1], 1))
        poly = glyph_polygon(group)

        if key in used_centers:
            # Paddle already emitted this group; replace with our higher-quality read.
            # Keep the paddle result as-is — only replace if shapematch gives
            # a different (hopefully correct) string.
            # We emit anyway and let anchor.py deduplicate by position.
            pass

        detections.append(OCRDetection(
            text=text,
            confidence=conf,
            polygon=poly,
            angle_deg=group.angle_deg,
            kind=group.kind,
        ))
        shape_matched += 1

    _log.info(
        "shapematch [%s p%d]: %d dimension groups matched by shape",
        pdf_path.name, page_number, shape_matched,
    )

    # Emit non-dimension paddle detections (header already handled by caller;
    # here we pass sub-plot, chain-dim, sentinel, neighbor, stone detections
    # through unchanged so anchor.py still has them).
    for det in paddle_body:
        if det.kind != "dimension":
            detections.append(det)

    return detections


# --- Header parsing ----------------------------------------------------------


@dataclass(frozen=True)
class FmbHeader:
    """Structured job metadata parsed from the FMB header block.

    The header sits in a fixed band at the top of the sheet and OCRs at high
    confidence. ``scale_denominator`` and ``stated_area_ha`` are the two values
    that gate geometric correctness downstream:

    * ``scale_denominator`` (e.g. 2021 from "Scale : 1 : 2021") is per-PDF and is
      applied in ``build_plot.py`` to turn pixel geometry into real-world metres.
    * ``stated_area_ha`` (e.g. 1.665 from "Area : Hect 01 Ares 66.50") is the
      government-stated area the anomaly layer cross-checks the computed polygon
      area against -- if the scale was read wrong, that check catches it.

    Any field is ``None`` when it could not be parsed.
    """

    survey_no: str | None = None
    district: str | None = None
    taluk: str | None = None
    village: str | None = None
    scale_denominator: int | None = None
    stated_area_ha: float | None = None


# Each header field is its own detection; these match the "Label : value" form
# seen across every fixture ("Scale : 1 : 2021", "Area : Hect 01 Ares 66.50").
_RE_SURVEY = re.compile(r"Survey\s*No\s*[:;]\s*(\S+)", re.IGNORECASE)
_RE_DISTRICT = re.compile(r"District\s*[:;]\s*(.+)", re.IGNORECASE)
_RE_TALUK = re.compile(r"Taluk\s*[:;]\s*(.+)", re.IGNORECASE)
_RE_VILLAGE = re.compile(r"Village\s*[:;]\s*(.+?)(?:\s*\[|$)", re.IGNORECASE)
_RE_SCALE = re.compile(r"Scale\s*[:;]\s*1\s*[:;]\s*(\d+)", re.IGNORECASE)
_RE_AREA = re.compile(r"Hect\s*(\d+)\s*Ares\s*([\d.,]+)", re.IGNORECASE)
# Legacy / non-metric area on older FMB sheets. TN revenue uses acre + cent
# (1 acre = 100 cents = 0.404686 ha; 1 cent = 0.00404686 ha). Parsed as a
# fallback so an older sheet still yields stated_area_ha and keeps the area
# cross-check gate (without it the gate silently switches off). Order: try the
# modern Hect/Ares first, then Acre/Cent.
_RE_AREA_ACRE = re.compile(r"Acres?\s*(\d+)\s*Cents?\s*([\d.,]+)", re.IGNORECASE)
# Generic single-value form: "Area : 1.665 Ha", "Area : 2.5 acres", "Area : 80 cents".
# Anchored on the Area label + an explicit unit so it cannot grab a stray number.
_RE_AREA_DECIMAL = re.compile(
    r"Area\s*[:;]?\s*([\d.,]+)\s*(hectares?|hect|ha|acres?|cents?)\b", re.IGNORECASE)
_ACRE_HA = 0.404686            # 1 acre in hectares
_CENT_HA = _ACRE_HA / 100.0    # 1 cent in hectares


def parse_header(
    detections: list[OCRDetection], *, min_confidence: float = 0.5
) -> FmbHeader:
    """Extract structured job metadata from OCR detections.

    Scans the recognized strings for the header fields. Only detections at or
    above ``min_confidence`` are considered, since header text reads cleanly and
    we would rather report a field as missing than parse a low-confidence guess.
    The first match per field wins. Field *values* are returned as read (the
    only conversions are structural: scale ratio -> denominator int, and
    "Hect H Ares A" -> hectares as a float).
    """
    header = {
        "survey_no": None,
        "district": None,
        "taluk": None,
        "village": None,
        "scale_denominator": None,
        "stated_area_ha": None,
    }
    for det in detections:
        if det.confidence < min_confidence:
            continue
        text = det.text

        if header["survey_no"] is None and (m := _RE_SURVEY.search(text)):
            header["survey_no"] = m.group(1).strip()
        if header["district"] is None and (m := _RE_DISTRICT.search(text)):
            header["district"] = m.group(1).strip()
        if header["taluk"] is None and (m := _RE_TALUK.search(text)):
            header["taluk"] = m.group(1).strip()
        if header["village"] is None and (m := _RE_VILLAGE.search(text)):
            header["village"] = m.group(1).strip()
        if header["scale_denominator"] is None and (m := _RE_SCALE.search(text)):
            header["scale_denominator"] = int(m.group(1))
        if header["stated_area_ha"] is None and (m := _RE_AREA.search(text)):
            hect = int(m.group(1))
            ares = float(m.group(2).replace(",", "."))
            header["stated_area_ha"] = hect + ares / 100.0
        # Acre/Cent fallback for older sheets (only if Hect/Ares didn't match).
        if header["stated_area_ha"] is None and (m := _RE_AREA_ACRE.search(text)):
            acres = int(m.group(1))
            cents = float(m.group(2).replace(",", "."))
            header["stated_area_ha"] = acres * _ACRE_HA + cents * _CENT_HA
        # Generic single-value "Area : <n> <unit>" fallback (ha / acres / cents).
        if header["stated_area_ha"] is None and (m := _RE_AREA_DECIMAL.search(text)):
            val = float(m.group(1).replace(",", "."))
            unit = m.group(2).lower()
            if unit.startswith(("ha", "hect")):
                header["stated_area_ha"] = val
            elif unit.startswith("acre"):
                header["stated_area_ha"] = val * _ACRE_HA
            elif unit.startswith("cent"):
                header["stated_area_ha"] = val * _CENT_HA

    return FmbHeader(**header)
