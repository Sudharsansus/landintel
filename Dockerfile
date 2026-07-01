FROM python:3.11-slim

# System libraries required by PaddleOCR / OpenCV.
# libgomp1   — OpenMP (paddle inference)
# libgl1     — OpenCV headless rendering (used by paddleocr internally)
# libglib2.0 — gthread / glib (linked by libGL)
# libsm6 libxext6 libxrender1 libfontconfig1 — X11 stubs (OpenCV imports but doesn't need display)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libfontconfig1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Dependency layer (cached until pyproject.toml changes) ──────────────────
# Stub the package so pip can install all dependencies from pyproject.toml
# without needing the full source tree. The real source is copied below and
# takes precedence via PYTHONPATH=/app/src.
COPY pyproject.toml README.md ./
RUN mkdir -p src/landintel && touch src/landintel/__init__.py
RUN pip install --no-cache-dir ".[dev]"

# ── Pre-download PP-OCRv5 mobile models ────────────────────────────────────
# Embeds model weights in the image so there is zero network dependency at
# runtime. Args must mirror _build_engine() in ocr.py exactly — same model
# names, same flags — so this warms the models the app actually loads.
# Model name constants: ocr.py:DEFAULT_DET_MODEL / DEFAULT_REC_MODEL.
# (use_angle_cls / use_gpu / show_log are PaddleOCR 2.x args; 3.x removed them)
RUN python -c "from paddleocr import PaddleOCR; PaddleOCR(text_detection_model_name='PP-OCRv5_mobile_det', text_recognition_model_name='en_PP-OCRv5_mobile_rec', use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False, enable_mkldnn=False)"

# ── ODA File Converter (DXF -> DWG, required by M2) ────────────────────────
# Download the Linux binary from:
#   https://www.opendesign.com/guestfiles/oda_file_converter
# (free, registration required — not redistributable, so NOT in git)
# Place it at bin/oda_converter/ODAFileConverter before running docker build.
# If absent, M2 will raise ConfigError at runtime with a clear message
# (config.py: oda_converter_path is checked at use-time in M2, not at startup).
COPY bin/ /app/bin/
RUN chmod +x /app/bin/oda_converter/ODAFileConverter 2>/dev/null || true

# ── Application source ──────────────────────────────────────────────────────
COPY src/ src/
COPY tests/ tests/

# PYTHONPATH puts src/ first so the real source overrides the pip-installed stub.
ENV PYTHONPATH=/app/src
ENV ODA_CONVERTER_PATH=/app/bin/oda_converter/ODAFileConverter

EXPOSE 8000

# Default: run the API.
# To run the worker instead, override CMD in docker-compose or systemd:
#   docker run landintel celery -A landintel.workers.celery_app worker --loglevel=info
CMD ["uvicorn", "landintel.api.main:app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
