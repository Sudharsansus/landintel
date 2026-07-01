"""Model-free tests for the OCR multi-angle augment merge + TensorRT/augment knobs.

The heavy OCR engines need PaddleOCR/torch + the rendered page, so these pin the
PURE logic the augmentation relies on: proximity-dedup merge (a number found by
both the glyph pass and the page-angle pass is not duplicated; a genuinely new
position is added) and the env-gated feature flags.
"""
from __future__ import annotations

import importlib

from landintel.pipeline.m1_extract.ocr import (OCRDetection,
                                               _merge_glyph_detections,
                                               multiangle_augment_enabled,
                                               onnx_trt_available)


def _det(x, y, text="40.0"):
    poly = ((x - 1, y - 1), (x + 1, y - 1), (x + 1, y + 1), (x - 1, y + 1))
    return OCRDetection(text=text, confidence=0.9, polygon=poly)


def test_merge_adds_new_positions_only():
    glyph = [_det(10, 10, "40.0"), _det(50, 50, "12.3")]
    # One coincides with an existing glyph detection (within dedup radius), one new.
    angle = [_det(10.5, 10.5, "40.0"), _det(200, 200, "99.9")]
    merged = _merge_glyph_detections(glyph, angle)
    texts = sorted(d.text for d in merged)
    assert texts == ["12.3", "40.0", "99.9"]          # the near-duplicate dropped
    assert len(merged) == 3


def test_merge_is_noop_without_augment_dets():
    glyph = [_det(10, 10), _det(50, 50)]
    assert _merge_glyph_detections(glyph, []) == glyph


def test_augment_flag_defaults_off(monkeypatch):
    monkeypatch.delenv("LANDINTEL_OCR_MULTIANGLE_AUGMENT", raising=False)
    assert multiangle_augment_enabled() is False
    monkeypatch.setenv("LANDINTEL_OCR_MULTIANGLE_AUGMENT", "1")
    assert multiangle_augment_enabled() is True


def test_trt_probe_returns_bool():
    # Just a clean boolean probe -- never raises even without onnxruntime/TRT.
    assert isinstance(onnx_trt_available(), bool)
