"""Tests for dark-reference (flat-field) normalization support."""
from __future__ import annotations

import numpy as np
import pytest

from hyperio import HSI
from hyperio.io import (
    _normalize_cube_with_reference,
    _read_sidecar_json,
    _write_sidecar_json,
)


def test_normalize_without_dark_matches_white_only():
    cube = np.random.default_rng(0).random((4, 5, 6), dtype=np.float32)
    ref = np.linspace(0.5, 1.0, 6, dtype=np.float32)
    out = _normalize_cube_with_reference(cube, ref, reference_multiplier=2.0)
    np.testing.assert_allclose(out, cube / (ref * 2.0), rtol=1e-6)


def test_normalize_flat_field_formula():
    cube = np.random.default_rng(1).random((4, 5, 6), dtype=np.float32)
    ref = np.linspace(0.5, 1.0, 6, dtype=np.float32)
    dark = np.full(6, 0.01, dtype=np.float32)
    out = _normalize_cube_with_reference(
        cube, ref, reference_multiplier=2.0, dark_reference=dark
    )
    expected = (cube - dark) / (ref * 2.0 - dark)
    np.testing.assert_allclose(out, expected, rtol=1e-6)


def test_normalize_dark_length_mismatch():
    with pytest.raises(ValueError):
        _normalize_cube_with_reference(
            np.zeros((2, 2, 3), dtype=np.float32),
            np.ones(3, dtype=np.float32),
            dark_reference=np.ones(2, dtype=np.float32),
        )


def test_sidecar_dark_reference_roundtrip(tmp_path):
    wl = np.linspace(400.0, 900.0, 10, dtype=np.float32)
    dark = np.linspace(0.0, 0.05, 10, dtype=np.float32)
    _write_sidecar_json(tmp_path / "cube.hdr", wl, dark_reference=dark)
    sc = _read_sidecar_json(tmp_path / "cube.hdr")
    assert sc is not None
    np.testing.assert_allclose(np.asarray(sc["dark_reference"], np.float32), dark, rtol=1e-6)


def test_hsi_read_recovers_dark_reference_from_sidecar(tmp_path):
    cube = np.random.default_rng(2).random((3, 3, 5), dtype=np.float32)
    wl = np.linspace(400.0, 900.0, 5, dtype=np.float32)
    dark = np.linspace(0.0, 0.02, 5, dtype=np.float32)

    h = HSI(cube, wl)
    h.write(tmp_path / "cube.hdr", metadata_json=True, dark_reference=dark)

    h2 = HSI.read(tmp_path / "cube.hdr", metadata_json=True)
    assert h2.dark_reference is not None
    np.testing.assert_allclose(h2.dark_reference, dark, rtol=1e-6)


def test_hsi_read_explicit_dark_reference_stored(tmp_path):
    cube = np.random.default_rng(3).random((3, 3, 5), dtype=np.float32)
    wl = np.linspace(400.0, 900.0, 5, dtype=np.float32)
    dark = np.linspace(0.0, 0.02, 5, dtype=np.float32)

    h = HSI(cube, wl)
    h.write(tmp_path / "cube.tiff", metadata_json=True)

    h2 = HSI.read(tmp_path / "cube.tiff", metadata_json=True, dark_reference=dark)
    np.testing.assert_allclose(h2.dark_reference, dark, rtol=1e-6)