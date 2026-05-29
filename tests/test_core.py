from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

from hyperio import HSI
from hyperio.io import (
    _align_wavelength_count,
    _canonicalize_reference_for_unit_cube,
    _normalize_cube_with_reference,
    _parse_payload_dict,
    _parse_shape_from_text,
    _reorder_hwc_to_target_shape,
)


def test_constructor_integer_cube_is_normalized_to_float32() -> None:
    cube = np.array([[[0, 127, 255]]], dtype=np.uint8)
    wavelengths = np.array([450.0, 550.0, 650.0], dtype=np.float32)

    hsi = HSI(cube, wavelengths)

    assert hsi.dtype == np.float32
    np.testing.assert_allclose(np.asarray(hsi)[0, 0], np.array([0.0, 127.0 / 255.0, 1.0], dtype=np.float32))


def test_constructor_validates_wavelength_shape() -> None:
    cube = np.zeros((4, 5, 3), dtype=np.float32)
    wavelengths = np.array([500.0, 600.0], dtype=np.float32)

    try:
        HSI(cube, wavelengths)
        assert False, "Expected ValueError for wavelength mismatch"
    except ValueError as exc:
        assert "does not match channels" in str(exc)


def test_nearest_band_and_window_average() -> None:
    cube = np.zeros((2, 2, 5), dtype=np.float32)
    for i in range(5):
        cube[:, :, i] = i

    wavelengths = np.array([500, 550, 600, 650, 700], dtype=np.float32)
    hsi = HSI(cube, wavelengths)

    band = hsi.nearest_band(602.0, window=0)
    avg = hsi.nearest_band(602.0, window=1)

    np.testing.assert_allclose(band, np.full((2, 2), 2.0, dtype=np.float32))
    np.testing.assert_allclose(avg, np.full((2, 2), 2.0, dtype=np.float32))


def test_rgb_output_shape_and_range() -> None:
    rng = np.random.default_rng(42)
    cube = rng.random((20, 10, 6), dtype=np.float32)
    wavelengths = np.array([450, 500, 550, 600, 650, 700], dtype=np.float32)

    hsi = HSI(cube, wavelengths)
    rgb = hsi.rgb()

    assert rgb.shape == (20, 10, 3)
    assert rgb.dtype == np.float32
    assert float(np.min(rgb)) >= 0.0
    assert float(np.max(rgb)) <= 1.0


def test_compute_index_maps_numeric_arguments_to_bands() -> None:
    cube = np.zeros((3, 3, 3), dtype=np.float32)
    cube[:, :, 0] = 1.0
    cube[:, :, 1] = 2.0
    cube[:, :, 2] = 3.0
    wavelengths = np.array([500, 600, 700], dtype=np.float32)
    hsi = HSI(cube, wavelengths)

    out = hsi.compute_index(lambda nir, red: (nir - red) / (nir + red + 1e-6), nir=700.0, red=500.0)

    expected = (3.0 - 1.0) / (3.0 + 1.0 + 1e-6)
    np.testing.assert_allclose(out, np.full((3, 3), expected, dtype=np.float32), rtol=1e-6)


def test_resize_and_rescale_keep_float32() -> None:
    cube = np.ones((10, 8, 4), dtype=np.float32)
    wavelengths = np.array([500, 600, 700, 800], dtype=np.float32)
    hsi = HSI(cube, wavelengths)

    resized = hsi.resize(4, 5)
    rescaled = hsi.rescale(0.5)

    assert resized.shape == (5, 4, 4)
    assert resized.dtype == np.float32
    assert rescaled.dtype == np.float32
    np.testing.assert_allclose(np.asarray(rescaled), 0.5)


def test_from_cube_with_explicit_wavelengths() -> None:
    cube = np.ones((6, 4, 3), dtype=np.uint16)
    wl = np.array([500.0, 510.0, 520.0], dtype=np.float32)

    hsi = HSI.from_cube(cube, wavelengths=wl)

    assert hsi.shape == (6, 4, 3)
    assert hsi.dtype == np.float32
    np.testing.assert_allclose(hsi.wavelengths, wl)
    np.testing.assert_allclose(np.max(np.asarray(hsi)), 1.0 / 65535.0)


def test_from_cube_with_min_max_resolution() -> None:
    cube = np.zeros((5, 5, 4), dtype=np.float32)

    hsi = HSI.from_cube(
        cube,
        min_wavelength=500.0,
        max_wavelength=530.0,
        spectral_resolution=10.0,
    )

    np.testing.assert_allclose(hsi.wavelengths, np.array([500.0, 510.0, 520.0, 530.0], dtype=np.float32))


def test_from_cube_with_optional_resolution_uses_default_step() -> None:
    cube = np.zeros((2, 2, 4), dtype=np.float32)

    hsi = HSI.from_cube(
        cube,
        min_wavelength=500.0,
        max_wavelength=540.0,
    )

    np.testing.assert_allclose(hsi.wavelengths, np.array([500.0, 510.0, 520.0, 530.0], dtype=np.float32))


def test_from_cube_rejects_ambiguous_or_incomplete_inputs() -> None:
    cube = np.zeros((2, 2, 3), dtype=np.float32)
    wl = np.array([500.0, 600.0, 700.0], dtype=np.float32)

    try:
        HSI.from_cube(cube, wavelengths=wl, min_wavelength=500.0, max_wavelength=700.0, spectral_resolution=100.0)
        assert False, "Expected ValueError for ambiguous constructor arguments"
    except ValueError as exc:
        assert "either wavelengths" in str(exc)

    try:
        HSI.from_cube(cube, min_wavelength=500.0, max_wavelength=700.0)
    except ValueError:
        assert False, "Did not expect ValueError when only min/max are provided"

    try:
        HSI.from_cube(cube, min_wavelength=500.0)
        assert False, "Expected ValueError for incomplete spectral range arguments"
    except ValueError as exc:
        assert "required" in str(exc)


def test_from_cube_rejects_range_channel_mismatch() -> None:
    cube = np.zeros((2, 2, 3), dtype=np.float32)

    try:
        HSI.from_cube(cube, min_wavelength=500.0, max_wavelength=700.0, spectral_resolution=200.0)
        assert False, "Expected ValueError for range/channel mismatch"
    except ValueError as exc:
        assert "exceed max_wavelength" in str(exc)


def test_align_wavelength_count_returns_exact_length_unchanged() -> None:
    wl = np.array([500.0, 510.0, 520.0], dtype=np.float32)
    out = _align_wavelength_count(wl, 3)
    np.testing.assert_allclose(out, wl)


def test_align_wavelength_count_finds_increasing_segment() -> None:
    wl = np.array([1.0, 42.0, 7.0, 500.0, 510.0, 520.0, 530.0, 9999.0], dtype=np.float32)
    out = _align_wavelength_count(wl, 4)
    np.testing.assert_allclose(out, np.array([500.0, 510.0, 520.0, 530.0], dtype=np.float32))


def test_parse_shape_from_text_json() -> None:
    shape = _parse_shape_from_text('{"shape": [256, 256, 300]}')
    assert shape == (256, 256, 300)


def test_reorder_hwc_to_target_shape_permutes_axes() -> None:
    cube = np.zeros((300, 256, 256), dtype=np.float32)
    out = _reorder_hwc_to_target_shape(cube, (256, 256, 300))
    assert out.shape == (256, 256, 300)


def test_canonicalize_reference_for_unit_cube_scales_255_range() -> None:
    cube = np.random.default_rng(0).random((4, 4, 3), dtype=np.float32)
    ref = np.array([255.0, 128.0, 64.0], dtype=np.float32)
    out = _canonicalize_reference_for_unit_cube(ref, cube)
    np.testing.assert_allclose(out, np.array([1.0, 128.0 / 255.0, 64.0 / 255.0], dtype=np.float32), rtol=1e-6)


def test_normalize_cube_with_reference_bandwise() -> None:
    cube = np.ones((2, 2, 3), dtype=np.float32)
    ref = np.array([1.0, 2.0, 4.0], dtype=np.float32)
    out = _normalize_cube_with_reference(cube, ref)
    np.testing.assert_allclose(out[:, :, 0], 1.0)
    np.testing.assert_allclose(out[:, :, 1], 0.5)
    np.testing.assert_allclose(out[:, :, 2], 0.25)


def test_parse_payload_dict_xml_with_reference_metadata() -> None:
    xml = (
        "<hybrint_hsi_metadata>"
        "<shape>[256, 256, 300]</shape>"
        "<reference_spectrum>[1.0, 2.0, 3.0]</reference_spectrum>"
        "<reference_multiplier>1.0</reference_multiplier>"
        "<reference_eps>1e-8</reference_eps>"
        "</hybrint_hsi_metadata>"
    )
    parsed = _parse_payload_dict(xml)
    assert parsed is not None
    assert parsed["shape"] == [256, 256, 300]
    assert parsed["reference_spectrum"] == [1.0, 2.0, 3.0]


def test_filter_savgol_matches_vectorized_scipy() -> None:
    rng = np.random.default_rng(123)
    cube = rng.random((6, 5, 41), dtype=np.float32)
    wavelengths = np.linspace(400.0, 800.0, 41, dtype=np.float32)
    hsi = HSI(cube, wavelengths)

    out = hsi.filter_savgol(window_length=31, polyorder=3)
    expected = savgol_filter(cube, window_length=31, polyorder=3, axis=2, mode="interp").astype(np.float32)

    assert out.shape == hsi.shape
    assert out.dtype == np.float32
    np.testing.assert_allclose(np.asarray(out), expected, rtol=1e-6, atol=1e-6)


def test_filter_savgol_validates_window_length() -> None:
    cube = np.ones((2, 2, 9), dtype=np.float32)
    wavelengths = np.linspace(500.0, 580.0, 9, dtype=np.float32)
    hsi = HSI(cube, wavelengths)

    try:
        hsi.filter_savgol(window_length=10, polyorder=3)
        assert False, "Expected ValueError for even window_length"
    except ValueError as exc:
        assert "odd" in str(exc)
