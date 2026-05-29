"""Equivalence tests: verify that hyperio produces identical output to use_hsi_io."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

from use_hsi_io import HSI as OldHSI
from use_hsi_io.io import (
    _align_wavelength_count as old_align,
    _canonicalize_reference_for_unit_cube as old_canonicalize,
    _normalize_cube_with_reference as old_normalize_ref,
    _parse_payload_dict as old_parse_payload,
    _parse_shape_from_text as old_parse_shape,
    _reorder_hwc_to_target_shape as old_reorder,
    read_hsd as old_read_hsd,
    read_line_scan_png_folder as old_read_png_folder,
)

from hyperio import HSI as NewHSI
from hyperio.io import (
    _align_wavelength_count as new_align,
    _canonicalize_reference_for_unit_cube as new_canonicalize,
    _normalize_cube_with_reference as new_normalize_ref,
    _parse_payload_dict as new_parse_payload,
    _parse_shape_from_text as new_parse_shape,
    _reorder_hwc_to_target_shape as new_reorder,
    read_hsd as new_read_hsd,
    read_line_scan_png_folder as new_read_png_folder,
)


def _make_hsi_data():
    rng = np.random.default_rng(42)
    cube_float = rng.random((8, 6, 10), dtype=np.float32)
    cube_int = rng.integers(0, 256, size=(8, 6, 10), dtype=np.uint8)
    wavelengths = np.linspace(400.0, 900.0, 10, dtype=np.float32)
    return cube_float, cube_int, wavelengths


def test_constructor_float_cube_identical():
    cube_float, _, wavelengths = _make_hsi_data()
    old = OldHSI(cube_float.copy(), wavelengths.copy())
    new = NewHSI(cube_float.copy(), wavelengths.copy())
    np.testing.assert_array_equal(np.asarray(old), np.asarray(new))
    np.testing.assert_array_equal(old.wavelengths, new.wavelengths)
    assert old.shape == new.shape
    assert old.dtype == new.dtype


def test_constructor_integer_cube_identical():
    _, cube_int, wavelengths = _make_hsi_data()
    old = OldHSI(cube_int.copy(), wavelengths.copy())
    new = NewHSI(cube_int.copy(), wavelengths.copy())
    np.testing.assert_allclose(np.asarray(old), np.asarray(new), rtol=1e-7)


def test_from_cube_explicit_wavelengths_identical():
    cube_float, _, wavelengths = _make_hsi_data()
    old = OldHSI.from_cube(cube_float.copy(), wavelengths=wavelengths.copy())
    new = NewHSI.from_cube(cube_float.copy(), wavelengths=wavelengths.copy())
    np.testing.assert_array_equal(np.asarray(old), np.asarray(new))
    np.testing.assert_array_equal(old.wavelengths, new.wavelengths)


def test_from_cube_min_max_identical():
    cube_float, _, _ = _make_hsi_data()
    old = OldHSI.from_cube(cube_float.copy(), min_wavelength=400.0, max_wavelength=900.0)
    new = NewHSI.from_cube(cube_float.copy(), min_wavelength=400.0, max_wavelength=900.0)
    np.testing.assert_array_equal(np.asarray(old), np.asarray(new))
    np.testing.assert_array_equal(old.wavelengths, new.wavelengths)


def test_nearest_band_identical():
    cube_float, _, wavelengths = _make_hsi_data()
    old = OldHSI(cube_float.copy(), wavelengths.copy())
    new = NewHSI(cube_float.copy(), wavelengths.copy())
    for wl in [450.0, 650.0, 890.0]:
        np.testing.assert_array_equal(old.nearest_band(wl), new.nearest_band(wl))
        np.testing.assert_array_equal(old.nearest_band(wl, window=1), new.nearest_band(wl, window=1))


def test_bands_from_wavelengths_identical():
    cube_float, _, wavelengths = _make_hsi_data()
    old = OldHSI(cube_float.copy(), wavelengths.copy())
    new = NewHSI(cube_float.copy(), wavelengths.copy())
    np.testing.assert_allclose(
        old.bands_from_wavelengths([450.0, 600.0, 800.0]),
        new.bands_from_wavelengths([450.0, 600.0, 800.0]),
        rtol=1e-7,
    )


def test_compute_index_identical():
    cube_float, _, wavelengths = _make_hsi_data()
    old = OldHSI(cube_float.copy(), wavelengths.copy())
    new = NewHSI(cube_float.copy(), wavelengths.copy())
    old_idx = old.compute_index(lambda nir, red: (nir - red) / (nir + red + 1e-6), nir=800.0, red=500.0)
    new_idx = new.compute_index(lambda nir, red: (nir - red) / (nir + red + 1e-6), nir=800.0, red=500.0)
    np.testing.assert_allclose(old_idx, new_idx, rtol=1e-6)


def test_rgb_identical():
    cube_float, _, wavelengths = _make_hsi_data()
    old = OldHSI(cube_float.copy(), wavelengths.copy())
    new = NewHSI(cube_float.copy(), wavelengths.copy())
    np.testing.assert_allclose(old.rgb(), new.rgb(), rtol=1e-6)


def test_resize_identical():
    cube_float, _, wavelengths = _make_hsi_data()
    old = OldHSI(cube_float.copy(), wavelengths.copy())
    new = NewHSI(cube_float.copy(), wavelengths.copy())
    old_resized = old.resize(4, 3)
    new_resized = new.resize(4, 3)
    np.testing.assert_allclose(np.asarray(old_resized), np.asarray(new_resized), rtol=1e-6)
    np.testing.assert_array_equal(old_resized.wavelengths, new_resized.wavelengths)


def test_rescale_identical():
    cube_float, _, wavelengths = _make_hsi_data()
    old = OldHSI(cube_float.copy(), wavelengths.copy())
    new = NewHSI(cube_float.copy(), wavelengths.copy())
    np.testing.assert_allclose(np.asarray(old.rescale(0.5)), np.asarray(new.rescale(0.5)), rtol=1e-7)


def test_filter_savgol_identical():
    cube_float, _, wavelengths = _make_hsi_data()
    old = OldHSI(cube_float.copy(), wavelengths.copy())
    new = NewHSI(cube_float.copy(), wavelengths.copy())
    old_filtered = old.filter_savgol(window_length=7, polyorder=3)
    new_filtered = new.filter_savgol(window_length=7, polyorder=3)
    np.testing.assert_allclose(np.asarray(old_filtered), np.asarray(new_filtered), rtol=1e-6)


def test_align_wavelength_count_identical():
    wl = np.array([1.0, 42.0, 7.0, 500.0, 510.0, 520.0, 530.0, 9999.0], dtype=np.float32)
    np.testing.assert_array_equal(old_align(wl.copy(), 4), new_align(wl.copy(), 4))


def test_normalize_cube_with_reference_identical():
    cube = np.ones((2, 2, 3), dtype=np.float32)
    ref = np.array([1.0, 2.0, 4.0], dtype=np.float32)
    np.testing.assert_allclose(old_normalize_ref(cube.copy(), ref.copy()), new_normalize_ref(cube.copy(), ref.copy()))


def test_canonicalize_reference_identical():
    rng = np.random.default_rng(0)
    cube = rng.random((4, 4, 3), dtype=np.float32)
    ref = np.array([255.0, 128.0, 64.0], dtype=np.float32)
    np.testing.assert_allclose(old_canonicalize(ref.copy(), cube.copy()), new_canonicalize(ref.copy(), cube.copy()))


def test_parse_payload_dict_identical():
    xml = (
        "<hybrint_hsi_metadata>"
        "<shape>[256, 256, 300]</shape>"
        "<reference_spectrum>[1.0, 2.0, 3.0]</reference_spectrum>"
        "</hybrint_hsi_metadata>"
    )
    assert old_parse_payload(xml) == new_parse_payload(xml)


def test_parse_shape_from_text_identical():
    text = '{"shape": [256, 256, 300]}'
    assert old_parse_shape(text) == new_parse_shape(text)


def test_reorder_hwc_to_target_shape_identical():
    cube = np.zeros((300, 256, 256), dtype=np.float32)
    np.testing.assert_array_equal(
        old_reorder(cube.copy(), (256, 256, 300)),
        new_reorder(cube.copy(), (256, 256, 300)),
    )


def _write_synthetic_hsd(path: Path, *, include_stepw: bool = True) -> tuple[np.ndarray, np.ndarray]:
    height, width, sr, d = 2, 3, 4, 2
    startw, endw = 500, 800
    average = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    coeff = np.array([[1.0, 0.5, 0.25, 0.125], [0.0, 1.0, 0.5, 0.25]], dtype=np.float32)
    scoredata = np.array([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [2.0, 1.0], [1.0, 2.0], [0.5, 1.5]], dtype=np.float32)
    expected_cube = (np.dot(scoredata, coeff) + average).reshape((height, width, sr)).astype(np.float32)
    expected_wl = np.linspace(startw, endw, sr, dtype=np.float32)
    header = np.array([height, width, sr, d, startw, 0, endw], dtype=np.int32)
    body_parts = []
    if include_stepw:
        body_parts.append(np.array([100.0], dtype=np.float32))
    body_parts.extend([average, coeff.reshape(-1), scoredata.reshape(-1)])
    body = np.concatenate(body_parts).astype(np.float32)
    with path.open("wb") as f:
        header.tofile(f)
        body.tofile(f)
    return expected_cube, expected_wl


def test_read_hsd_identical(tmp_path: Path):
    path = tmp_path / "synthetic.hsd"
    _write_synthetic_hsd(path)
    old_result = old_read_hsd(path)
    new_result = new_read_hsd(path)
    np.testing.assert_allclose(old_result.cube, new_result.cube, rtol=1e-6)
    np.testing.assert_allclose(old_result.wavelengths, new_result.wavelengths, rtol=1e-6)


def test_read_hsd_no_stepw_identical(tmp_path: Path):
    path = tmp_path / "synthetic_no_stepw.hsd"
    _write_synthetic_hsd(path, include_stepw=False)
    old_result = old_read_hsd(path)
    new_result = new_read_hsd(path)
    np.testing.assert_allclose(old_result.cube, new_result.cube, rtol=1e-6)
    np.testing.assert_allclose(old_result.wavelengths, new_result.wavelengths, rtol=1e-6)


def test_read_hsd_explicit_wavelengths_identical(tmp_path: Path):
    path = tmp_path / "synthetic_override.hsd"
    _write_synthetic_hsd(path)
    override = np.array([410.0, 420.0, 430.0, 440.0], dtype=np.float32)
    old_result = old_read_hsd(path, wavelengths=override.copy())
    new_result = new_read_hsd(path, wavelengths=override.copy())
    np.testing.assert_allclose(old_result.cube, new_result.cube, rtol=1e-6)
    np.testing.assert_array_equal(old_result.wavelengths, new_result.wavelengths)


def test_read_line_scan_png_folder_identical(tmp_path: Path):
    folder = tmp_path / "line_scan"
    folder.mkdir()
    h, c = 4, 5
    rng = np.random.default_rng(99)
    for i in range(3):
        line = rng.integers(0, 65535, size=(h, c), dtype=np.uint16)
        Image.fromarray(line).save(folder / f"{i:05d}.png")

    wl = np.linspace(500.0, 700.0, c, dtype=np.float32)
    old_result = old_read_png_folder(folder, wavelengths=wl.copy())
    new_result = new_read_png_folder(folder, wavelengths=wl.copy())
    np.testing.assert_allclose(old_result.cube, new_result.cube, rtol=1e-6)
    np.testing.assert_array_equal(old_result.wavelengths, new_result.wavelengths)


def test_read_channel_png_folder_identical(tmp_path: Path):
    folder = tmp_path / "channel_stack"
    folder.mkdir()
    h, w, c = 3, 4, 3
    rng = np.random.default_rng(77)
    for i in range(c):
        ch = rng.integers(0, 65535, size=(h, w), dtype=np.uint16)
        Image.fromarray(ch).save(folder / f"scene_{i + 1}.png")

    wl = np.linspace(500.0, 700.0, c, dtype=np.float32)
    old_result = old_read_png_folder(folder, wavelengths=wl.copy(), line_cam=False)
    new_result = new_read_png_folder(folder, wavelengths=wl.copy(), line_cam=False)
    np.testing.assert_allclose(old_result.cube, new_result.cube, rtol=1e-6)
    np.testing.assert_array_equal(old_result.wavelengths, new_result.wavelengths)


def test_copy_identical():
    cube_float, _, wavelengths = _make_hsi_data()
    old = OldHSI(cube_float.copy(), wavelengths.copy())
    new = NewHSI(cube_float.copy(), wavelengths.copy())
    old_copy = old.copy()
    new_copy = new.copy()
    np.testing.assert_array_equal(np.asarray(old_copy), np.asarray(new_copy))
    np.testing.assert_array_equal(old_copy.wavelengths, new_copy.wavelengths)


def test_repr_same_structure():
    cube_float, _, wavelengths = _make_hsi_data()
    old = OldHSI(cube_float.copy(), wavelengths.copy())
    new = NewHSI(cube_float.copy(), wavelengths.copy())
    assert str(old) == str(new)
