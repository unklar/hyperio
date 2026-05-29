from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from hyperio import HSI


TEST_IMAGES = Path(__file__).resolve().parents[1] / "test_images"


def _write_synthetic_hsd(path: Path, *, include_stepw: bool = True) -> tuple[np.ndarray, np.ndarray]:
    height, width, sr, d = 2, 3, 4, 2
    startw, endw = 500, 800

    average = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    coeff = np.array(
        [
            [1.0, 0.5, 0.25, 0.125],
            [0.0, 1.0, 0.5, 0.25],
        ],
        dtype=np.float32,
    )
    scoredata = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [2.0, 1.0],
            [1.0, 2.0],
            [0.5, 1.5],
        ],
        dtype=np.float32,
    )

    expected_cube = (np.dot(scoredata, coeff) + average).reshape((height, width, sr)).astype(np.float32)
    expected_wl = np.linspace(startw, endw, sr, dtype=np.float32)

    header = np.array([height, width, sr, d, startw, 0, endw], dtype=np.int32)
    body_parts = []
    if include_stepw:
        body_parts.append(np.array([100.0], dtype=np.float32))
    body_parts.extend(
        [
            average,
            coeff.reshape(-1),
            scoredata.reshape(-1),
        ]
    )
    body = np.concatenate(body_parts).astype(np.float32)

    with path.open("wb") as f:
        header.tofile(f)
        body.tofile(f)

    return expected_cube, expected_wl


@pytest.mark.skipif(
    not (TEST_IMAGES / "hsi_000.jp2").exists(),
    reason="JP2 test image not available",
)
def test_read_jp2_with_explicit_wavelengths() -> None:
    path = TEST_IMAGES / "hsi_000.jp2"

    for c in (3, 64, 128, 256, 274, 300):
        wl = np.linspace(400.0, 1000.0, c, dtype=np.float32)
        try:
            hsi = HSI.read(path, wavelengths=wl)
            assert hsi.wavelengths.shape[0] == hsi.shape[2]
            assert hsi.dtype == np.float32
            return
        except ValueError:
            continue

    raise AssertionError("Could not determine JP2 channel count for explicit wavelengths test")


@pytest.mark.skipif(
    not (TEST_IMAGES / "hsi_000.jp2").exists(),
    reason="JP2 test image not available",
)
def test_read_jp2_requires_wavelengths_or_embedded_metadata() -> None:
    path = TEST_IMAGES / "hsi_000.jp2"
    try:
        hsi = HSI.read(path)
        assert hsi.wavelengths.shape[0] == hsi.shape[2]
    except ValueError as exc:
        assert "wavelength" in str(exc).lower()


def test_read_hsd_reconstructs_cube_and_wavelengths(tmp_path: Path) -> None:
    path = tmp_path / "synthetic.hsd"
    expected_cube, expected_wl = _write_synthetic_hsd(path)

    hsi = HSI.read(path)

    assert hsi.shape == expected_cube.shape
    assert hsi.dtype == np.float32
    np.testing.assert_allclose(np.asarray(hsi), expected_cube, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(hsi.wavelengths, expected_wl, rtol=1e-6, atol=1e-6)


def test_read_hsd_accepts_explicit_wavelength_override(tmp_path: Path) -> None:
    path = tmp_path / "synthetic_override.hsd"
    expected_cube, _ = _write_synthetic_hsd(path)
    override = np.array([410.0, 420.0, 430.0, 440.0], dtype=np.float32)

    hsi = HSI.read(path, wavelengths=override)

    np.testing.assert_allclose(np.asarray(hsi), expected_cube, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(hsi.wavelengths, override)


def test_read_hsd_without_stepw_payload_is_supported(tmp_path: Path) -> None:
    path = tmp_path / "synthetic_no_stepw.hsd"
    expected_cube, expected_wl = _write_synthetic_hsd(path, include_stepw=False)

    hsi = HSI.read(path)

    np.testing.assert_allclose(np.asarray(hsi), expected_cube, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(hsi.wavelengths, expected_wl, rtol=1e-6, atol=1e-6)


def test_read_line_scan_png_folder_stacks_lines_in_numeric_order(tmp_path: Path) -> None:
    folder = tmp_path / "line_scan"
    folder.mkdir()

    h, c = 4, 5
    line_1 = np.arange(h * c, dtype=np.uint16).reshape(h, c)
    line_2 = (line_1 + 100).astype(np.uint16)
    line_3 = (line_1 + 200).astype(np.uint16)

    Image.fromarray(line_1).save(folder / "00002.png")
    Image.fromarray(line_2).save(folder / "00003.png")
    Image.fromarray(line_3).save(folder / "00004.png")

    wl = np.linspace(500.0, 700.0, c, dtype=np.float32)
    hsi = HSI.read(folder, wavelengths=wl)

    assert hsi.shape == (h, 3, c)
    assert hsi.dtype == np.float32
    np.testing.assert_allclose(hsi.wavelengths, wl)

    expected = np.stack([line_1, line_2, line_3], axis=1).astype(np.float32) / 65535.0
    np.testing.assert_allclose(np.asarray(hsi), expected, rtol=1e-6, atol=1e-6)


def test_read_line_scan_png_folder_requires_wavelengths(tmp_path: Path) -> None:
    folder = tmp_path / "line_scan_missing_wl"
    folder.mkdir()

    line = np.arange(12, dtype=np.uint16).reshape(3, 4)
    Image.fromarray(line).save(folder / "00001.png")

    with pytest.raises(ValueError, match="[Ww]avelength"):
        HSI.read(folder)


def test_read_line_scan_png_folder_transposes_when_spectral_axis_is_first(tmp_path: Path) -> None:
    folder = tmp_path / "line_scan_transposed"
    folder.mkdir()

    h, c = 6, 4
    base = np.arange(h * c, dtype=np.uint16).reshape(h, c)
    line_1 = base.T
    line_2 = (base + 50).T

    Image.fromarray(line_1).save(folder / "00010.png")
    Image.fromarray(line_2).save(folder / "00011.png")

    wl = np.linspace(500.0, 650.0, c, dtype=np.float32)
    hsi = HSI.read(folder, wavelengths=wl)

    assert hsi.shape == (h, 2, c)
    expected = np.stack([base, base + 50], axis=1).astype(np.float32) / 65535.0
    np.testing.assert_allclose(np.asarray(hsi), expected, rtol=1e-6, atol=1e-6)


def test_read_channel_png_folder_with_suffix_index_names(tmp_path: Path) -> None:
    folder = tmp_path / "channel_stack_suffix"
    folder.mkdir()

    h, w, c = 3, 4, 3
    ch0 = np.arange(h * w, dtype=np.uint16).reshape(h, w)
    ch1 = (ch0 + 100).astype(np.uint16)
    ch2 = (ch0 + 200).astype(np.uint16)

    Image.fromarray(ch0).save(folder / "scene_1.png")
    Image.fromarray(ch1).save(folder / "scene_2.png")
    Image.fromarray(ch2).save(folder / "scene_3.png")

    wl = np.linspace(500.0, 700.0, c, dtype=np.float32)
    hsi = HSI.read(folder, wavelengths=wl, line_cam=False)

    assert hsi.shape == (h, w, c)
    expected = np.stack([ch0, ch1, ch2], axis=2).astype(np.float32) / 65535.0
    np.testing.assert_allclose(np.asarray(hsi), expected, rtol=1e-6, atol=1e-6)


def test_read_channel_png_folder_with_numeric_names(tmp_path: Path) -> None:
    folder = tmp_path / "channel_stack_numeric"
    folder.mkdir()

    h, w, c = 2, 5, 2
    ch0 = np.arange(h * w, dtype=np.uint16).reshape(h, w)
    ch1 = (ch0 + 50).astype(np.uint16)

    Image.fromarray(ch0).save(folder / "00000.png")
    Image.fromarray(ch1).save(folder / "00001.png")

    wl = np.linspace(600.0, 700.0, c, dtype=np.float32)
    hsi = HSI.read(folder, wavelengths=wl, line_cam=False)

    assert hsi.shape == (h, w, c)
    expected = np.stack([ch0, ch1], axis=2).astype(np.float32) / 65535.0
    np.testing.assert_allclose(np.asarray(hsi), expected, rtol=1e-6, atol=1e-6)


def test_read_channel_png_folder_with_min_max_wavelength(tmp_path: Path) -> None:
    folder = tmp_path / "channel_stack_min_max"
    folder.mkdir()

    h, w, c = 3, 3, 3
    ch0 = np.arange(h * w, dtype=np.uint16).reshape(h, w)
    ch1 = (ch0 + 10).astype(np.uint16)
    ch2 = (ch0 + 20).astype(np.uint16)

    Image.fromarray(ch0).save(folder / "cube_1.png")
    Image.fromarray(ch1).save(folder / "cube_2.png")
    Image.fromarray(ch2).save(folder / "cube_3.png")

    hsi = HSI.read(folder, min_wavelength=500.0, max_wavelength=700.0, line_cam=False)

    np.testing.assert_allclose(hsi.wavelengths, np.array([500.0, 600.0, 700.0], dtype=np.float32))


def test_read_rejects_both_explicit_and_min_max_wavelengths(tmp_path: Path) -> None:
    folder = tmp_path / "channel_stack_conflict"
    folder.mkdir()
    arr = np.arange(9, dtype=np.uint16).reshape(3, 3)
    Image.fromarray(arr).save(folder / "00000.png")

    with pytest.raises(ValueError, match="either wavelengths or min/max"):
        HSI.read(
            folder,
            wavelengths=np.array([500.0], dtype=np.float32),
            min_wavelength=500.0,
            max_wavelength=600.0,
            line_cam=False,
        )


def test_read_rejects_incomplete_min_max_wavelengths(tmp_path: Path) -> None:
    folder = tmp_path / "channel_stack_incomplete_bounds"
    folder.mkdir()
    arr = np.arange(9, dtype=np.uint16).reshape(3, 3)
    Image.fromarray(arr).save(folder / "00000.png")

    with pytest.raises(ValueError, match="Both min_wavelength and max_wavelength"):
        HSI.read(folder, min_wavelength=500.0, line_cam=False)
