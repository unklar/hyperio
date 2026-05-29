"""Tests for JSON sidecar metadata (metadata_json=True)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hyperio import HSI
from hyperio.io import (
    read_auto,
    write_auto,
    _sidecar_json_path,
)

TEST_IMAGES = Path(__file__).resolve().parents[1] / "test_images"
JP2_PATH = TEST_IMAGES / "hsi_000.jp2"


@pytest.fixture()
def source_hsi() -> HSI:
    pytest.importorskip("rasterio")
    if not JP2_PATH.exists():
        pytest.skip("JP2 test image not available")
    return HSI.read(str(JP2_PATH))


class TestSidecarPath:
    def test_sidecar_path_for_hdr(self) -> None:
        p = _sidecar_json_path(Path("/data/cube.hdr"))
        assert p == Path("/data/cube.json")

    def test_sidecar_path_for_tiff(self) -> None:
        p = _sidecar_json_path(Path("/data/cube.tiff"))
        assert p == Path("/data/cube.json")

    def test_sidecar_path_for_jp2(self) -> None:
        p = _sidecar_json_path(Path("/data/cube.jp2"))
        assert p == Path("/data/cube.json")

    def test_sidecar_path_for_hsd(self) -> None:
        p = _sidecar_json_path(Path("/data/cube.hsd"))
        assert p == Path("/data/cube.json")

    def test_sidecar_path_for_directory(self) -> None:
        p = _sidecar_json_path(Path("/data/png_folder"))
        assert p == Path("/data/png_folder/metadata.json")


class TestWriteSidecar:
    def test_envi_creates_sidecar(self, source_hsi: HSI, tmp_path: Path) -> None:
        source_hsi.write(str(tmp_path / "cube.hdr"), metadata_json=True)
        sidecar = tmp_path / "cube.json"
        assert sidecar.exists()
        meta = json.loads(sidecar.read_text())
        assert "wavelengths" in meta
        np.testing.assert_allclose(meta["wavelengths"], source_hsi.wavelengths, rtol=1e-6)

    def test_tiff_creates_sidecar(self, source_hsi: HSI, tmp_path: Path) -> None:
        source_hsi.write(str(tmp_path / "cube.tiff"), metadata_json=True)
        sidecar = tmp_path / "cube.json"
        assert sidecar.exists()
        meta = json.loads(sidecar.read_text())
        assert "wavelengths" in meta
        np.testing.assert_allclose(meta["wavelengths"], source_hsi.wavelengths, rtol=1e-6)

    def test_hsd_creates_sidecar(self, source_hsi: HSI, tmp_path: Path) -> None:
        source_hsi.write(str(tmp_path / "cube.hsd"), metadata_json=True)
        sidecar = tmp_path / "cube.json"
        assert sidecar.exists()
        meta = json.loads(sidecar.read_text())
        assert "wavelengths" in meta
        np.testing.assert_allclose(meta["wavelengths"], source_hsi.wavelengths, rtol=1e-6)

    def test_jp2_creates_sidecar(self, source_hsi: HSI, tmp_path: Path) -> None:
        source_hsi.write(str(tmp_path / "cube.jp2"), metadata_json=True)
        sidecar = tmp_path / "cube.json"
        assert sidecar.exists()
        meta = json.loads(sidecar.read_text())
        assert "wavelengths" in meta
        np.testing.assert_allclose(meta["wavelengths"], source_hsi.wavelengths, rtol=1e-6)

    def test_png_folder_creates_sidecar(self, source_hsi: HSI, tmp_path: Path) -> None:
        folder = tmp_path / "png_out"
        source_hsi.write(str(folder), metadata_json=True, line_cam=True)
        sidecar = folder / "metadata.json"
        assert sidecar.exists()
        meta = json.loads(sidecar.read_text())
        assert "wavelengths" in meta
        np.testing.assert_allclose(meta["wavelengths"], source_hsi.wavelengths, rtol=1e-6)

    def test_no_sidecar_by_default(self, source_hsi: HSI, tmp_path: Path) -> None:
        source_hsi.write(str(tmp_path / "cube.hdr"))
        sidecar = tmp_path / "cube.json"
        assert not sidecar.exists()


class TestReadSidecar:
    def test_envi_sidecar_wavelengths(self, source_hsi: HSI, tmp_path: Path) -> None:
        source_hsi.write(str(tmp_path / "cube.hdr"), metadata_json=True)
        hsi_back = HSI.read(str(tmp_path / "cube.hdr"), metadata_json=True)
        np.testing.assert_allclose(hsi_back.wavelengths, source_hsi.wavelengths, rtol=1e-6)

    def test_tiff_sidecar_wavelengths(self, source_hsi: HSI, tmp_path: Path) -> None:
        source_hsi.write(str(tmp_path / "cube.tiff"), metadata_json=True)
        hsi_back = HSI.read(str(tmp_path / "cube.tiff"), metadata_json=True)
        np.testing.assert_allclose(hsi_back.wavelengths, source_hsi.wavelengths, rtol=1e-6)

    def test_hsd_sidecar_wavelengths_exact(self, source_hsi: HSI, tmp_path: Path) -> None:
        source_hsi.write(str(tmp_path / "cube.hsd"), metadata_json=True)
        hsi_back = HSI.read(str(tmp_path / "cube.hsd"), metadata_json=True)
        np.testing.assert_allclose(hsi_back.wavelengths, source_hsi.wavelengths, rtol=1e-6)

    def test_png_folder_sidecar_wavelengths(self, source_hsi: HSI, tmp_path: Path) -> None:
        folder = tmp_path / "png_out"
        source_hsi.write(str(folder), metadata_json=True, line_cam=True)
        hsi_back = HSI.read(str(folder), metadata_json=True, line_cam=True)
        np.testing.assert_allclose(hsi_back.wavelengths, source_hsi.wavelengths, rtol=1e-6)

    def test_sidecar_missing_falls_back_to_embedded(self, source_hsi: HSI, tmp_path: Path) -> None:
        source_hsi.write(str(tmp_path / "cube.hdr"))
        sidecar = tmp_path / "cube.json"
        assert not sidecar.exists()
        hsi_back = HSI.read(str(tmp_path / "cube.hdr"), metadata_json=True)
        np.testing.assert_allclose(hsi_back.wavelengths, source_hsi.wavelengths, atol=1e-3)


class TestSidecarWithReferenceSpectrum:
    def test_write_with_reference_spectrum(self, source_hsi: HSI, tmp_path: Path) -> None:
        ref = np.ones(source_hsi.shape[2], dtype=np.float32) * 0.5
        write_auto(
            source_hsi.cube,
            source_hsi.wavelengths,
            str(tmp_path / "cube.hdr"),
            metadata_json=True,
            reference_spectrum=ref,
            reference_multiplier=2.0,
            reference_eps=1e-6,
        )
        sidecar = tmp_path / "cube.json"
        meta = json.loads(sidecar.read_text())
        assert "reference_spectrum" in meta
        np.testing.assert_allclose(meta["reference_spectrum"], ref, rtol=1e-6)
        assert meta["reference_multiplier"] == 2.0
        assert abs(meta["reference_eps"] - 1e-6) < 1e-12

    def test_read_with_reference_spectrum(self, source_hsi: HSI, tmp_path: Path) -> None:
        ref = np.ones(source_hsi.shape[2], dtype=np.float32) * 0.5
        write_auto(
            source_hsi.cube,
            source_hsi.wavelengths,
            str(tmp_path / "cube.hdr"),
            metadata_json=True,
            reference_spectrum=ref,
            reference_multiplier=2.0,
            reference_eps=1e-6,
        )
        result = read_auto(str(tmp_path / "cube.hdr"), metadata_json=True)
        assert result.reference_spectrum is not None
        np.testing.assert_allclose(result.reference_spectrum, ref, rtol=1e-6)
        assert abs(result.reference_multiplier - 2.0) < 1e-8
        assert abs(result.reference_eps - 1e-6) < 1e-12

    def test_default_reference_fields_omitted(self, tmp_path: Path) -> None:
        cube = np.random.rand(10, 10, 5).astype(np.float32)
        wl = np.array([400.0, 500.0, 600.0, 700.0, 800.0], dtype=np.float32)
        hsi = HSI(cube, wl)
        hsi.write(str(tmp_path / "cube.hdr"), metadata_json=True)
        meta = json.loads((tmp_path / "cube.json").read_text())
        assert "reference_spectrum" not in meta
        assert "reference_multiplier" not in meta
        assert "reference_eps" not in meta


class TestHsdSidecarExactWavelengths:
    def test_hsd_sidecar_gives_exact_wavelengths(self, source_hsi: HSI, tmp_path: Path) -> None:
        source_hsi.write(str(tmp_path / "cube.hsd"), metadata_json=True)
        hsi_no_json = HSI.read(str(tmp_path / "cube.hsd"))
        hsi_with_json = HSI.read(str(tmp_path / "cube.hsd"), metadata_json=True)

        # Without JSON: HSD only stores integer endpoints → approximate
        # With JSON: full wavelength vector is preserved → exact
        with_json_diff = np.max(np.abs(hsi_with_json.wavelengths - source_hsi.wavelengths))
        without_json_diff = np.max(np.abs(hsi_no_json.wavelengths - source_hsi.wavelengths))
        assert with_json_diff < without_json_diff
        np.testing.assert_allclose(hsi_with_json.wavelengths, source_hsi.wavelengths, rtol=1e-6)


class TestPngFolderSidecarNoExplicitWavelengthsNeeded:
    def test_png_folder_with_sidecar_no_explicit_wavelengths(
        self, source_hsi: HSI, tmp_path: Path
    ) -> None:
        folder = tmp_path / "png_out"
        source_hsi.write(str(folder), metadata_json=True, line_cam=True)

        # Without metadata_json, this would require explicit wavelengths
        hsi_back = HSI.read(str(folder), metadata_json=True, line_cam=True)
        assert hsi_back.shape == source_hsi.shape
        np.testing.assert_allclose(hsi_back.wavelengths, source_hsi.wavelengths, rtol=1e-6)
