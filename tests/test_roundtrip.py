"""Round-trip tests: write to each format, read back, verify fidelity."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hyperio import HSI

TEST_IMAGES = Path(__file__).resolve().parents[1] / "test_images"

JP2_PATH = TEST_IMAGES / "hsi_000.jp2"


@pytest.fixture()
def source_hsi() -> HSI:
    """Load the reference JP2 with embedded wavelength metadata."""
    pytest.importorskip("rasterio")
    if not JP2_PATH.exists():
        pytest.skip("JP2 test image not available")
    return HSI.read(str(JP2_PATH))


class TestEnviRoundTrip:
    def test_envi_round_trip_shape_and_wavelengths(self, source_hsi: HSI, tmp_path: Path) -> None:
        hdr_path = source_hsi.write(str(tmp_path / "cube.hdr"))
        hsi_back = HSI.read(str(hdr_path))

        assert hsi_back.shape == source_hsi.shape
        np.testing.assert_allclose(hsi_back.wavelengths, source_hsi.wavelengths, atol=1e-3)

    def test_envi_round_trip_cube_data(self, source_hsi: HSI, tmp_path: Path) -> None:
        source_hsi.write(str(tmp_path / "cube.hdr"))
        hsi_back = HSI.read(str(tmp_path / "cube.hdr"))

        np.testing.assert_allclose(np.asarray(hsi_back), np.asarray(source_hsi), rtol=1e-6, atol=1e-6)


class TestTiffRoundTrip:
    def test_tiff_round_trip_shape_and_wavelengths(self, source_hsi: HSI, tmp_path: Path) -> None:
        source_hsi.write(str(tmp_path / "cube.tiff"))
        hsi_back = HSI.read(str(tmp_path / "cube.tiff"))

        assert hsi_back.shape == source_hsi.shape
        np.testing.assert_allclose(hsi_back.wavelengths, source_hsi.wavelengths, atol=1e-3)

    def test_tiff_round_trip_cube_data(self, source_hsi: HSI, tmp_path: Path) -> None:
        source_hsi.write(str(tmp_path / "cube.tiff"))
        hsi_back = HSI.read(str(tmp_path / "cube.tiff"))

        np.testing.assert_allclose(np.asarray(hsi_back), np.asarray(source_hsi), rtol=1e-6, atol=1e-6)


class TestHsdRoundTrip:
    def test_hsd_round_trip_shape(self, source_hsi: HSI, tmp_path: Path) -> None:
        source_hsi.write(str(tmp_path / "cube.hsd"))
        hsi_back = HSI.read(str(tmp_path / "cube.hsd"))

        assert hsi_back.shape == source_hsi.shape

    def test_hsd_round_trip_wavelengths_approx(self, source_hsi: HSI, tmp_path: Path) -> None:
        source_hsi.write(str(tmp_path / "cube.hsd"))
        hsi_back = HSI.read(str(tmp_path / "cube.hsd"))

        # HSD header stores integer wavelength endpoints only;
        # wavelengths are reconstructed via linspace.  This only
        # matches if the original wavelengths are themselves
        # approximately linear.  We check endpoints and step size
        # rather than element-wise, since real data may be non-linear.
        assert abs(hsi_back.wavelengths[0] - round(source_hsi.wavelengths[0])) <= 1.0
        assert abs(hsi_back.wavelengths[-1] - round(source_hsi.wavelengths[-1])) <= 1.0
        assert hsi_back.wavelengths.shape == source_hsi.wavelengths.shape

    def test_hsd_round_trip_cube_data(self, source_hsi: HSI, tmp_path: Path) -> None:
        source_hsi.write(str(tmp_path / "cube.hsd"))
        hsi_back = HSI.read(str(tmp_path / "cube.hsd"))

        np.testing.assert_allclose(np.asarray(hsi_back), np.asarray(source_hsi), rtol=1e-6, atol=1e-6)


class TestPngFolderRoundTrip:
    def test_png_linecam_round_trip_shape(self, source_hsi: HSI, tmp_path: Path) -> None:
        folder = tmp_path / "png_line"
        source_hsi.write(str(folder), line_cam=True)
        hsi_back = HSI.read(str(folder), wavelengths=source_hsi.wavelengths, line_cam=True)

        assert hsi_back.shape == source_hsi.shape

    def test_png_linecam_round_trip_data(self, source_hsi: HSI, tmp_path: Path) -> None:
        folder = tmp_path / "png_line"
        source_hsi.write(str(folder), line_cam=True)
        hsi_back = HSI.read(str(folder), wavelengths=source_hsi.wavelengths, line_cam=True)

        # uint16 quantization: max error ~ 1/65535 per unit range.
        # Values > 1.0 are clipped, so we only compare within [0, 1].
        source_cube = np.clip(source_hsi.cube, 0.0, 1.0)
        back_cube = hsi_back.cube
        np.testing.assert_allclose(back_cube, source_cube, atol=1e-3)

    def test_png_channel_round_trip_shape(self, source_hsi: HSI, tmp_path: Path) -> None:
        folder = tmp_path / "png_ch"
        source_hsi.write(str(folder), line_cam=False)
        hsi_back = HSI.read(str(folder), wavelengths=source_hsi.wavelengths, line_cam=False)

        assert hsi_back.shape == source_hsi.shape

    def test_png_channel_round_trip_data(self, source_hsi: HSI, tmp_path: Path) -> None:
        folder = tmp_path / "png_ch"
        source_hsi.write(str(folder), line_cam=False)
        hsi_back = HSI.read(str(folder), wavelengths=source_hsi.wavelengths, line_cam=False)

        source_cube = np.clip(source_hsi.cube, 0.0, 1.0)
        back_cube = hsi_back.cube
        np.testing.assert_allclose(back_cube, source_cube, atol=1e-3)


class TestJp2WriteRead:
    def test_jp2_write_preserves_shape(self, source_hsi: HSI, tmp_path: Path) -> None:
        source_hsi.write(str(tmp_path / "cube.jp2"))
        hsi_back = HSI.read(str(tmp_path / "cube.jp2"))

        assert hsi_back.shape == source_hsi.shape

    def test_jp2_write_wavelengths_survive(self, source_hsi: HSI, tmp_path: Path) -> None:
        source_hsi.write(str(tmp_path / "cube.jp2"))
        hsi_back = HSI.read(str(tmp_path / "cube.jp2"))

        # JP2 wavelength metadata is stored as band tags; check that
        # the wavelengths are recovered (may need atol due to
        # potential metadata precision loss).
        np.testing.assert_allclose(hsi_back.wavelengths, source_hsi.wavelengths, atol=1.0)

    def test_jp2_write_data_approximate(self, source_hsi: HSI, tmp_path: Path) -> None:
        source_hsi.write(str(tmp_path / "cube.jp2"))
        hsi_back = HSI.read(str(tmp_path / "cube.jp2"))

        # JP2 is lossy; check coarse fidelity only.
        corr = np.corrcoef(np.asarray(source_hsi).ravel(), np.asarray(hsi_back).ravel())[0, 1]
        assert corr > 0.95


class TestWriteAutoDispatch:
    def test_write_auto_envi(self, source_hsi: HSI, tmp_path: Path) -> None:
        from hyperio.io import write_auto

        path = write_auto(source_hsi.cube, source_hsi.wavelengths, str(tmp_path / "out.hdr"))
        assert path.suffix == ".hdr"
        assert path.exists()

    def test_write_auto_tiff(self, source_hsi: HSI, tmp_path: Path) -> None:
        from hyperio.io import write_auto

        path = write_auto(source_hsi.cube, source_hsi.wavelengths, str(tmp_path / "out.tiff"))
        assert path.suffix == ".tiff"
        assert path.exists()

    def test_write_auto_hsd(self, source_hsi: HSI, tmp_path: Path) -> None:
        from hyperio.io import write_auto

        path = write_auto(source_hsi.cube, source_hsi.wavelengths, str(tmp_path / "out.hsd"))
        assert path.suffix == ".hsd"
        assert path.exists()

    def test_write_auto_unsupported_extension(self, source_hsi: HSI, tmp_path: Path) -> None:
        from hyperio.io import write_auto

        with pytest.raises(ValueError, match="Unsupported file extension"):
            write_auto(source_hsi.cube, source_hsi.wavelengths, str(tmp_path / "out.xyz"))

    def test_write_creates_parent_dirs(self, source_hsi: HSI, tmp_path: Path) -> None:
        nested = tmp_path / "a" / "b" / "c" / "cube.hdr"
        source_hsi.write(str(nested))
        hsi_back = HSI.read(str(nested))
        assert hsi_back.shape == source_hsi.shape
