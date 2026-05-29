"""Tests for code review fixes: copy, __eq__, subset_wavelengths, signed int normalization."""
from __future__ import annotations

import numpy as np
import pytest

from hyperio import HSI


class TestCopyPreservesReferenceMetadata:
    def test_copy_preserves_reference_spectrum(self) -> None:
        cube = np.random.rand(10, 10, 5).astype(np.float32)
        wl = np.array([400.0, 500.0, 600.0, 700.0, 800.0], dtype=np.float32)
        ref = np.ones(5, dtype=np.float32) * 0.5
        hsi = HSI(cube, wl)
        hsi._reference_spectrum = ref
        hsi._reference_multiplier = 2.0
        hsi._reference_eps = 1e-6

        copied = hsi.copy()
        assert copied._reference_spectrum is not None
        np.testing.assert_allclose(copied._reference_spectrum, ref)
        assert copied._reference_multiplier == 2.0
        assert abs(copied._reference_eps - 1e-6) < 1e-12

    def test_copy_without_reference(self) -> None:
        cube = np.random.rand(10, 10, 5).astype(np.float32)
        wl = np.array([400.0, 500.0, 600.0, 700.0, 800.0], dtype=np.float32)
        hsi = HSI(cube, wl)

        copied = hsi.copy()
        assert copied._reference_spectrum is None
        assert copied._reference_multiplier == 1.0
        assert copied._reference_eps == 1e-8


class TestHSIEq:
    def test_equal_hsi(self) -> None:
        cube = np.random.rand(10, 10, 5).astype(np.float32)
        wl = np.array([400.0, 500.0, 600.0, 700.0, 800.0], dtype=np.float32)
        h1 = HSI(cube, wl)
        h2 = HSI(cube.copy(), wl.copy())
        assert h1 == h2

    def test_unequal_cube(self) -> None:
        cube = np.zeros((10, 10, 5), dtype=np.float32)
        wl = np.array([400.0, 500.0, 600.0, 700.0, 800.0], dtype=np.float32)
        h1 = HSI(cube, wl)
        h2 = HSI(cube.copy(), wl.copy())
        h2[0, 0, 0] = 1.0
        assert h1 != h2

    def test_unequal_wavelengths(self) -> None:
        cube = np.zeros((10, 10, 5), dtype=np.float32)
        wl1 = np.array([400.0, 500.0, 600.0, 700.0, 800.0], dtype=np.float32)
        wl2 = np.array([410.0, 510.0, 610.0, 710.0, 810.0], dtype=np.float32)
        h1 = HSI(cube, wl1)
        h2 = HSI(cube.copy(), wl2)
        assert h1 != h2

    def test_not_equal_to_other_type(self) -> None:
        cube = np.zeros((10, 10, 5), dtype=np.float32)
        wl = np.array([400.0, 500.0, 600.0, 700.0, 800.0], dtype=np.float32)
        h1 = HSI(cube, wl)
        assert h1 != "not an HSI"

    def test_hash_raises(self) -> None:
        cube = np.zeros((10, 10, 5), dtype=np.float32)
        wl = np.array([400.0, 500.0, 600.0, 700.0, 800.0], dtype=np.float32)
        h1 = HSI(cube, wl)
        with pytest.raises(TypeError, match="not hashable"):
            hash(h1)


class TestSubsetWavelengths:
    def test_subset_returns_correct_shape(self) -> None:
        cube = np.random.rand(10, 10, 10).astype(np.float32)
        wl = np.linspace(400, 850, 10, dtype=np.float32)
        hsi = HSI(cube, wl)

        sub = hsi.subset_wavelengths(500.0, 700.0)
        assert sub.shape[2] < hsi.shape[2]
        assert sub.shape[0] == hsi.shape[0]
        assert sub.shape[1] == hsi.shape[1]

    def test_subset_wavelengths_are_in_range(self) -> None:
        cube = np.random.rand(10, 10, 10).astype(np.float32)
        wl = np.linspace(400, 850, 10, dtype=np.float32)
        hsi = HSI(cube, wl)

        sub = hsi.subset_wavelengths(500.0, 700.0)
        assert sub.wavelengths.min() >= 500.0
        assert sub.wavelengths.max() <= 700.0

    def test_subset_preserves_reference(self) -> None:
        cube = np.random.rand(10, 10, 10).astype(np.float32)
        wl = np.linspace(400, 850, 10, dtype=np.float32)
        ref = np.ones(10, dtype=np.float32) * 0.5
        hsi = HSI(cube, wl)
        hsi._reference_spectrum = ref
        hsi._reference_multiplier = 2.0

        sub = hsi.subset_wavelengths(500.0, 700.0)
        assert sub._reference_spectrum is not None
        assert sub._reference_spectrum.shape == sub.wavelengths.shape
        assert sub._reference_multiplier == 2.0

    def test_subset_empty_range_raises(self) -> None:
        cube = np.random.rand(10, 10, 10).astype(np.float32)
        wl = np.linspace(400, 850, 10, dtype=np.float32)
        hsi = HSI(cube, wl)

        with pytest.raises(ValueError, match="No wavelengths found"):
            hsi.subset_wavelengths(900.0, 1000.0)


class TestSignedIntNormalization:
    def test_int8_normalized_to_unit_range(self) -> None:
        from hyperio._utils import to_float32_cube

        arr = np.array([[-128, 0, 127]], dtype=np.int8)
        result = to_float32_cube(arr)
        np.testing.assert_allclose(result[0, 0], 0.0, atol=1e-6)
        np.testing.assert_allclose(result[0, 2], 1.0, atol=1e-2)
        assert result.dtype == np.float32

    def test_uint16_normalized(self) -> None:
        from hyperio._utils import to_float32_cube

        arr = np.array([[0, 32768, 65535]], dtype=np.uint16)
        result = to_float32_cube(arr)
        np.testing.assert_allclose(result[0, 0], 0.0, atol=1e-6)
        np.testing.assert_allclose(result[0, 2], 1.0, atol=1e-4)
