from __future__ import annotations

import numpy as np
import pytest

from hyperio import HSI


def _make_hsi(h: int = 10, w: int = 10, c: int = 50, seed: int = 42) -> HSI:
    rng = np.random.RandomState(seed)
    cube = rng.rand(h, w, c).astype(np.float32)
    wavelengths = np.linspace(400.0, 900.0, c, dtype=np.float32)
    return HSI(cube, wavelengths)


def _make_hsi_with_ref(h: int = 10, w: int = 10, c: int = 50) -> HSI:
    hsi = _make_hsi(h, w, c)
    hsi._reference_spectrum = np.ones(c, dtype=np.float32) * 0.5
    hsi._reference_multiplier = 2.0
    hsi._reference_eps = 1e-6
    return hsi


# ---- Crop ----

class TestCrop:
    def test_basic_crop(self):
        hsi = _make_hsi(20, 20, 50)
        cropped = hsi.crop(2, 8, 3, 15)
        assert cropped.shape == (6, 12, 50)
        np.testing.assert_array_equal(cropped.cube, hsi.cube[2:8, 3:15, :])

    def test_boundary_clamping(self):
        hsi = _make_hsi(10, 10, 50)
        cropped = hsi.crop(-5, 100, -3, 200)
        assert cropped.shape == hsi.shape

    def test_invalid_crop(self):
        hsi = _make_hsi(10, 10, 50)
        with pytest.raises(ValueError):
            hsi.crop(5, 5, 0, 10)

    def test_crop_preserves_wavelengths(self):
        hsi = _make_hsi()
        cropped = hsi.crop(0, 5, 0, 5)
        np.testing.assert_array_equal(cropped.wavelengths, hsi.wavelengths)

    def test_crop_preserves_reference(self):
        hsi = _make_hsi_with_ref()
        cropped = hsi.crop(0, 5, 0, 5)
        np.testing.assert_array_equal(cropped.reference_spectrum, hsi.reference_spectrum)


# ---- Mask spectra ----

class TestMaskSpectra:
    def test_boolean_mask(self):
        hsi = _make_hsi(10, 10, 50)
        mask = np.zeros((10, 10), dtype=bool)
        mask[0, 0] = True
        mask[1, 1] = True
        spectra = hsi.mask_spectra(mask)
        assert spectra.shape == (2, 50)
        np.testing.assert_array_equal(spectra[0], hsi.cube[0, 0, :])
        np.testing.assert_array_equal(spectra[1], hsi.cube[1, 1, :])

    def test_label_map(self):
        hsi = _make_hsi(10, 10, 50)
        labels = np.zeros((10, 10), dtype=np.int32)
        labels[0:3, 0:3] = 5
        spectra = hsi.mask_spectra(labels, label=5)
        assert spectra.shape == (9, 50)

    def test_int_mask_nonzero(self):
        hsi = _make_hsi(10, 10, 50)
        mask = np.zeros((10, 10), dtype=np.int32)
        mask[0, 0] = 1
        spectra = hsi.mask_spectra(mask)
        assert spectra.shape == (1, 50)

    def test_wrong_shape_mask(self):
        hsi = _make_hsi(10, 10, 50)
        with pytest.raises(ValueError):
            hsi.mask_spectra(np.zeros((5, 5), dtype=bool))


# ---- Predefined spectral indices ----

class TestSpectralIndices:
    def test_ndvi_shape(self):
        hsi = _make_hsi()
        result = hsi.ndvi()
        assert result.shape == (10, 10)

    def test_ndvi_custom_wavelengths(self):
        hsi = _make_hsi()
        result = hsi.ndvi(nir=850.0, red=650.0)
        assert result.shape == (10, 10)

    def test_all_indices_run(self):
        hsi = _make_hsi()
        for method in ("ndvi", "ndwi", "mndwi", "evi", "savi", "msavi", "mcari", "pri"):
            result = getattr(hsi, method)()
            assert result.shape == (10, 10), f"{method} returned wrong shape"

    def test_evi_shape(self):
        hsi = _make_hsi()
        result = hsi.evi()
        assert result.shape == (10, 10)


# ---- Continuum removal ----

class TestContinuumRemoval:
    def test_flat_spectrum(self):
        spectrum = np.ones(50, dtype=np.float32) * 0.5
        wavelengths = np.linspace(400, 900, 50, dtype=np.float32)
        hsi = HSI(np.tile(spectrum, (5, 5, 1)), wavelengths)
        removed = hsi.continuum_remove()
        np.testing.assert_allclose(removed.cube, 1.0, atol=1e-5)

    def test_fast_mode_shape(self):
        hsi = _make_hsi()
        removed = hsi.continuum_remove(per_pixel=False)
        assert removed.shape == hsi.shape

    def test_per_pixel_mode_shape(self):
        hsi = _make_hsi(5, 5, 20)
        removed = hsi.continuum_remove(per_pixel=True)
        assert removed.shape == hsi.shape

    def test_continuum_removed_per_pixel_bounded(self):
        hsi = _make_hsi(5, 5, 20)
        removed = hsi.continuum_remove(per_pixel=True)
        assert np.all(removed.cube >= -0.01)
        assert np.all(removed.cube <= 1.05)


# ---- Normalization ----

class TestNormalize:
    def test_minmax(self):
        hsi = _make_hsi()
        normed = hsi.normalize("minmax")
        assert normed.shape == hsi.shape
        flat = normed.cube.reshape(-1, hsi.shape[2])
        np.testing.assert_allclose(flat.min(axis=0), 0.0, atol=1e-5)
        np.testing.assert_allclose(flat.max(axis=0), 1.0, atol=1e-5)

    def test_l2(self):
        hsi = _make_hsi()
        normed = hsi.normalize("l2")
        flat = normed.cube.reshape(-1, hsi.shape[2])
        norms = np.linalg.norm(flat, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-4)

    def test_reference(self):
        hsi = _make_hsi_with_ref()
        normed = hsi.normalize("reference")
        assert normed.shape == hsi.shape

    def test_reference_explicit(self):
        hsi = _make_hsi()
        ref = np.ones(hsi.shape[2], dtype=np.float32) * 0.5
        normed = hsi.normalize("reference", reference_spectrum=ref)
        assert normed.shape == hsi.shape

    def test_reference_missing_raises(self):
        hsi = _make_hsi()
        with pytest.raises(ValueError):
            hsi.normalize("reference")

    def test_mean_centering(self):
        hsi = _make_hsi()
        normed = hsi.normalize("mean")
        flat = normed.cube.reshape(-1, hsi.shape[2])
        np.testing.assert_allclose(flat.mean(axis=1), 0.0, atol=1e-5)

    def test_invalid_method(self):
        hsi = _make_hsi()
        with pytest.raises(ValueError):
            hsi.normalize("unknown")


# ---- Savgol enhancements ----

class TestSavgolEnhancements:
    def test_derivative_order_1(self):
        hsi = _make_hsi()
        result = hsi.filter_savgol(window_length=7, polyorder=2, deriv=1)
        assert result.shape == hsi.shape

    def test_derivative_order_2(self):
        hsi = _make_hsi()
        result = hsi.filter_savgol(window_length=7, polyorder=3, deriv=2)
        assert result.shape == hsi.shape

    def test_window_length_nm(self):
        hsi = _make_hsi(c=50)
        result = hsi.filter_savgol(polyorder=2, window_length_nm=50.0)
        assert result.shape == hsi.shape

    def test_window_length_nm_overrides(self):
        hsi = _make_hsi()
        result1 = hsi.filter_savgol(window_length=7, polyorder=2, window_length_nm=50.0)
        assert result1.shape == hsi.shape


# ---- SAM ----

class TestSAM:
    def test_zero_angle_self(self):
        hsi = _make_hsi(5, 5, 20)
        ref = hsi.cube[2, 3, :]
        angles = hsi.sam(reference=ref)
        assert angles[2, 3] < 0.01

    def test_orthogonal_spectrum(self):
        hsi = _make_hsi(1, 1, 4)
        hsi._cube[0, 0, :] = [1.0, 0.0, 0.0, 0.0]
        ref = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        angles = hsi.sam(reference=ref)
        np.testing.assert_allclose(angles[0, 0], np.pi / 2, atol=1e-4)

    def test_sam_uses_stored_reference(self):
        hsi = _make_hsi_with_ref()
        angles = hsi.sam()
        assert angles.shape == (10, 10)

    def test_sam_no_reference_raises(self):
        hsi = _make_hsi()
        with pytest.raises(ValueError):
            hsi.sam()


# ---- Band statistics ----

class TestBandStatistics:
    def test_mean_spectrum_shape(self):
        hsi = _make_hsi()
        mean = hsi.mean_spectrum()
        assert mean.shape == (50,)

    def test_band_std_shape(self):
        hsi = _make_hsi()
        std = hsi.band_std()
        assert std.shape == (50,)

    def test_band_cov_shape(self):
        hsi = _make_hsi()
        cov = hsi.band_cov()
        assert cov.shape == (50, 50)

    def test_mean_known_values(self):
        cube = np.ones((4, 4, 3), dtype=np.float32)
        cube[:, :, 0] = 1.0
        cube[:, :, 1] = 2.0
        cube[:, :, 2] = 3.0
        hsi = HSI(cube, np.array([1.0, 2.0, 3.0], dtype=np.float32))
        np.testing.assert_allclose(hsi.mean_spectrum(), [1.0, 2.0, 3.0])


# ---- PCA ----

class TestPCA:
    def test_pca_reduces_dims(self):
        hsi = _make_hsi(10, 10, 50)
        result = hsi.pca(n_components=5)
        assert result.shape == (10, 10, 5)

    def test_pca_wavelengths(self):
        hsi = _make_hsi()
        result = hsi.pca(n_components=3)
        np.testing.assert_array_equal(result.wavelengths, [1.0, 2.0, 3.0])

    def test_pca_metadata(self):
        hsi = _make_hsi()
        result = hsi.pca(n_components=5)
        assert hasattr(result, "_pca_explained_variance_ratio_")
        assert result._pca_explained_variance_ratio_.shape == (5,)
        assert hasattr(result, "_pca_components_")
        assert result._pca_components_.shape == (5, 50)

    def test_pca_variance_explained(self):
        hsi = _make_hsi()
        result = hsi.pca(n_components=50)
        total_var = result._pca_explained_variance_ratio_.sum()
        np.testing.assert_allclose(total_var, 1.0, atol=1e-4)


# ---- MNF ----

class TestMNF:
    def test_mnf_reduces_dims(self):
        hsi = _make_hsi(20, 20, 30)
        result = hsi.mnf(n_components=5)
        assert result.shape == (20, 20, 5)

    def test_mnf_wavelengths(self):
        hsi = _make_hsi(20, 20, 30)
        result = hsi.mnf(n_components=3)
        np.testing.assert_array_equal(result.wavelengths, [1.0, 2.0, 3.0])

    def test_mnf_metadata(self):
        hsi = _make_hsi(20, 20, 30)
        result = hsi.mnf(n_components=5)
        assert hasattr(result, "_mnf_eigenvalues_")
        assert result._mnf_eigenvalues_.shape == (5,)


# ---- K-means ----

class TestKMeans:
    def test_euclidean(self):
        hsi = _make_hsi(20, 20, 30)
        labels, info = hsi.kmeans(n_clusters=3, metric="euclidean", random_state=0)
        assert labels.shape == (20, 20)
        assert info["centroids"].shape == (3, 30)
        assert info["metric"] == "euclidean"

    def test_sam(self):
        hsi = _make_hsi(20, 20, 30)
        labels, info = hsi.kmeans(n_clusters=3, metric="sam", random_state=0)
        assert labels.shape == (20, 20)
        assert info["metric"] == "sam"

    def test_correlation(self):
        hsi = _make_hsi(20, 20, 30)
        labels, info = hsi.kmeans(n_clusters=3, metric="correlation", random_state=0)
        assert labels.shape == (20, 20)

    def test_sid(self):
        hsi = _make_hsi(20, 20, 30)
        labels, info = hsi.kmeans(n_clusters=3, metric="sid", random_state=0, n_init=2, max_iter=50)
        assert labels.shape == (20, 20)
        assert info["centroids"].shape == (3, 30)

    def test_manhattan(self):
        hsi = _make_hsi(20, 20, 30)
        labels, info = hsi.kmeans(n_clusters=3, metric="manhattan", random_state=0, n_init=2, max_iter=50)
        assert labels.shape == (20, 20)

    def test_chebyshev(self):
        hsi = _make_hsi(20, 20, 30)
        labels, info = hsi.kmeans(n_clusters=3, metric="chebyshev", random_state=0, n_init=2, max_iter=50)
        assert labels.shape == (20, 20)

    def test_invalid_metric(self):
        hsi = _make_hsi()
        with pytest.raises(ValueError):
            hsi.kmeans(metric="cosine")

    def test_cluster_count(self):
        hsi = _make_hsi(20, 20, 30)
        labels, _ = hsi.kmeans(n_clusters=4, metric="euclidean", random_state=0)
        unique = np.unique(labels)
        assert len(unique) == 4

    def test_mask_spectra_with_kmeans_labels(self):
        hsi = _make_hsi(20, 20, 30)
        labels, _ = hsi.kmeans(n_clusters=3, metric="euclidean", random_state=0)
        class0 = hsi.mask_spectra(labels, label=0)
        assert class0.ndim == 2
        assert class0.shape[1] == 30
