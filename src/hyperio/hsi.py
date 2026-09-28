from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.signal import savgol_filter

from .io import read_auto, write_auto
from ._utils import (
    average_window,
    continuum_remove_spectrum,
    ensure_wavelengths_match_channels,
    kmeans_plusplus_init,
    normalize_l2,
    normalize_mean,
    normalize_minmax,
    normalize_reference,
    normalize_to_hwc,
    percentile_stretch,
    resize_hsi,
    sid_distance,
    spectral_angle,
    to_float32_cube,
    upper_convex_hull_points,
)

_logger = logging.getLogger(__name__)

_SUPPORTED_KMEANS_METRICS = ("euclidean", "sam", "correlation", "sid", "manhattan", "chebyshev")


class HSI:
    """Hyperspectral image wrapper with NumPy-like behavior and wavelength metadata."""

    def __init__(self, cube: np.ndarray, wavelengths: np.ndarray):
        cube_arr = normalize_to_hwc(np.asarray(cube))
        cube_arr = to_float32_cube(cube_arr)

        wl_arr = ensure_wavelengths_match_channels(np.asarray(wavelengths), cube_arr.shape[2])

        self._cube = cube_arr
        self.wavelengths = wl_arr
        self._reference_spectrum: np.ndarray | None = None
        self._reference_multiplier: float = 1.0
        self._reference_eps: float = 1e-8

    @property
    def shape(self) -> tuple[int, int, int]:
        return self._cube.shape

    @property
    def dtype(self) -> np.dtype:
        return self._cube.dtype

    @property
    def ndim(self) -> int:
        return self._cube.ndim

    def __array__(self, dtype: Any | None = None) -> np.ndarray:
        if dtype is None:
            return self._cube
        return np.asarray(self._cube, dtype=dtype)

    def __getitem__(self, item: Any) -> Any:
        return self._cube[item]

    def __setitem__(self, key: Any, value: Any) -> None:
        self._cube[key] = value

    def __len__(self) -> int:
        return len(self._cube)

    def __repr__(self) -> str:
        return (
            f"HSI(shape={self.shape}, dtype={self.dtype}, "
            f"wavelengths_shape={self.wavelengths.shape})"
        )

    @property
    def cube(self) -> np.ndarray:
        return self._cube

    @property
    def reference_spectrum(self) -> np.ndarray | None:
        return self._reference_spectrum

    @property
    def reference_multiplier(self) -> float:
        return self._reference_multiplier

    @property
    def reference_eps(self) -> float:
        return self._reference_eps

    def copy(self) -> "HSI":
        new = HSI(self._cube.copy(), self.wavelengths.copy())
        if self._reference_spectrum is not None:
            new._reference_spectrum = self._reference_spectrum.copy()
        new._reference_multiplier = self._reference_multiplier
        new._reference_eps = self._reference_eps
        return new

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HSI):
            return NotImplemented
        return (
            self.shape == other.shape
            and np.allclose(self._cube, other._cube)
            and np.allclose(self.wavelengths, other.wavelengths)
        )

    def __hash__(self) -> int:
        raise TypeError("HSI objects are not hashable")

    def subset_wavelengths(self, min_wl: float, max_wl: float) -> "HSI":
        """Return a new HSI containing only bands within [min_wl, max_wl].

        Useful for extracting a spectral region of interest without
        manually indexing the cube.
        """
        mask = (self.wavelengths >= min_wl) & (self.wavelengths <= max_wl)
        if not np.any(mask):
            raise ValueError(
                f"No wavelengths found in range [{min_wl}, {max_wl}]. "
                f"Available range: [{self.wavelengths.min():.1f}, {self.wavelengths.max():.1f}]"
            )
        new_cube = self._cube[:, :, mask]
        new_wl = self.wavelengths[mask]
        new_hsi = HSI(new_cube, new_wl)
        if self._reference_spectrum is not None:
            new_hsi._reference_spectrum = self._reference_spectrum[mask]
        new_hsi._reference_multiplier = self._reference_multiplier
        new_hsi._reference_eps = self._reference_eps
        return new_hsi

    @classmethod
    def from_cube(
        cls,
        cube: np.ndarray,
        wavelengths: np.ndarray | list[float] | None = None,
        min_wavelength: float | None = None,
        max_wavelength: float | None = None,
        spectral_resolution: float | None = None,
    ) -> "HSI":
        """Build HSI from cube and either explicit wavelengths or spectral range parameters."""
        cube_arr = normalize_to_hwc(np.asarray(cube))
        channels = int(cube_arr.shape[2])

        has_explicit = wavelengths is not None
        has_any_range_arg = (
            min_wavelength is not None
            or max_wavelength is not None
            or spectral_resolution is not None
        )
        has_required_range_bounds = min_wavelength is not None and max_wavelength is not None

        if not has_explicit and has_any_range_arg and not has_required_range_bounds:
            raise ValueError(
                "When wavelengths is not provided, both min_wavelength and max_wavelength are required."
            )

        if has_explicit == has_required_range_bounds:
            raise ValueError(
                "Provide either wavelengths or (min_wavelength and max_wavelength, with optional spectral_resolution)."
            )

        if has_explicit:
            wl_arr = np.asarray(wavelengths, dtype=np.float32)
            return cls(cube_arr, wl_arr)

        assert min_wavelength is not None
        assert max_wavelength is not None
        if spectral_resolution is None:
            spectral_resolution = (float(max_wavelength) - float(min_wavelength)) / float(channels)

        if spectral_resolution <= 0:
            raise ValueError("spectral_resolution must be > 0")

        wl_arr = (
            np.float32(min_wavelength)
            + np.arange(channels, dtype=np.float32) * np.float32(spectral_resolution)
        ).astype(np.float32)

        if wl_arr[0] < float(min_wavelength) - 1e-6:
            raise ValueError("Generated wavelengths are below min_wavelength")
        if wl_arr[-1] > float(max_wavelength) + 1e-6:
            raise ValueError("Generated wavelengths exceed max_wavelength")

        return cls(cube_arr, wl_arr)

    @classmethod
    def read(
        cls,
        path: str | Path,
        wavelengths: np.ndarray | list[float] | None = None,
        min_wavelength: float | None = None,
        max_wavelength: float | None = None,
        line_cam: bool = True,
        normalize: bool = True,
        metadata_json: bool = False,
    ) -> "HSI":
        result = read_auto(
            path,
            wavelengths=wavelengths,
            min_wavelength=min_wavelength,
            max_wavelength=max_wavelength,
            line_cam=line_cam,
            normalize=normalize,
            metadata_json=metadata_json,
        )
        hsi = cls(result.cube, result.wavelengths)
        hsi._reference_spectrum = result.reference_spectrum
        hsi._reference_multiplier = result.reference_multiplier
        hsi._reference_eps = result.reference_eps
        return hsi

    def write(self, path: str | Path, **kwargs: Any) -> Path:
        """Write the HSI to disk. Format is inferred from the file extension.

        Supported extensions: ``.hdr`` (ENVI), ``.tif``/``.tiff``,
        ``.jp2``, ``.hsd``, or a directory path (PNG folder).

        Wavelength metadata is embedded in each format's native metadata
        store where possible (ENVI header, TIFF ImageJ metadata, JP2
        band tags, HSD header).  PNG folders do not store wavelengths.

        If ``metadata_json=True``, a JSON sidecar file is written
        alongside the image containing full wavelength and reference
        spectrum metadata.

        For PNG folder output, pass ``line_cam=True`` (default) or
        ``line_cam=False`` via *kwargs*.

        Returns the path of the file that was written.
        """
        from .io import write_png_folder

        p = Path(path)
        metadata_json = kwargs.get("metadata_json", False)
        ref_kw: dict[str, Any] = {}
        if self._reference_spectrum is not None:
            ref_kw["reference_spectrum"] = self._reference_spectrum
            ref_kw["reference_multiplier"] = self._reference_multiplier
            ref_kw["reference_eps"] = self._reference_eps

        if p.is_dir() or (not p.suffix):
            line_cam = kwargs.get("line_cam", True)
            return write_png_folder(
                self._cube, self.wavelengths, p,
                line_cam=line_cam,
                metadata_json=metadata_json,
                **ref_kw,
            )
        return write_auto(
            self._cube, self.wavelengths, p,
            metadata_json=metadata_json,
            **ref_kw,
        )

    def nearest_band_index(self, wavelength: float) -> int:
        idx = int(np.argmin(np.abs(self.wavelengths - np.float32(wavelength))))
        return idx

    def nearest_band(self, wavelength: float, window: int = 0, window_agg: str = "mean") -> np.ndarray:
        idx = self.nearest_band_index(wavelength)
        return average_window(self._cube, idx, window, agg=window_agg)

    def bands_from_wavelengths(self, wavelengths: list[float], window: int = 0, window_agg: str = "mean") -> np.ndarray:
        bands = [self.nearest_band(wl, window=window, window_agg=window_agg) for wl in wavelengths]
        return np.stack(bands, axis=2).astype(np.float32)

    def compute_index(self, formula: Callable[..., Any], **kwargs: Any) -> np.ndarray:
        window = int(kwargs.pop("window", 0))
        window_agg = kwargs.pop("window_agg", "mean")

        formula_args: dict[str, Any] = {}
        for key, value in kwargs.items():
            if isinstance(value, (int, float, np.floating)):
                formula_args[key] = self.nearest_band(float(value), window=window, window_agg=window_agg)
            else:
                formula_args[key] = value

        out = formula(**formula_args)
        return np.asarray(out, dtype=np.float32)

    def rgb(
        self,
        wl_red: float = 670.0,
        wl_green: float = 550.0,
        wl_blue: float = 475.0,
        window: int = 0,
        window_agg: str = "mean",
        q_low: float = 1.0,
        q_high: float = 99.0,
    ) -> np.ndarray:
        rgb_cube = self.bands_from_wavelengths([wl_red, wl_green, wl_blue], window=window, window_agg=window_agg)
        return percentile_stretch(rgb_cube, q_low=q_low, q_high=q_high)

    def rescale(self, factor: float) -> "HSI":
        return HSI((self._cube * np.float32(factor)).astype(np.float32), self.wavelengths)

    def resize(self, width: int, height: int) -> "HSI":
        resized = resize_hsi(self._cube, width=width, height=height)
        return HSI(resized, self.wavelengths)

    # ---- Spatial crop ----

    def crop(self, y1: int, y2: int, x1: int, x2: int) -> "HSI":
        """Return a new HSI with the spatial subregion [y1:y2, x1:x2].

        Indices are clamped to image bounds.  Raises ``ValueError``
        if the clamped region is empty.
        """
        h, w = self.shape[0], self.shape[1]
        y1, y2 = max(0, y1), min(h, y2)
        x1, x2 = max(0, x1), min(w, x2)
        if y1 >= y2 or x1 >= x2:
            raise ValueError(f"Empty crop region after clamping: y=[{y1},{y2}), x=[{x1},{x2})")
        new_cube = self._cube[y1:y2, x1:x2, :]
        new_hsi = HSI(new_cube, self.wavelengths.copy())
        if self._reference_spectrum is not None:
            new_hsi._reference_spectrum = self._reference_spectrum.copy()
        new_hsi._reference_multiplier = self._reference_multiplier
        new_hsi._reference_eps = self._reference_eps
        return new_hsi

    # ---- Masked spectra extraction ----

    def mask_spectra(self, mask: np.ndarray, label: int | None = None) -> np.ndarray:
        """Extract spectra at masked pixels as a 2D array (N, C).

        Args:
            mask: 2D boolean array of shape (H, W), or a 2D integer
                label map.  When *label* is provided, pixels equal to
                *label* are selected from the integer map.
            label: If *mask* is an integer label map, select pixels
                equal to this value.

        Returns:
            (N, C) float32 array of selected spectra.
        """
        mask = np.asarray(mask)
        if mask.ndim != 2:
            raise ValueError(f"mask must be 2D, got shape {mask.shape}")
        if mask.shape != self.shape[:2]:
            raise ValueError(f"mask shape {mask.shape} doesn't match image spatial shape {self.shape[:2]}")
        if label is not None:
            bool_mask = mask == label
        elif mask.dtype == bool:
            bool_mask = mask
        else:
            bool_mask = mask != 0
        return self._cube[bool_mask].astype(np.float32)

    # ---- Predefined spectral indices ----

    def ndvi(self, nir: float = 800.0, red: float = 670.0, **kw: Any) -> np.ndarray:
        return self.compute_index(lambda nir, red: (nir - red) / (nir + red + 1e-6), nir=nir, red=red, **kw)

    def ndwi(self, green: float = 560.0, nir: float = 800.0, **kw: Any) -> np.ndarray:
        return self.compute_index(lambda green, nir: (green - nir) / (green + nir + 1e-6), green=green, nir=nir, **kw)

    def mndwi(self, green: float = 560.0, swir: float = 1240.0, **kw: Any) -> np.ndarray:
        return self.compute_index(lambda green, swir: (green - swir) / (green + swir + 1e-6), green=green, swir=swir, **kw)

    def evi(self, nir: float = 800.0, red: float = 670.0, blue: float = 470.0, **kw: Any) -> np.ndarray:
        return self.compute_index(
            lambda nir, red, blue: 2.5 * (nir - red) / (nir + 6.0 * red - 7.5 * blue + 1.0),
            nir=nir, red=red, blue=blue, **kw,
        )

    def savi(self, nir: float = 800.0, red: float = 670.0, L: float = 0.5, **kw: Any) -> np.ndarray:
        return self.compute_index(
            lambda nir, red: (nir - red) * (1.0 + L) / (nir + red + L),
            nir=nir, red=red, **kw,
        )

    def msavi(self, nir: float = 800.0, red: float = 670.0, **kw: Any) -> np.ndarray:
        return self.compute_index(
            lambda nir, red: (2.0 * nir + 1.0 - np.sqrt((2.0 * nir + 1.0) ** 2 - 8.0 * (nir - red))) / 2.0,
            nir=nir, red=red, **kw,
        )

    def mcari(self, r700: float = 700.0, r670: float = 670.0, r550: float = 550.0, **kw: Any) -> np.ndarray:
        return self.compute_index(
            lambda r700, r670, r550: ((r700 - r670) - 0.2 * (r700 - r550)) * (r700 / (r670 + 1e-6)),
            r700=r700, r670=r670, r550=r550, **kw,
        )

    def pri(self, r531: float = 531.0, r570: float = 570.0, **kw: Any) -> np.ndarray:
        return self.compute_index(
            lambda r531, r570: (r531 - r570) / (r531 + r570 + 1e-6),
            r531=r531, r570=r570, **kw,
        )

    # ---- Continuum removal ----

    def continuum_remove(self, per_pixel: bool = False) -> "HSI":
        """Remove the convex-hull continuum from spectra.

        When *per_pixel* is False (default), the continuum is computed
        from the mean spectrum and applied to all pixels — fast and
        suitable for most use cases.  When *per_pixel* is True, a
        per-pixel hull is computed (accurate but slow for large images).
        """
        h, w, c = self.shape
        wavelengths = self.wavelengths

        if not per_pixel:
            mean_spec = self._cube.mean(axis=(0, 1))
            hull_wl, hull_vals = upper_convex_hull_points(mean_spec, wavelengths)
            continuum = np.interp(wavelengths, hull_wl, hull_vals).astype(np.float32)
            denom = np.where(continuum > 0, continuum, 1.0)
            removed = (self._cube / denom).astype(np.float32)
        else:
            flat = self._cube.reshape(-1, c)
            removed_flat = np.empty_like(flat)
            for i in range(flat.shape[0]):
                removed_flat[i] = continuum_remove_spectrum(flat[i], wavelengths)
            removed = removed_flat.reshape(h, w, c)

        new_hsi = HSI(removed, wavelengths.copy())
        if self._reference_spectrum is not None:
            new_hsi._reference_spectrum = self._reference_spectrum.copy()
        new_hsi._reference_multiplier = self._reference_multiplier
        new_hsi._reference_eps = self._reference_eps
        return new_hsi

    # ---- Normalization ----

    def normalize(self, method: str = "minmax", **kwargs: Any) -> "HSI":
        """Normalize the cube and return a new HSI.

        Methods:
            ``minmax``  — per-band min-max to [0, 1]
            ``l2``      — per-pixel L2 (unit vector)
            ``reference`` — divide by white reference spectrum
            ``mean``    — per-pixel mean-centering

        For ``reference``, pass ``reference_spectrum`` and optionally
        ``multiplier`` and ``eps`` via *kwargs*, or use the stored
        reference spectrum.
        """
        valid = ("minmax", "l2", "reference", "mean")
        if method not in valid:
            raise ValueError(f"Unknown normalization method '{method}'. Choose from {valid}")

        if method == "minmax":
            normalized = normalize_minmax(self._cube)
        elif method == "l2":
            normalized = normalize_l2(self._cube)
        elif method == "reference":
            ref = kwargs.get("reference_spectrum", self._reference_spectrum)
            if ref is None:
                raise ValueError("No reference spectrum available. Pass reference_spectrum= or load one.")
            ref = np.asarray(ref, dtype=np.float32)
            mult = kwargs.get("multiplier", self._reference_multiplier)
            eps = kwargs.get("eps", self._reference_eps)
            normalized = normalize_reference(self._cube, ref, mult, eps)
        elif method == "mean":
            normalized = normalize_mean(self._cube)

        new_hsi = HSI(normalized, self.wavelengths.copy())
        if self._reference_spectrum is not None:
            new_hsi._reference_spectrum = self._reference_spectrum.copy()
        new_hsi._reference_multiplier = self._reference_multiplier
        new_hsi._reference_eps = self._reference_eps
        return new_hsi

    # ---- Savitzky-Golay enhancements ----

    def filter_savgol(
        self,
        window_length: int = 31,
        polyorder: int = 3,
        *,
        deriv: int = 0,
        delta: float = 1.0,
        window_length_nm: float | None = None,
    ) -> "HSI":
        """Apply Savitzky-Golay filtering (and optional derivatives) to each pixel spectrum.

        New in v0.3.0:
            *deriv* — spectral derivative order (0=smoothing, 1=first
            derivative, 2=second derivative).

            *delta* — wavelength spacing for derivative scaling.  Defaults
            to 1.0 (per-sample).  Pass the mean wavelength spacing in nm
            for physically meaningful derivative units.

            *window_length_nm* — if provided, the window length is
            computed from the wavelength axis instead of band index.
            Overrides *window_length*.
        """
        if window_length_nm is not None:
            wl_spacings = np.diff(self.wavelengths)
            mean_spacing = float(np.median(wl_spacings)) if len(wl_spacings) > 0 else 1.0
            if mean_spacing <= 0:
                mean_spacing = 1.0
            window_length = max(3, int(round(window_length_nm / mean_spacing)))
            if window_length % 2 == 0:
                window_length += 1

        if window_length <= 0:
            raise ValueError("window_length must be > 0")
        if window_length % 2 == 0:
            raise ValueError("window_length must be odd")
        if polyorder < 0:
            raise ValueError("polyorder must be >= 0")
        if polyorder >= window_length:
            raise ValueError("polyorder must be smaller than window_length")
        if window_length > self.shape[2]:
            raise ValueError(
                f"window_length ({window_length}) cannot be larger than number of bands ({self.shape[2]})"
            )

        filtered = savgol_filter(
            self._cube,
            window_length=window_length,
            polyorder=polyorder,
            deriv=deriv,
            delta=delta,
            axis=2,
            mode="interp",
        ).astype(np.float32)
        return HSI(filtered, self.wavelengths)

    # ---- Spectral Angle Mapper ----

    def sam(self, reference: np.ndarray | None = None) -> np.ndarray:
        """Compute Spectral Angle Mapper between each pixel and a reference.

        Args:
            reference: 1D reference spectrum of length C.  If None,
                uses the stored ``reference_spectrum``.

        Returns:
            2D float32 array of shape (H, W) with angles in radians.
        """
        if reference is None:
            reference = self._reference_spectrum
        if reference is None:
            raise ValueError("No reference spectrum available. Pass reference= or load one.")
        reference = np.asarray(reference, dtype=np.float32)

        h, w, c = self.shape
        flat = self._cube.reshape(-1, c)
        angles = spectral_angle(flat, reference)
        return angles.reshape(h, w)

    # ---- Band statistics ----

    def mean_spectrum(self) -> np.ndarray:
        """Mean spectrum across all pixels. Shape (C,)."""
        return self._cube.mean(axis=(0, 1)).astype(np.float32)

    def band_std(self) -> np.ndarray:
        """Standard deviation per band. Shape (C,)."""
        return self._cube.std(axis=(0, 1)).astype(np.float32)

    def band_cov(self) -> np.ndarray:
        """Band covariance matrix. Shape (C, C)."""
        h, w, c = self.shape
        flat = self._cube.reshape(-1, c)
        return np.cov(flat, rowvar=False).astype(np.float32)

    # ---- PCA ----

    def pca(self, n_components: int, whiten: bool = False, random_state: int | None = None) -> "HSI":
        """Principal Component Analysis transform.

        Returns a new HSI with *n_components* bands.  The returned
        HSI has metadata attributes ``_pca_explained_variance_ratio_``
        and ``_pca_components_`` attached.
        """
        from sklearn.decomposition import PCA

        h, w, c = self.shape
        flat = self._cube.reshape(-1, c)

        model = PCA(n_components=n_components, whiten=whiten, random_state=random_state)
        transformed = model.fit_transform(flat).astype(np.float32)

        new_cube = transformed.reshape(h, w, n_components)
        new_wl = np.arange(1, n_components + 1, dtype=np.float32)
        new_hsi = HSI(new_cube, new_wl)
        new_hsi._pca_explained_variance_ratio_ = model.explained_variance_ratio_.astype(np.float32)
        new_hsi._pca_components_ = model.components_.astype(np.float32)
        return new_hsi

    # ---- MNF ----

    def mnf(self, n_components: int) -> "HSI":
        """Minimum Noise Fraction transform.

        Noise is estimated via shift-difference along both spatial
        axes.  The data is whitened with respect to the noise
        covariance, then PCA is applied to the whitened data.

        Returns a new HSI with *n_components* bands.  The returned
        HSI has metadata attributes ``_mnf_eigenvalues_`` and
        ``_mnf_noise_fraction_`` attached.
        """
        h, w, c = self.shape
        flat = self._cube.reshape(-1, c)

        noise_bands = []
        if h > 1:
            noise_bands.append((self._cube[1:, :, :] - self._cube[:-1, :, :]).reshape(-1, c))
        if w > 1:
            noise_bands.append((self._cube[:, 1:, :] - self._cube[:, :-1, :]).reshape(-1, c))
        if not noise_bands:
            raise ValueError("Image too small for MNF noise estimation (need at least 2 rows or columns)")
        noise_data = np.concatenate(noise_bands, axis=0)
        noise_cov = np.cov(noise_data, rowvar=False).astype(np.float64)

        eigvals, eigvecs = np.linalg.eigh(noise_cov)
        eigvals = np.maximum(eigvals, 1e-12)
        whitener = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

        whitened = (flat.astype(np.float64) @ whitener).astype(np.float32)

        total_cov = np.cov(whitened, rowvar=False).astype(np.float64)
        mnf_eigvals, mnf_eigvecs = np.linalg.eigh(total_cov)

        idx = np.argsort(mnf_eigvals)[::-1]
        mnf_eigvals = mnf_eigvals[idx]
        mnf_eigvecs = mnf_eigvecs[:, idx]

        n_comp = min(n_components, c)
        transform = (whitener @ mnf_eigvecs[:, :n_comp]).astype(np.float32)
        result = (flat @ transform).astype(np.float32)

        new_cube = result.reshape(h, w, n_comp)
        new_wl = np.arange(1, n_comp + 1, dtype=np.float32)
        new_hsi = HSI(new_cube, new_wl)
        new_hsi._mnf_eigenvalues_ = mnf_eigvals[:n_comp].astype(np.float32)
        noise_fractions = eigvals[idx[:n_comp]] if len(eigvals) >= n_comp else np.zeros(n_comp, dtype=np.float32)
        new_hsi._mnf_noise_fraction_ = noise_fractions.astype(np.float32)
        return new_hsi

    # ---- K-means clustering ----

    def kmeans(
        self,
        n_clusters: int = 8,
        metric: str = "euclidean",
        max_iter: int = 300,
        n_init: int = 10,
        random_state: int | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """K-means clustering with multiple distance metrics.

        Supported metrics:
            ``euclidean``    — standard Euclidean distance (sklearn)
            ``sam``          — Spectral Angle Mapper (L2-normalize then Euclidean)
            ``correlation``  — spectral correlation (mean-center + L2-normalize then Euclidean)
            ``sid``          — Spectral Information Divergence (custom)
            ``manhattan``    — L1 / cityblock distance (custom)
            ``chebyshev``   — L-infinity distance (custom)

        Returns (labels, info) where:
            labels — 2D int32 array of shape (H, W) with cluster assignments
            info   — dict with keys ``centroids``, ``inertia``, ``n_iter``, ``metric``
        """
        if metric not in _SUPPORTED_KMEANS_METRICS:
            raise ValueError(f"Unknown metric '{metric}'. Choose from {_SUPPORTED_KMEANS_METRICS}")

        h, w, c = self.shape
        flat = self._cube.reshape(-1, c)

        if metric in ("euclidean", "sam", "correlation"):
            return self._kmeans_sklearn(flat, n_clusters, metric, max_iter, n_init, random_state, h, w)
        return self._kmeans_custom(flat, n_clusters, metric, max_iter, n_init, random_state, h, w, c)

    def _kmeans_sklearn(
        self,
        flat: np.ndarray,
        n_clusters: int,
        metric: str,
        max_iter: int,
        n_init: int,
        random_state: int | None,
        h: int,
        w: int,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        from sklearn.cluster import KMeans

        X = flat.copy()
        if metric == "sam":
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms[norms < 1e-12] = 1.0
            X = X / norms
        elif metric == "correlation":
            X = X - X.mean(axis=1, keepdims=True)
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms[norms < 1e-12] = 1.0
            X = X / norms

        model = KMeans(
            n_clusters=n_clusters,
            max_iter=max_iter,
            n_init=n_init,
            random_state=random_state,
            init="k-means++",
        )
        labels = model.fit_predict(X)

        centroids = model.cluster_centers_.astype(np.float32)
        if metric == "sam":
            norms_c = np.linalg.norm(centroids, axis=1, keepdims=True)
            norms_c[norms_c < 1e-12] = 1.0
            centroids = centroids / norms_c

        labels_2d = labels.reshape(h, w).astype(np.int32)
        info: dict[str, Any] = {
            "centroids": centroids,
            "inertia": float(model.inertia_),
            "n_iter": int(model.n_iter_),
            "metric": metric,
        }
        return labels_2d, info

    def _kmeans_custom(
        self,
        flat: np.ndarray,
        n_clusters: int,
        metric: str,
        max_iter: int,
        n_init: int,
        random_state: int | None,
        h: int,
        w: int,
        c: int,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        from scipy.spatial.distance import cdist

        best_labels = None
        best_inertia = np.inf
        best_centroids = None
        best_n_iter = 0

        for run in range(n_init):
            rs = random_state + run if random_state is not None else None
            centroids = kmeans_plusplus_init(flat, n_clusters, metric, random_state=rs)

            labels = np.zeros(flat.shape[0], dtype=np.int32)
            for iteration in range(max_iter):
                if metric == "sid":
                    dists = sid_distance(flat, centroids)
                elif metric == "manhattan":
                    dists = cdist(flat, centroids, metric="cityblock")
                elif metric == "chebyshev":
                    dists = cdist(flat, centroids, metric="chebyshev")
                else:
                    raise ValueError(f"Unsupported custom metric: {metric}")

                new_labels = np.argmin(dists, axis=1).astype(np.int32)
                if np.array_equal(new_labels, labels) and iteration > 0:
                    labels = new_labels
                    break
                labels = new_labels

                for k in range(n_clusters):
                    mask = labels == k
                    if np.any(mask):
                        centroids[k] = flat[mask].mean(axis=0)

            if metric == "sid":
                final_dists = sid_distance(flat, centroids)
            elif metric == "manhattan":
                final_dists = cdist(flat, centroids, metric="cityblock")
            else:
                final_dists = cdist(flat, centroids, metric="chebyshev")
            inertia = float(final_dists[np.arange(len(labels)), labels].sum())

            if inertia < best_inertia:
                best_inertia = inertia
                best_labels = labels.copy()
                best_centroids = centroids.copy()
                best_n_iter = iteration + 1

        labels_2d = best_labels.reshape(h, w)
        info: dict[str, Any] = {
            "centroids": best_centroids,
            "inertia": best_inertia,
            "n_iter": best_n_iter,
            "metric": metric,
        }
        return labels_2d, info
