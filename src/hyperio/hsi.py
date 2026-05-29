from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.signal import savgol_filter

from .io import read_auto, write_auto
from ._utils import (
    average_window,
    ensure_wavelengths_match_channels,
    normalize_to_hwc,
    percentile_stretch,
    resize_hsi,
    to_float32_cube,
)

_logger = logging.getLogger(__name__)


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

    def nearest_band(self, wavelength: float, window: int = 0) -> np.ndarray:
        idx = self.nearest_band_index(wavelength)
        return average_window(self._cube, idx, window)

    def bands_from_wavelengths(self, wavelengths: list[float], window: int = 0) -> np.ndarray:
        bands = [self.nearest_band(wl, window=window) for wl in wavelengths]
        return np.stack(bands, axis=2).astype(np.float32)

    def compute_index(self, formula: Callable[..., Any], **kwargs: Any) -> np.ndarray:
        window = int(kwargs.pop("window", 0))

        formula_args: dict[str, Any] = {}
        for key, value in kwargs.items():
            if isinstance(value, (int, float, np.floating)):
                formula_args[key] = self.nearest_band(float(value), window=window)
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
        q_low: float = 1.0,
        q_high: float = 99.0,
    ) -> np.ndarray:
        rgb_cube = self.bands_from_wavelengths([wl_red, wl_green, wl_blue], window=window)
        return percentile_stretch(rgb_cube, q_low=q_low, q_high=q_high)

    def rescale(self, factor: float) -> "HSI":
        return HSI((self._cube * np.float32(factor)).astype(np.float32), self.wavelengths)

    def resize(self, width: int, height: int) -> "HSI":
        resized = resize_hsi(self._cube, width=width, height=height)
        return HSI(resized, self.wavelengths)

    def filter_savgol(self, window_length: int = 31, polyorder: int = 3) -> "HSI":
        """Apply Savitzky-Golay filtering to each pixel spectrum and return a new HSI."""
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
            axis=2,
            mode="interp",
        ).astype(np.float32)
        return HSI(filtered, self.wavelengths)
