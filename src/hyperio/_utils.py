from __future__ import annotations

import re
from typing import Iterable

import numpy as np
from skimage.transform import resize as sk_resize


def to_float32_cube(arr: np.ndarray) -> np.ndarray:
    """Convert an array to float32. Integer arrays are normalized to [0, 1].

    For unsigned integer types, divides by ``iinfo.max``.
    For signed integer types, shifts and scales so that the full
    integer range maps to ``[0, 1]``: ``(arr - min) / (max - min)``.
    """
    if np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        if np.issubdtype(arr.dtype, np.signedinteger):
            denom = float(info.max - info.min)
            if denom <= 0.0:
                return arr.astype(np.float32)
            return (arr.astype(np.float32) - float(info.min)) / denom
        denom = float(info.max)
        if denom <= 0.0:
            return arr.astype(np.float32)
        return arr.astype(np.float32) / denom
    return arr.astype(np.float32)


def normalize_to_hwc(arr: np.ndarray, *, prefer_last_channel: bool = True) -> np.ndarray:
    """Normalize array shape to (H, W, C).

    Supported inputs:
    - 2D: (H, W) -> (H, W, 1)
    - 3D: (H, W, C) or (C, H, W)
    """
    if arr.ndim == 2:
        return arr[:, :, None]

    if arr.ndim != 3:
        raise ValueError(f"Expected 2D or 3D array, got shape {arr.shape}")

    if prefer_last_channel:
        c_last = arr.shape[-1]
        c_first = arr.shape[0]

        if c_last <= 512:
            return arr
        if c_first <= 512:
            return np.moveaxis(arr, 0, -1)
        return arr

    return arr


def parse_envi_wavelengths(value: object) -> np.ndarray:
    """Parse ENVI wavelength metadata into float32 array."""
    if value is None:
        raise ValueError("ENVI metadata does not contain wavelength")

    if isinstance(value, (list, tuple)):
        vals = [float(v) for v in value]
        return np.asarray(vals, dtype=np.float32)

    if isinstance(value, str):
        numbers = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", value)
        if not numbers:
            raise ValueError("Could not parse ENVI wavelength string")
        return np.asarray([float(x) for x in numbers], dtype=np.float32)

    raise ValueError(f"Unsupported ENVI wavelength metadata type: {type(value)!r}")


def percentile_stretch(rgb: np.ndarray, q_low: float, q_high: float) -> np.ndarray:
    """Apply percentile stretch channel-wise and clamp to [0, 1]."""
    out = rgb.astype(np.float32, copy=True)
    for c in range(out.shape[-1]):
        band = out[:, :, c]
        finite = np.isfinite(band)
        if not np.any(finite):
            out[:, :, c] = 0.0
            continue

        vals = band[finite]
        lo = float(np.percentile(vals, q_low))
        hi = float(np.percentile(vals, q_high))
        if hi <= lo:
            hi = lo + 1e-6

        stretched = (band - lo) / (hi - lo)
        stretched[~finite] = 0.0
        out[:, :, c] = np.clip(stretched, 0.0, 1.0).astype(np.float32)

    return out


def resize_hsi(cube: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize HSI cube in spatial dimensions with bilinear interpolation."""
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")

    resized = sk_resize(
        cube,
        (height, width, cube.shape[2]),
        order=1,
        preserve_range=True,
        anti_aliasing=True,
    )
    return resized.astype(np.float32)


def ensure_wavelengths_match_channels(wavelengths: np.ndarray, channels: int) -> np.ndarray:
    """Validate wavelengths shape against channel count and return float32 array."""
    wl = np.asarray(wavelengths, dtype=np.float32)
    if wl.ndim != 1:
        raise ValueError(f"wavelengths must be 1D, got shape {wl.shape}")
    if wl.shape[0] != channels:
        raise ValueError(
            f"wavelength count ({wl.shape[0]}) does not match channels ({channels})"
        )
    return wl


def try_parse_wavelengths_from_xml_like_text(text: str) -> np.ndarray | None:
    """Try to extract wavelength vectors from XML-like text fragments."""
    lowered = text.lower()
    key_idx = lowered.find("wavelength")
    if key_idx < 0:
        return None

    snippet = text[key_idx : key_idx + 40000]
    numbers = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", snippet)
    if len(numbers) < 3:
        return None

    vals = np.asarray([float(x) for x in numbers], dtype=np.float32)
    return vals


def average_window(cube: np.ndarray, band_idx: int, window: int) -> np.ndarray:
    """Return a 2D band or averaged neighborhood around band index."""
    if window < 0:
        raise ValueError("window must be >= 0")

    if window == 0:
        return cube[:, :, band_idx]

    lo = max(0, band_idx - window)
    hi = min(cube.shape[2], band_idx + window + 1)
    return cube[:, :, lo:hi].mean(axis=2, dtype=np.float32).astype(np.float32)
