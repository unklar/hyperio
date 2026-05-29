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


def upper_convex_hull_points(spectrum: np.ndarray, wavelengths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute the upper convex hull of a spectrum.

    The upper convex hull is the sequence of points such that the
    line connecting consecutive points lies above or on all
    intermediate data points.  This defines the "continuum" for
    continuum removal.

    Returns (hull_wavelengths, hull_values).
    """
    n = len(spectrum)
    if n < 2:
        return wavelengths.copy(), spectrum.copy()

    indices = [0]
    for i in range(1, n):
        while len(indices) >= 2:
            j = indices[-1]
            k = indices[-2]
            dx1 = wavelengths[j] - wavelengths[k]
            dy1 = spectrum[j] - spectrum[k]
            dx2 = wavelengths[i] - wavelengths[k]
            dy2 = spectrum[i] - spectrum[k]
            cross = dx1 * dy2 - dy1 * dx2
            if cross >= 0:
                indices.pop()
            else:
                break
        indices.append(i)

    return wavelengths[indices].copy(), spectrum[indices].copy()


def continuum_remove_spectrum(spectrum: np.ndarray, wavelengths: np.ndarray) -> np.ndarray:
    """Remove the continuum from a single spectrum.

    Computes the upper convex hull, interpolates it onto the
    wavelength grid, and divides the spectrum by the continuum.
    Result values are in [0, 1], where absorption features
    appear as dips below 1.0.
    """
    hull_wl, hull_vals = upper_convex_hull_points(spectrum, wavelengths)
    continuum = np.interp(wavelengths, hull_wl, hull_vals)
    out = np.where(continuum > 0, spectrum / continuum, 0.0)
    return out.astype(np.float32)


def normalize_minmax(cube: np.ndarray) -> np.ndarray:
    """Per-band min-max normalization to [0, 1]."""
    h, w, c = cube.shape
    flat = cube.reshape(-1, c)
    band_min = flat.min(axis=0, keepdims=True)
    band_max = flat.max(axis=0, keepdims=True)
    denom = band_max - band_min
    denom[denom <= 0] = 1.0
    normalized = (flat - band_min) / denom
    return normalized.reshape(h, w, c).astype(np.float32)


def normalize_l2(cube: np.ndarray) -> np.ndarray:
    """Per-pixel L2 normalization (unit vector)."""
    h, w, c = cube.shape
    flat = cube.reshape(-1, c)
    norms = np.linalg.norm(flat, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    normalized = flat / norms
    return normalized.reshape(h, w, c).astype(np.float32)


def normalize_reference(cube: np.ndarray, ref: np.ndarray, multiplier: float = 1.0, eps: float = 1e-8) -> np.ndarray:
    """White-reference normalization: cube / (ref * multiplier + eps)."""
    denom = ref * multiplier + eps
    return (cube / denom).astype(np.float32)


def normalize_mean(cube: np.ndarray) -> np.ndarray:
    """Per-pixel mean-centering: spectrum - mean(spectrum)."""
    h, w, c = cube.shape
    flat = cube.reshape(-1, c)
    means = flat.mean(axis=1, keepdims=True)
    centered = flat - means
    return centered.reshape(h, w, c).astype(np.float32)


def spectral_angle(spectra: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Compute spectral angle between each row of *spectra* and *reference*.

    Args:
        spectra: (N, C) array of spectra
        reference: (C,) reference spectrum

    Returns:
        (N,) array of angles in radians
    """
    ref_norm = np.linalg.norm(reference)
    if ref_norm < 1e-12:
        raise ValueError("Reference spectrum has zero norm")
    spec_norms = np.linalg.norm(spectra, axis=1)
    dots = spectra @ reference
    cos_angles = dots / (spec_norms * ref_norm + 1e-12)
    cos_angles = np.clip(cos_angles, -1.0, 1.0)
    return np.arccos(cos_angles).astype(np.float32)


def sid_distance(spectra: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Compute Spectral Information Divergence between spectra and centroids.

    Args:
        spectra: (N, C) array of spectra (must be non-negative)
        centroids: (K, C) array of cluster centroids

    Returns:
        (N, K) array of SID values
    """
    eps = 1e-12
    p = spectra + eps
    p = p / p.sum(axis=1, keepdims=True)

    q = centroids + eps
    q = q / q.sum(axis=1, keepdims=True)

    log_p = np.log(p)
    log_q = np.log(q)

    d_pq = np.sum(p[:, None, :] * (log_p[:, None, :] - log_q[None, :, :]), axis=2)
    d_qp = np.sum(q[None, :, :] * (log_q[None, :, :] - log_p[:, None, :]), axis=2)

    return (d_pq + d_qp).astype(np.float32)


def kmeans_plusplus_init(X: np.ndarray, n_clusters: int, metric: str, random_state: int | None = None) -> np.ndarray:
    """K-means++ initialization for arbitrary distance metrics.

    Args:
        X: (N, C) data
        n_clusters: number of clusters
        metric: one of 'sid', 'manhattan', 'chebyshev'
        random_state: random seed

    Returns:
        (n_clusters, C) initial centroids
    """
    rng = np.random.RandomState(random_state)
    n = X.shape[0]

    first_idx = rng.randint(0, n)
    centroids = [X[first_idx]]

    for _ in range(1, n_clusters):
        if metric == "sid":
            dists = sid_distance(X, np.array(centroids))
            min_dists = dists.min(axis=1)
        elif metric == "manhattan":
            from scipy.spatial.distance import cdist
            dists = cdist(X, np.array(centroids), metric="cityblock")
            min_dists = dists.min(axis=1)
        elif metric == "chebyshev":
            from scipy.spatial.distance import cdist
            dists = cdist(X, np.array(centroids), metric="chebyshev")
            min_dists = dists.min(axis=1)
        else:
            raise ValueError(f"kmeans++ init not supported for metric '{metric}'")

        min_dists = np.maximum(min_dists, 0.0)
        total = min_dists.sum()
        if total <= 0 or not np.isfinite(total):
            next_idx = rng.randint(0, n)
        else:
            prob = min_dists / total
            next_idx = rng.choice(n, p=prob)
        centroids.append(X[next_idx])

    return np.array(centroids, dtype=np.float32)
