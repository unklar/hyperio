# hyperio

Python library for hyperspectral image I/O and processing with a NumPy-like `HSI` class.

## Features

- Internal cube storage always as `float32` with shape `(H, W, C)`.
- Integer input is automatically normalized to `[0.0, 1.0]`.
- Wavelength-aware band access and RGB rendering.
- Readers **and writers** for ENVI, JPEG 2000, TIFF, HSD, and PNG folder formats.
- **JSON sidecar metadata** (`metadata_json=True`) for exact wavelength round-trips across all formats.
- Unified read/write API with optional explicit wavelength overrides.
- **Predefined spectral indices**: NDVI, NDWI, MNDWI, EVI, SAVI, MSAVI, MCARI, PRI.
- **Continuum removal** (convex-hull normalization) in fast or per-pixel mode.
- **Normalization**: min-max, L2, white-reference, mean-centering.
- **Flat-field correction**: optional dark (black) reference for
  `R = (raw − dark) / (white − dark)` radiometric normalization.
- **Savitzky-Golay filtering** with derivative support and wavelength-based window sizing.
- **Spectral Angle Mapper** (SAM).
- **Band statistics**: mean, standard deviation, covariance.
- **PCA and MNF** transforms.
- **K-means clustering** with six distance metrics (Euclidean, SAM, correlation, SID, Manhattan, Chebyshev).
- **Masked spectra extraction** for downstream analysis.
- Spatial cropping and resizing.

## Supported File Formats

| Format | Extension(s) | Reader | Writer | Wavelength Storage | Round-trip Fidelity |
|---|---|---|---|---|---|
| ENVI | `.hdr` | `read_envi` | `write_envi` | ENVI header (exact) | Exact (float32) |
| JPEG 2000 | `.jp2` | `read_jp2` | `write_jp2` | Rasterio band tags (exact) | Approximate (lossy + uint16) |
| TIFF | `.tif`, `.tiff` | `read_tiff` | `write_tiff` | ImageJ metadata (exact) | Exact (float32) |
| HSD (HSICityV2) | `.hsd` | `read_hsd` | `write_hsd` | Integer start/end (approximate) | Exact (float32 raw cube) |
| PNG folder (line-scan) | directory | `read_line_scan_png_folder` | `write_png_folder` | None | Approximate (uint16; clips >1.0) |
| PNG folder (channel-stack) | directory | `read_line_scan_png_folder` | `write_png_folder` | None | Approximate (uint16; clips >1.0); use `line_cam=False` |

## Installation

```bash
pip install git+https://github.com/unklar/hyperio.git
```

## Quick Usage

```python
from hyperio import HSI

hsi = HSI.read("image.hdr")
red = hsi.nearest_band(670.0)
rgb = hsi.rgb()

# NDVI
ndvi = hsi.ndvi()

# Build from raw data with explicit wavelengths
hsi_from_wl = HSI.from_cube(cube, wavelengths=[500.0, 510.0, 520.0])

# Build from raw data with spectral range
hsi_from_range = HSI.from_cube(
    cube,
    min_wavelength=500.0,
    max_wavelength=520.0,
)
```

## Processing

```python
# Spatial crop
cropped = hsi.crop(50, 200, 100, 300)

# Extract masked spectra as (N, C) array
labels, info = hsi.kmeans(n_clusters=5)
class3 = hsi.mask_spectra(labels, label=3)

# Predefined indices
ndvi = hsi.ndvi()
evi = hsi.evi()

# Spectral smoothing and derivatives
smoothed = hsi.filter_savgol(window_length=15, polyorder=3)
deriv1 = hsi.filter_savgol(window_length=15, polyorder=3, deriv=1, delta=2.0)
smoothed_nm = hsi.filter_savgol(window_length_nm=50.0, polyorder=3)

# Continuum removal
removed = hsi.continuum_remove()

# Normalization
normed = hsi.normalize("minmax")
normed = hsi.normalize("l2")
normed = hsi.normalize("reference")
normed = hsi.normalize("mean")

# Spectral Angle Mapper
angles = hsi.sam(reference=my_ref)

# Band statistics
mean = hsi.mean_spectrum()
std = hsi.band_std()
cov = hsi.band_cov()

# PCA and MNF
pca_hsi = hsi.pca(n_components=10)
mnf_hsi = hsi.mnf(n_components=10)

# K-means clustering
labels, info = hsi.kmeans(n_clusters=5, metric="euclidean")
labels, info = hsi.kmeans(n_clusters=5, metric="sam")
labels, info = hsi.kmeans(n_clusters=5, metric="sid")

# Resize and rescale
small = hsi.resize(width=256, height=256)
half = hsi.rescale(0.5)
```

## Writing HSI Data

The `HSI.write` method infers the output format from the file extension:

```python
# ENVI format (exact round-trip)
hsi.write("output.hdr")

# TIFF format (exact round-trip)
hsi.write("output.tiff")

# JPEG 2000 (lossy; wavelengths preserved via band tags)
hsi.write("output.jp2")

# HSD format (raw cube; wavelength endpoints only)
hsi.write("output.hsd")

# PNG folder (uint16; no wavelength storage)
hsi.write("output_folder/")                     # line_cam=True (default)
hsi.write("output_folder/", line_cam=False)     # channel-stack mode
```

Lower-level writer functions are also available:

```python
from hyperio.io import write_envi, write_tiff, write_jp2, write_hsd, write_png_folder, write_auto

path = write_envi(hsi.cube, hsi.wavelengths, "output.hdr")
```

## JSON Sidecar Metadata

Pass `metadata_json=True` to write or read a JSON sidecar file alongside the image. The sidecar stores the full wavelength vector and optional reference spectrum, enabling **exact wavelength round-trips** even for formats that normally lose precision (HSD, PNG folder):

```python
# Write with a JSON sidecar
hsi.write("output.hsd", metadata_json=True)

# Read it back — wavelengths are loaded from the sidecar
hsi_back = HSI.read("output.hsd", metadata_json=True)

# PNG folder: no need to pass wavelengths= explicitly
hsi.write("output_folder/", metadata_json=True)
hsi_back = HSI.read("output_folder/", metadata_json=True)
```

Sidecar naming: `cube.hdr` → `cube.json`, `cube.jp2` → `cube.json`, `folder/` → `folder/metadata.json`.

Sidecar contents (only `wavelengths` is required; other fields are included when non-default):

```json
{
  "wavelengths": [383.83, 385.86, ...],
  "reference_spectrum": [17.0, 19.0, ...],
  "reference_multiplier": 2.0,
  "reference_eps": 1e-6,
  "dark_reference": [0.0, 0.0, ...]
}
```

## Dark Reference / Flat-Field Correction

For radiometrically calibrated reflectance, pass a dark (black) reference
spectrum.  When present, the white-reference normalization becomes the
standard flat-field correction `R = (raw − dark) / (white − dark)`:

```python
import numpy as np
from hyperio import HSI

dark = np.load("black_reference.npy")   # one value per band, same units as raw

# Explicit: apply only to this read
hsi = HSI.read("image.jp2", dark_reference=dark)

# Automatic: store the dark reference in the image metadata / JSON sidecar,
# and it is applied on every subsequent read.
hsi.write("image.jp2", metadata_json=True, dark_reference=dark)
hsi2 = HSI.read("image.jp2", metadata_json=True)   # dark applied automatically
print(hsi2.dark_reference.shape)
```

The dark reference is carried by `ReadResult`, `Jp2Metadata` and `HSI`
(`hsi.dark_reference`), and is preserved through `copy()`, cropping and the
other HSI-transforming methods.

## I/O Functions

Lower-level readers are available directly:

```python
from hyperio.io import read_envi, read_tiff, read_jp2, read_hsd, read_line_scan_png_folder, read_auto

result = read_envi("cube.hdr")
print(result.cube.shape, result.wavelengths.shape)
```

Each reader returns a `ReadResult(cube, wavelengths, reference_spectrum, reference_multiplier, reference_eps)` dataclass.

## K-means Distance Metrics

| Metric | Description | Implementation |
|--------|-------------|----------------|
| `euclidean` | Standard Euclidean distance | scikit-learn KMeans |
| `sam` | Spectral Angle Mapper | L2-normalize then Euclidean |
| `correlation` | Spectral correlation | Mean-center + L2-normalize then Euclidean |
| `sid` | Spectral Information Divergence | Custom with k-means++ init |
| `manhattan` | L1 / cityblock distance | Custom with k-means++ init |
| `chebyshev` | L-infinity distance | Custom with k-means++ init |

## Documentation

The Sphinx docs sources live in the `docs/` folder. To build them locally:

```bash
pip install hyperio[docs]
cd docs && sphinx-build -b html . _build/html
```

## License

BSD 3-Clause License. See [LICENSE](LICENSE) for details.
