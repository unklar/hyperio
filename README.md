# hyperio

Python library for hyperspectral image I/O and processing with a NumPy-like `HSI` class.

## Features

- Internal cube storage always as `float32` with shape `(H, W, C)`.
- Integer input is automatically normalized to `[0.0, 1.0]`.
- Wavelength-aware band access and RGB rendering.
- Readers for ENVI, JPEG 2000, TIFF, HSD, and PNG folder formats.
- Unified read API with optional explicit wavelength overrides.
- Savitzky-Golay spectral smoothing, spatial resizing, and spectral index computation.

## Supported File Formats

| Format | Extension(s) | Reader | Wavelength Source | Notes |
|---|---|---|---|---|
| ENVI | `.hdr` | `read_envi` | ENVI metadata or explicit | Uses the `spectral` library |
| JPEG 2000 | `.jp2` | `read_jp2` | Embedded XML/UUID metadata or explicit | Optional reference-spectrum normalization |
| TIFF | `.tif`, `.tiff` | `read_tiff` | TIFF metadata or explicit | Uses `tifffile` |
| HSD (HSICityV2) | `.hsd` | `read_hsd` | Embedded header or explicit | Reconstructs cube from PCA coefficients |
| PNG folder (line-scan) | directory | `read_line_scan_png_folder` | Explicit only | Each PNG = one scan line |
| PNG folder (channel-stack) | directory | `read_line_scan_png_folder` | Explicit only | Each PNG = one spectral channel; use `line_cam=False` |

## Installation

```bash
pip install hyperio
```

## Quick Usage

```python
from hyperio import HSI

hsi = HSI.read("image.hdr")
red = hsi.nearest_band(670.0)
rgb = hsi.rgb()

# NDVI with compute_index
ndvi = hsi.compute_index(
    lambda nir, red: (nir - red) / (nir + red + 1e-6),
    nir=800.0,
    red=670.0,
)

# MCARI with compute_index
mcari = hsi.compute_index(
    lambda r700, r670, r550: ((r700 - r670) - 0.2 * (r700 - r550)) * (r700 / (r670 + 1e-6)),
    r700=700.0,
    r670=670.0,
    r550=550.0,
)

# Build from raw data with explicit wavelengths
hsi_from_wl = HSI.from_cube(cube, wavelengths=[500.0, 510.0, 520.0])

# Build from raw data with spectral range
hsi_from_range = HSI.from_cube(
    cube,
    min_wavelength=500.0,
    max_wavelength=520.0,
)

# Custom spectral resolution
hsi_from_range_custom = HSI.from_cube(
    cube,
    min_wavelength=500.0,
    max_wavelength=520.0,
    spectral_resolution=10.0,
)
```

## I/O Functions

Lower-level readers are available directly:

```python
from hyperio.io import read_envi, read_tiff, read_jp2, read_hsd, read_line_scan_png_folder, read_auto

result = read_envi("cube.hdr")
print(result.cube.shape, result.wavelengths.shape)
```

Each reader returns a `ReadResult(cube, wavelengths)` dataclass.

## Processing

```python
# Spectral smoothing
smoothed = hsi.filter_savgol(window_length=15, polyorder=3)

# Spatial resize
small = hsi.resize(width=256, height=256)

# Value rescaling
half = hsi.rescale(0.5)
```

## Documentation

Full API documentation is available at [hyperio.readthedocs.io](https://hyperio.readthedocs.io).

To build the docs locally:

```bash
pip install hyperio[docs]
cd docs && make html
```

## License

BSD 3-Clause License. See [LICENSE](LICENSE) for details.
