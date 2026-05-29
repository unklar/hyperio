# hyperio

Python library for hyperspectral image I/O and processing with a NumPy-like `HSI` class.

## Features

- Internal cube storage always as `float32` with shape `(H, W, C)`.
- Integer input is automatically normalized to `[0.0, 1.0]`.
- Wavelength-aware band access and RGB rendering.
- Readers **and writers** for ENVI, JPEG 2000, TIFF, HSD, and PNG folder formats.
- Unified read/write API with optional explicit wavelength overrides.
- Savitzky-Golay spectral smoothing, spatial resizing, and spectral index computation.

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
