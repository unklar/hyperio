Changelog
=========

v0.2.0 (2025)
--------------

Added write support for all supported formats:

- ``write_envi`` — ENVI format with full wavelength metadata
- ``write_tiff`` — TIFF with ImageJ metadata wavelength storage
- ``write_jp2`` — JPEG 2000 with rasterio band tags (lossy)
- ``write_hsd`` — HSD raw-cube variant (no PCA compression)
- ``write_png_folder`` — folder of uint16 PNG images
- ``write_auto`` — dispatch writer based on file extension
- ``HSI.write`` — convenience method that dispatches by extension

Added JSON sidecar metadata (``metadata_json``):

- All readers and writers accept ``metadata_json=True`` to write/read
  a ``.json`` sidecar file alongside the image
- Sidecar stores full wavelength vector, reference spectrum,
  ``reference_multiplier``, and ``reference_eps``
- Enables exact wavelength round-trips for HSD and PNG folder formats
- ``ReadResult`` now carries ``reference_spectrum``,
  ``reference_multiplier``, and ``reference_eps`` fields
- ``HSI`` class now stores and forwards reference spectrum data on
  write

Other improvements:

- ``read_jp2`` now falls back to rasterio band tags when no
  XML/UUID wavelength metadata is found
- ``write_jp2`` converts float32 → uint16 for JP2OpenJPEG compatibility
- HSD writer uses ``round()`` instead of ``int()`` for wavelength
  endpoints to reduce rounding error
- Full round-trip test suite (19 tests) covering all formats
- JSON sidecar test suite (21 tests) covering all formats

v0.1.0 (2025)
--------------

Initial release of ``hyperio``, a fork and rename of ``use-hsi-io`` with:

- ``HSI`` class with NumPy-like interface
- Readers for ENVI, JP2, TIFF, HSD, and PNG folder formats
- Savitzky-Golay spectral smoothing
- Spectral index computation (``compute_index``)
- Spatial resizing and value rescaling
- RGB rendering with percentile stretching
