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

New features:

- ``HSI.subset_wavelengths(min_wl, max_wl)`` — extract a spectral
  region of interest as a new ``HSI`` object
- ``HSI.__eq__`` — value-based equality comparison of cube and
  wavelengths
- ``HSI.__hash__`` raises ``TypeError`` (HSI objects are unhashable)

Bug fixes:

- ``HSI.copy()`` now preserves ``reference_spectrum``,
  ``reference_multiplier``, and ``reference_eps``
- ``read_envi`` no longer crashes with an unhelpful error when the
  ENVI header has no wavelength metadata; it now falls through to
  ``_resolve_wavelengths`` which produces a clear error message
- ``to_float32_cube`` now correctly normalizes signed integer arrays
  (e.g. ``int8``) to ``[0, 1]`` using ``(arr - min) / (max - min)``
  instead of ``arr / max``

Safety and diagnostics:

- ``write_jp2`` and ``write_png_folder`` now log a warning when the
  cube contains values outside ``[0, 1]`` that will be clipped during
  uint16 conversion
- ``write_png_folder`` warns when ``metadata_json`` is not enabled,
  since wavelengths will be lost
- ``_align_wavelength_count`` logs a warning when truncating
  wavelength arrays that exceed the channel count
- ``read_line_scan_png_folder`` now uses ``with Image.open(...)`` to
  ensure file handles are released promptly

Performance:

- ``read_line_scan_png_folder`` now pre-allocates the output cube and
  fills it in-place instead of building a list of arrays then
  stacking (reduces peak memory by ~50%)

Other improvements:

- ``read_jp2`` now falls back to rasterio band tags when no
  XML/UUID wavelength metadata is found
- ``write_jp2`` converts float32 → uint16 for JP2OpenJPEG compatibility
- HSD writer uses ``round()`` instead of ``int()`` for wavelength
  endpoints to reduce rounding error
- Full round-trip test suite (19 tests) covering all formats
- JSON sidecar test suite (21 tests) covering all formats
- Code review fix test suite (13 tests)

v0.1.0 (2025)
--------------

Initial release of ``hyperio``, a fork and rename of ``use-hsi-io`` with:

- ``HSI`` class with NumPy-like interface
- Readers for ENVI, JP2, TIFF, HSD, and PNG folder formats
- Savitzky-Golay spectral smoothing
- Spectral index computation (``compute_index``)
- Spatial resizing and value rescaling
- RGB rendering with percentile stretching
