Supported File Formats
======================

hyperio supports reading **and writing** hyperspectral data in the
following formats:

.. list-table::
   :header-rows: 1
   :widths: 15 15 20 20 30

   * - Format
     - Extension(s)
     - Reader
     - Writer
     - Wavelength Storage
   * - ENVI
     - ``.hdr``
     - :func:`~hyperio.io.read_envi`
     - :func:`~hyperio.io.write_envi`
     - ENVI header metadata (exact)
   * - JPEG 2000
     - ``.jp2``
     - :func:`~hyperio.io.read_jp2`
     - :func:`~hyperio.io.write_jp2`
     - Rasterio band tags (exact); lossy compression
   * - TIFF
     - ``.tif``, ``.tiff``
     - :func:`~hyperio.io.read_tiff`
     - :func:`~hyperio.io.write_tiff`
     - ImageJ metadata tag (exact)
   * - HSD (HSICityV2)
     - ``.hsd``
     - :func:`~hyperio.io.read_hsd`
     - :func:`~hyperio.io.write_hsd`
     - Integer start/end endpoints in header (approximate)
   * - PNG folder (line-scan)
     - directory
     - :func:`~hyperio.io.read_line_scan_png_folder`
     - :func:`~hyperio.io.write_png_folder`
     - None (must provide externally)
   * - PNG folder (channel-stack)
     - directory
     - :func:`~hyperio.io.read_line_scan_png_folder`
     - :func:`~hyperio.io.write_png_folder`
     - None (must provide externally); use ``line_cam=False``

Unified reader / writer
-----------------------

The :meth:`~hyperio.hsi.HSI.read` class method and
:func:`~hyperio.io.read_auto` function auto-detect the format from the
file extension or directory path and dispatch to the appropriate reader.

Similarly, :meth:`~hyperio.hsi.HSI.write` and
:func:`~hyperio.io.write_auto` dispatch to the appropriate writer
based on the file extension:

- ``.hdr`` → :func:`~hyperio.io.write_envi`
- ``.tif`` / ``.tiff`` → :func:`~hyperio.io.write_tiff`
- ``.jp2`` → :func:`~hyperio.io.write_jp2`
- ``.hsd`` → :func:`~hyperio.io.write_hsd`
- directory (or no extension) → :func:`~hyperio.io.write_png_folder`

Wavelength resolution order
---------------------------

For every **reader**, wavelengths are resolved in the following priority
order:

1. **Explicit** ``wavelengths`` parameter (always wins if provided).
2. **Range** ``min_wavelength`` + ``max_wavelength`` (linearly spaced).
3. **Embedded** metadata extracted from the file itself (format-specific).

If none are available, a ``ValueError`` is raised.

For every **writer**, wavelengths are embedded in the format's native
metadata store where possible:

- **ENVI**: full wavelength vector in the ``.hdr`` file (exact).
- **TIFF**: full wavelength vector in ImageJ metadata (exact).
- **JP2**: full wavelength vector as JSON in rasterio band tag (exact);
  however pixel data is lossy-compressed and quantized to uint16.
- **HSD**: only integer start/end endpoints in the header (approximate;
  intermediate wavelengths are reconstructed via ``linspace``).
- **PNG folder**: no wavelength storage — wavelengths must be persisted
  separately by the caller.

Round-trip fidelity
-------------------

The following table summarizes expected round-trip fidelity when writing
and reading back data:

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Format
     - Cube Fidelity
     - Wavelength Fidelity
   * - ENVI
     - Exact (float32)
     - Exact
   * - TIFF
     - Exact (float32)
     - Exact
   * - JP2
     - Approximate (lossy compression + uint16 quantization)
     - Exact (via band tags)
   * - HSD
     - Exact (float32 raw cube)
     - Approximate (integer endpoints only)
   * - PNG folder
     - Approximate (uint16 quantization; values > 1.0 clipped)
     - None
