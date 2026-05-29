Supported File Formats
======================

hyperio supports reading hyperspectral data from the following formats:

.. list-table::
   :header-rows: 1
   :widths: 15 15 25 25 20

   * - Format
     - Extension(s)
     - Reader
     - Wavelength Source
     - Notes
   * - ENVI
     - ``.hdr``
     - :func:`~hyperio.io.read_envi`
     - ENVI metadata or explicit
     - Uses the ``spectral`` library
   * - JPEG 2000
     - ``.jp2``
     - :func:`~hyperio.io.read_jp2`
     - Embedded XML/UUID metadata or explicit
     - Optional reference-spectrum normalization
   * - TIFF
     - ``.tif``, ``.tiff``
     - :func:`~hyperio.io.read_tiff`
     - TIFF metadata or explicit
     - Uses ``tifffile``
   * - HSD (HSICityV2)
     - ``.hsd``
     - :func:`~hyperio.io.read_hsd`
     - Embedded header (startw/endw) or explicit
     - Reconstructs cube from PCA coefficients
   * - PNG folder (line-scan)
     - directory
     - :func:`~hyperio.io.read_line_scan_png_folder`
     - Explicit only
     - Each PNG = one scan line
   * - PNG folder (channel-stack)
     - directory
     - :func:`~hyperio.io.read_line_scan_png_folder`
     - Explicit only
     - Each PNG = one spectral channel; use ``line_cam=False``

Unified reader
--------------

The :meth:`~hyperio.hsi.HSI.read` class method and
:func:`~hyperio.io.read_auto` function auto-detect the format from the
file extension or directory path and dispatch to the appropriate reader.

Wavelength resolution order
---------------------------

For every reader, wavelengths are resolved in the following priority order:

1. **Explicit** ``wavelengths`` parameter (always wins if provided).
2. **Range** ``min_wavelength`` + ``max_wavelength`` (linearly spaced).
3. **Embedded** metadata extracted from the file itself (format-specific).

If none are available, a ``ValueError`` is raised.
