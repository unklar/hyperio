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

JSON sidecar metadata (``metadata_json``)
------------------------------------------

All readers and writers accept a ``metadata_json=True`` argument that
enables a **JSON sidecar file** alongside the image.  The sidecar
stores the full wavelength vector, reference spectrum, and
normalization parameters in an unambiguous, format-independent way.

**Sidecar naming convention:**

- For file paths: ``cube.hdr`` → ``cube.json``, ``cube.jp2`` → ``cube.json``, etc.
- For directories (PNG folder): ``folder/`` → ``folder/metadata.json``

**Sidecar contents** (all fields except ``wavelengths`` are optional):

.. code-block:: json

   {
     "wavelengths": [383.83, 385.86, ...],
     "reference_spectrum": [17.0, 19.0, ...],
     "reference_multiplier": 2.0,
     "reference_eps": 1e-6
   }

When ``metadata_json=True`` is passed to a **reader**, the sidecar is
loaded and its wavelength data takes priority over any embedded
format-specific metadata (but is still overridden by an explicit
``wavelengths`` parameter).  If the sidecar does not exist, the reader
falls back to embedded metadata as usual.

When ``metadata_json=True`` is passed to a **writer**, the sidecar is
written in addition to the format-specific image file.  This is
especially useful for:

- **HSD**: preserves the full wavelength vector instead of only integer
  endpoints, enabling exact round-trips.
- **PNG folders**: stores wavelengths so that ``read`` no longer
  requires an explicit ``wavelengths`` argument.
- **JP2**: provides a reliable, easily-parseable copy of the wavelength
  data that does not depend on rasterio band tag support.

The ``HSI.write`` method forwards ``metadata_json`` and any reference
spectrum attached to the ``HSI`` instance:

.. code-block:: python

   # Write with a JSON sidecar
   hsi.write("cube.hsd", metadata_json=True)

   # Read it back — no explicit wavelengths needed even for HSD
   hsi_back = HSI.read("cube.hsd", metadata_json=True)
   np.testing.assert_allclose(hsi_back.wavelengths, hsi.wavelengths)

Lower-level writer functions also accept ``metadata_json``,
``reference_spectrum``, ``reference_multiplier``, and ``reference_eps``:

.. code-block:: python

   from hyperio.io import write_hsd

   write_hsd(
       cube, wavelengths, "cube.hsd",
       metadata_json=True,
       reference_spectrum=ref,
       reference_multiplier=2.0,
       reference_eps=1e-6,
   )

Wavelength resolution order
---------------------------

For every **reader**, wavelengths are resolved in the following priority
order:

1. **Explicit** ``wavelengths`` parameter (always wins if provided).
2. **Range** ``min_wavelength`` + ``max_wavelength`` (linearly spaced).
3. **JSON sidecar** (when ``metadata_json=True`` and the ``.json`` file exists).
4. **Embedded** metadata extracted from the file itself (format-specific).

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
  separately by the caller (or via ``metadata_json=True``).

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
     - Exact (via band tags or JSON sidecar)
   * - HSD
     - Exact (float32 raw cube)
     - Approximate (integer endpoints only); **exact with ``metadata_json=True``**
   * - PNG folder
     - Approximate (uint16 quantization; values > 1.0 clipped)
     - None (without sidecar); **exact with ``metadata_json=True``**
