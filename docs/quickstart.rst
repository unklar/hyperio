Quick Start
===========

Installation
------------

.. code-block:: bash

   pip install hyperio

Reading an HSI image
--------------------

.. code-block:: python

   from hyperio import HSI

   hsi = HSI.read("image.hdr")

   # Access band by wavelength
   red = hsi.nearest_band(670.0)

   # Render an RGB preview
   rgb = hsi.rgb()

   # Compute a spectral index
   ndvi = hsi.compute_index(
       lambda nir, red: (nir - red) / (nir + red + 1e-6),
       nir=800.0,
       red=670.0,
   )

Writing an HSI image
--------------------

The :meth:`~hyperio.hsi.HSI.write` method infers the output format from
the file extension:

.. code-block:: python

   # ENVI format (exact round-trip)
   hsi.write("output.hdr")

   # TIFF format (exact round-trip)
   hsi.write("output.tiff")

   # JPEG 2000 (lossy; wavelengths preserved via band tags)
   hsi.write("output.jp2")

   # HSD format (raw cube; wavelength endpoints only)
   hsi.write("output.hsd")

   # PNG folder (uint16; no wavelength storage)
   hsi.write("output_folder/")          # line_cam=True (default)
   hsi.write("output_folder/", line_cam=False)  # channel-stack mode

JSON sidecar metadata
^^^^^^^^^^^^^^^^^^^^^

Pass ``metadata_json=True`` to write or read a JSON sidecar file
alongside the image.  The sidecar stores the full wavelength vector
and optional reference spectrum, enabling **exact wavelength
round-trips** even for formats that normally lose precision (HSD, PNG
folder):

.. code-block:: python

   # Write with a JSON sidecar
   hsi.write("output.hsd", metadata_json=True)

   # Read it back — wavelengths are loaded from the sidecar
   hsi_back = HSI.read("output.hsd", metadata_json=True)

   # PNG folder: no need to pass wavelengths= explicitly
   hsi.write("output_folder/", metadata_json=True)
   hsi_back = HSI.read("output_folder/", metadata_json=True)

Lower-level writer functions are also available:

.. code-block:: python

   from hyperio.io import write_envi, write_tiff, write_jp2, write_hsd, write_png_folder

   path = write_envi(hsi.cube, hsi.wavelengths, "output.hdr")

Building from raw data
----------------------

.. code-block:: python

   import numpy as np
   from hyperio import HSI

   # From explicit wavelengths
   hsi = HSI.from_cube(cube, wavelengths=[500.0, 510.0, 520.0])

   # From spectral range
   hsi = HSI.from_cube(cube, min_wavelength=400.0, max_wavelength=1000.0)

   # From spectral range with custom resolution
   hsi = HSI.from_cube(
       cube,
       min_wavelength=400.0,
       max_wavelength=1000.0,
       spectral_resolution=5.0,
   )

Spectral smoothing
------------------

.. code-block:: python

   smoothed = hsi.filter_savgol(window_length=15, polyorder=3)

Spatial resize
--------------

.. code-block:: python

   small = hsi.resize(width=256, height=256)
