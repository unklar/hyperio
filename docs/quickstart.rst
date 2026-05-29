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
