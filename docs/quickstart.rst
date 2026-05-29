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

Spatial crop
------------

.. code-block:: python

   # Extract a spatial subregion
   cropped = hsi.crop(y1=50, y2=200, x1=100, x2=300)

Masked spectra extraction
-------------------------

Extract spectra at selected pixels as a 2D array for downstream
analysis (e.g. scikit-learn, plotting):

.. code-block:: python

   # Boolean mask
   mask = np.zeros((hsi.shape[0], hsi.shape[1]), dtype=bool)
   mask[10:20, 30:40] = True
   spectra = hsi.mask_spectra(mask)  # shape (N, C)

   # From a label map + label value
   labels, info = hsi.kmeans(n_clusters=5)
   class3 = hsi.mask_spectra(labels, label=3)  # shape (N, C)

Predefined spectral indices
----------------------------

.. code-block:: python

   ndvi = hsi.ndvi()
   ndwi = hsi.ndwi()
   mndwi = hsi.mndwi()
   evi = hsi.evi()
   savi = hsi.savi()
   msavi = hsi.msavi()
   mcari = hsi.mcari()
   pri = hsi.pri()

   # Custom wavelengths
   ndvi_custom = hsi.ndvi(nir=850.0, red=650.0)

Spectral smoothing and derivatives
-----------------------------------

.. code-block:: python

   # Smoothing
   smoothed = hsi.filter_savgol(window_length=15, polyorder=3)

   # First derivative (physically scaled by wavelength spacing in nm)
   deriv1 = hsi.filter_savgol(window_length=15, polyorder=3, deriv=1, delta=2.0)

   # Second derivative
   deriv2 = hsi.filter_savgol(window_length=15, polyorder=3, deriv=2)

   # Window length in nanometers instead of band count
   smoothed = hsi.filter_savgol(window_length_nm=50.0, polyorder=3)

Continuum removal
-----------------

Remove the convex-hull continuum from spectra.  Absorption features
appear as dips below 1.0.

.. code-block:: python

   # Fast mode: compute continuum from mean spectrum (default)
   removed = hsi.continuum_remove()

   # Per-pixel mode: accurate but slow for large images
   removed = hsi.continuum_remove(per_pixel=True)

Normalization
-------------

.. code-block:: python

   # Per-band min-max to [0, 1]
   normed = hsi.normalize("minmax")

   # Per-pixel L2 (unit vector)
   normed = hsi.normalize("l2")

   # White-reference normalization
   normed = hsi.normalize("reference")  # uses stored reference_spectrum
   normed = hsi.normalize("reference", reference_spectrum=ref_array)

   # Per-pixel mean-centering
   normed = hsi.normalize("mean")

Spectral Angle Mapper
---------------------

.. code-block:: python

   # Using a stored reference spectrum
   angles = hsi.sam()

   # Using an explicit reference
   angles = hsi.sam(reference=my_reference_spectrum)

   # angles is a 2D array in radians
   angles_degrees = np.degrees(angles)

Band statistics
---------------

.. code-block:: python

   mean = hsi.mean_spectrum()   # shape (C,)
   std = hsi.band_std()         # shape (C,)
   cov = hsi.band_cov()         # shape (C, C)

PCA and MNF transforms
----------------------

.. code-block:: python

   # PCA — reduce to 10 components
   pca_hsi = hsi.pca(n_components=10)
   print(pca_hsi._pca_explained_variance_ratio_)

   # MNF — noise-adjusted transform (better SNR ordering)
   mnf_hsi = hsi.mnf(n_components=10)
   print(mnf_hsi._mnf_eigenvalues_)

K-means clustering
------------------

.. code-block:: python

   # Euclidean K-means
   labels, info = hsi.kmeans(n_clusters=5, metric="euclidean")

   # Spectral Angle Mapper distance
   labels, info = hsi.kmeans(n_clusters=5, metric="sam")

   # Spectral Information Divergence
   labels, info = hsi.kmeans(n_clusters=5, metric="sid", n_init=3)

   # Other metrics: correlation, manhattan, chebyshev
   labels, info = hsi.kmeans(n_clusters=5, metric="correlation")

   # info contains: centroids, inertia, n_iter, metric
   print(info["centroids"].shape)  # (5, C)

   # Extract spectra for a specific cluster
   class2 = hsi.mask_spectra(labels, label=2)

Spectral smoothing
------------------

.. code-block:: python

   smoothed = hsi.filter_savgol(window_length=15, polyorder=3)

Spatial resize
--------------

.. code-block:: python

   small = hsi.resize(width=256, height=256)
