.. hyperio documentation master file

hyperio
=======

A Python library for hyperspectral image I/O and processing with a
NumPy-like ``HSI`` class.

Features
--------

- Internal cube storage always as ``float32`` with shape ``(H, W, C)``.
- Integer input is automatically normalized to ``[0.0, 1.0]``.
- Wavelength-aware band access and RGB rendering.
- Readers for **ENVI**, **JPEG 2000**, **TIFF**, **HSD**, and **PNG folder** formats.
- Unified read API with optional explicit wavelength overrides.
- Savitzky-Golay spectral smoothing, spatial resizing, and spectral index
  computation.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   quickstart
   formats
   api
   changelog


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
