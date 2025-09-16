jaxloudnorm Documentation
=========================

A JAX-based implementation of the ITU-R BS.1770-4 loudness algorithm.

jaxloudnorm is a port of pyloudnorm that leverages JAX for GPU acceleration and vectorized operations, providing significant speedups for batch processing of audio files.

Features
--------

* ITU-R BS.1770-4 compliant loudness measurement
* GPU acceleration through JAX
* Support for batch processing with ``jax.vmap``
* Support for various filter classes (K-weighting, Fenton/Lee, DeMan)
* FIR approximation mode for faster GPU computation
* Peak and loudness normalization utilities

Installation
------------

.. code-block:: bash

   pip install jaxloudnorm

Or install from GitHub:

.. code-block:: bash

   pip install git+https://github.com/DBraun/jaxloudnorm

Quick Start
-----------

.. code-block:: python

   import soundfile as sf
   import jaxloudnorm as jln

   # Load audio
   data, rate = sf.read("audio.wav")
   data = data.T # convert to (channels, samples) for jaxloudnorm

   # Create meter
   meter = jln.Meter(rate, use_fir=True)  # Use FIR for GPU speedup

   # Measure loudness
   loudness = meter.integrated_loudness(data)

   # Normalize audio
   loudness_normalized = jln.normalize.loudness(data, loudness, -14.0)
   peak_normalized = jln.normalize.peak(data, -1.0)

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/meter
   api/normalize
   api/iirfilter
   api/lfilter
   api/util

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`