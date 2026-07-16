# Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
(categories: Added, Changed, Deprecated, Removed, Fixed, Security), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.3.2 - 2026-07-16

### Removed

- `pysoundfile` is no longer a runtime dependency. The library never imported it — it was used only by the test suite and the README examples — so `soundfile>=0.12.1` now lives in the `test` extra under the project's current name.

### Fixed

- Installing jaxloudnorm no longer clobbers `soundfile` for other packages in the same environment. `pysoundfile` is the pre-2017 name of the project now published as `python-soundfile`; both distributions install a top-level `soundfile.py`, so pip silently overwrote a modern `soundfile` with the 0.9.0.post1 module from 2017, without the bundled libsndfile that current versions ship. Environments sharing jaxloudnorm with librosa, torchaudio, or anything else built on `soundfile` were affected.

  Upgrading does not repair an environment that already installed 0.3.1 or earlier, because removing `pysoundfile` cannot restore the files it overwrote. Run `pip install --force-reinstall soundfile` to do that.

## 0.3.1 - 2026-06-29

### Fixed

- `Meter.integrated_loudness` now returns `-inf` LUFS for digital silence instead of `NaN`. When no gating block exceeds the −70 LKFS absolute threshold (e.g. pure silence), the gated mean previously divided a zero sum by a zero count (`0/0`); the empty-gate division is now guarded so the result is the mathematical limit of zero gated power, `-inf`. Applies to both the direct and `jax.vmap`-batched paths and both `use_fir` settings.

## 0.3.0 - 2025-09-16

First release tracked in this changelog. For earlier history, see the git log and the `v0.1.1` / `v0.1.0` tags.
