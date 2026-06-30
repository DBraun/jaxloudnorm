# Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
(categories: Added, Changed, Deprecated, Removed, Fixed, Security), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.3.1 - 2026-06-29

### Fixed

- `Meter.integrated_loudness` now returns `-inf` LUFS for digital silence instead of `NaN`. When no gating block exceeds the −70 LKFS absolute threshold (e.g. pure silence), the gated mean previously divided a zero sum by a zero count (`0/0`); the empty-gate division is now guarded so the result is the mathematical limit of zero gated power, `-inf`. Applies to both the direct and `jax.vmap`-batched paths and both `use_fir` settings.

## 0.3.0 - 2025-09-16

First release tracked in this changelog. For earlier history, see the git log and the `v0.1.1` / `v0.1.0` tags.
