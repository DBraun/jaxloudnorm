# note: to be removed from here once I move it to its own package

import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.signal import convolve
from jax._src.util import canonicalize_axis
from jax._src.api_util import _ensure_index_tuple

from jaxtyping import ArrayLike, Array

from typing import Sequence, Set


def _axes_to_mapped_axes(axes: Sequence[int] | int, ndim: int) -> Set[int]:
    """Convert axes to mapped axes for vmap operations.

    Args:
        axes: Axis or sequence of axes to operate on.
        ndim: Number of dimensions in the array.

    Returns:
        Set[int]: Set of axes to map over (all axes except the specified ones).
    """
    axes = _ensure_index_tuple(axes)
    axes = tuple(canonicalize_axis(ax, ndim) for ax in axes)
    return set(range(ndim)) - set(axes)


def lfilter(
    b: ArrayLike, a: ArrayLike, x: ArrayLike, axis: int = -1, zi: Array | None = None
) -> Array | tuple[Array, Array]:
    """Apply a digital IIR or FIR filter to a signal.

    Filter a data sequence using a digital filter. This function applies a
    linear digital filter twice, once forward and once backwards. The combined
    filter has zero phase and a filter order twice that of the original.

    Args:
        b: Numerator coefficient array of the filter.
        a: Denominator coefficient array of the filter. If a[0] is not 1,
            both a and b are normalized by a[0].
        x: Input signal array to be filtered.
        axis: Axis along which to apply the filter. Defaults to -1.
        zi: Initial conditions for the filter delays. If None, initial rest
            is assumed. Defaults to None.

    Returns:
        If zi is None, returns the filtered signal array.
        If zi is provided, returns a tuple of (filtered_signal, final_conditions).
    """
    a_: Array = jnp.atleast_1d(jnp.asarray(a))
    b_: Array = jnp.atleast_1d(jnp.asarray(b))
    x_: Array = jnp.atleast_1d(jnp.asarray(x))

    if len(a_) == 1:
        b_ /= a_
        out_full = jnp.apply_along_axis(
            lambda y: convolve(b_, y, mode="full"), axis, x_
        )

        ind = out_full.ndim * [slice(None)]

        if zi is not None:
            ind[axis] = slice(zi.shape[axis])
            out_full = out_full.at[tuple(ind)].set(out_full[tuple(ind)] + zi)

        ind[axis] = slice(out_full.shape[axis] - len(b_) + 1)
        out = out_full[tuple(ind)]

        if zi is None:
            return out
        else:
            ind[axis] = slice(out_full.shape[axis] - len(b_) + 1, None)
            zf = out_full[tuple(ind)]
            return out, zf

    else:
        max_coef_len = max(len(b_), len(a_))

        b_, a_ = (
            x if len(x) == max_coef_len else jnp.pad(x, (0, max_coef_len - len(x)))
            for x in (b_, a_)
        )

        b_, a_ = (x / a_[0] for x in (b_, a_))

        _lfilter = lambda x, zi: _lfilter_unbatched(
            b_,
            a_,
            x,
            zi=zi if zi is not None else jnp.zeros_like(x, shape=max_coef_len - 1),
        )

        mapped_axes = _axes_to_mapped_axes(axis, x_.ndim)

        for axis in sorted(mapped_axes):
            _lfilter = jax.vmap(_lfilter, in_axes=axis, out_axes=axis)
        out, zi_ = _lfilter(x_, zi)

        if zi is None:
            return out
        else:
            return out, zi_


def _lfilter_unbatched(b: Array, a: Array, x: Array, zi: Array) -> tuple[Array, Array]:
    """Apply IIR filter to unbatched 1D signal.

    Internal function that performs the actual IIR filtering on a single
    1D signal without batching.

    Args:
        b: Normalized numerator coefficients.
        a: Normalized denominator coefficients (flipped internally).
        x: 1D input signal.
        zi: Initial filter state.

    Returns:
        tuple: A tuple containing:
            - out: Filtered output signal
            - zi: Final filter state
    """

    original_length = x.shape[0]

    # Convolve with just the B feedforward coefficients.
    x = convolve(x, b, mode='full', method='direct')
    x = x[:original_length]

    # flip coefficients to make the recursive `tick_feedback` function easier
    a = jnp.flip(a)

    def tick_feedback(state: Array, x_: Array) -> tuple[Array, Array]:
        y = x_ - jnp.dot(state, a[:-1])  # assume a[-1] is 1.
        state = jnp.concatenate([state[1:], y.reshape(-1)])

        return state, y

    zi, out = lax.scan(tick_feedback, zi, x)

    return out, zi


def approximate_iir_as_fir(b: Array, a: Array, data: Array, zeros: Array, axis=-1):
    """Approximate IIR filter as FIR filter for faster computation.

    Compute impulse response of IIR filter and perform filtering as a
    convolution, avoiding the sequentiality of lfilter. This is particularly
    useful for GPU computation where convolution can be highly parallelized.

    Inspired by Descript AudioTools:
    https://github.com/descriptinc/audiotools/blob/7776c296c711db90176a63ff808c26e0ee087263/audiotools/core/loudness.py#L52-L100

    Args:
        b: Numerator coefficients of the IIR filter.
        a: Denominator coefficients of the IIR filter.
        data: Input signal to filter.
        zeros: Number of zeros for the impulse response (longer = more accurate).
        axis: Axis along which to apply the filter. Defaults to -1.

    Returns:
        Array: Filtered signal approximated using FIR convolution.
    """

    impulse = jnp.concatenate([jnp.ones((1,)), jnp.zeros(shape=(zeros - 1,))])

    impulse_response = lfilter(b, a, impulse)

    # Expand impulse response to broadcast correctly with data
    # For axis=-1 (time axis), add channel dimension at axis 0
    # For other axes, adjust accordingly
    if axis == -1 or axis == data.ndim - 1:
        impulse_response = jnp.expand_dims(impulse_response, axis=0)
    else:
        impulse_response = jnp.expand_dims(impulse_response, axis=1)

    original_length = data.shape[axis]

    data = jax.scipy.signal.convolve(impulse_response, data, mode='full', method='fft')

    data = jax.lax.dynamic_slice_in_dim(data, 0, original_length, axis=axis)

    return data
