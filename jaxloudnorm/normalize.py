import warnings
import jax.numpy as jnp


def peak(data, target):
    """Peak normalize a signal.

    Normalize an input signal to a user specified peak amplitude.

    Args:
        data: Input multichannel audio data.
        target: Desired peak amplitude in dB.

    Returns:
        ndarray: Peak normalized output data.
    """
    # find the amplitude of the largest peak
    current_peak = jnp.max(jnp.abs(data))

    # calculate the gain needed to scale to the desired peak level
    gain = jnp.power(10.0, target / 20.0) / current_peak
    output = gain * data

    # check for potentially clipped samples
    # if jnp.max(jnp.abs(output)) >= 1.0:
    #     warnings.warn("Possible clipped samples in output.")

    return output


def loudness(data, input_loudness, target_loudness):
    """Loudness normalize a signal.

    Normalize an input signal to a target loudness in dB LUFS.

    Args:
        data: Input multichannel audio data.
        input_loudness: Loudness of the input in dB LUFS.
        target_loudness: Target loudness of the output in dB LUFS.

    Returns:
        ndarray: Loudness normalized output data.
    """
    # calculate the gain needed to scale to the desired loudness level
    delta_loudness = target_loudness - input_loudness
    gain = jnp.power(10.0, delta_loudness / 20.0)

    output = gain * data

    # check for potentially clipped samples
    # if jnp.max(jnp.abs(output)) >= 1.0:
    #     warnings.warn("Possible clipped samples in output.")

    return output
