def valid_audio(data, rate, block_size):
    """Validate input audio data.

    Ensure input has valid dimensions and sufficient length for analysis.

    Args:
        data: Input audio data with shape (channels, samples) or (samples,) for mono.
        rate: Sampling rate of the input audio in Hz.
        block_size: Analysis block size in seconds.

    Returns:
        bool: True if valid audio.

    Raises:
        ValueError: If audio has more than 5 channels or length is less than block size.
    """
    if data.ndim == 2 and data.shape[0] > 5:
        raise ValueError("Audio must have five channels or less.")

    # For (C, T) format, time is the last dimension
    time_axis = -1 if data.ndim == 2 else 0
    if data.shape[time_axis] < block_size * rate:
        raise ValueError("Audio must have length greater than the block size.")

    return True
