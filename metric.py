import numpy as np

def snr(signal_power, noise_power):
    signal_power = np.nan_to_num(signal_power)
    noise_power = np.nan_to_num(noise_power)

    try:
        snr = 10*np.log10(signal_power/noise_power)
    except ZeroDivisionError:
        snr = 50.0
    return snr

def calculate_snr_per_sample(reconstruction, target):
    """
    Compute the signal-to-noise ratio (SNR) for each channel of the reconstruction.

    Input:
        reconstruction: np.ndarray:np.float32, shape (12, num_samples)
        target: np.ndarray:np.float32, shape (12, num_samples)

    Returns:
        snr: np.ndarray:np.float32, shape (12,)
    """

    assert reconstruction.shape == target.shape, f"Reconstruction and target must have the same shape, got {reconstruction.shape} and {target.shape}"
    assert reconstruction.dtype == np.float32, f"Reconstruction must be of type np.float32, got {reconstruction.dtype}"
    assert target.dtype == np.float32, f"Target must be of type np.float32, got {target.dtype}"

    # Create masks containing the nan values for both target and reconstruction
    nan_mask_target = np.isnan(target)
    nan_mask_reconstruction = np.isnan(reconstruction)

    target -= np.nanmean(target, axis=1, keepdims=True)
    reconstruction -= np.nanmean(reconstruction, axis=1, keepdims=True)

    # Compute the signal power
    signal_power = np.sum(np.nan_to_num(target)**2, axis=1)

    # Compute the noise power
    noise = np.nan_to_num(target - reconstruction)

    noise_power = np.sum(noise**2, axis=1)

    # Compute the SNR per channel
    snr_per_channel = snr(signal_power, noise_power)

    # Compute the fraction of reconstructed samples
    nan_mask_reconstruction = nan_mask_reconstruction.flatten()
    nan_mask_target = nan_mask_target.flatten()

    nan_mask_reconstruction = nan_mask_reconstruction[nan_mask_target!=True]

    alpha = 1-nan_mask_reconstruction.astype(np.float32).mean()

    return snr_per_channel, alpha

def calculate_mean_snr_batch(reconstruction_batch, target_batch):
    """
    Compute the mean signal-to-noise ratio (SNR) for each sample in the batch.

    Input:
        reconstruction_batch: np.ndarray:np.float32, shape (batch_size, 12, num_samples)
        target_batch: np.ndarray:np.float32, shape (batch_size, 12, num_samples)

    Returns:
        snr: np.ndarray:np.float32, shape (batch_size,)
    """

    assert reconstruction_batch.shape == target_batch.shape, f"Reconstruction and target must have the same shape, got {reconstruction_batch.shape} and {target_batch.shape}"
    assert reconstruction_batch.dtype == np.float32, f"Reconstruction must be of type np.float32, got {reconstruction_batch.dtype}"
    assert target_batch.dtype == np.float32, f"Target must be of type np.float32, got {target_batch.dtype}"

    B, C, N = reconstruction_batch.shape

    assert C == 12, f"Reconstruction and target must have 12 channels, got {C}"

    alpha_values = np.zeros(B, dtype=np.float32)
    snr_values = np.zeros((B,12), dtype=np.float32)

    for i in range(B):
        snr_values[i], alpha_values[i] = calculate_snr_per_sample(reconstruction_batch[i], target_batch[i])

    mean_snr = np.mean(snr_values)
    mean_alpha = np.mean(alpha_values)

    total_score = mean_snr * mean_alpha

    return total_score, snr_values, alpha_values