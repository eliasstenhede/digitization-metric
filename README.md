# Signal-to-Noise Ratio (SNR) Metric with Reconstruction Fraction

This repository implements a metric that combines the **Signal-to-Noise Ratio (SNR)** with the fraction of reconstructed points. It is used to assess the quality of signal reconstruction across multiple channels, factoring in both the signal quality and how much of the signal has been successfully reconstructed.

## Overview

The metric is based on the following steps:

1. **Signal-to-Noise Ratio (SNR)**:  
   SNR is calculated for each signal channel, comparing the power of the original signal (target) to the power of the noise (difference between the target and the reconstructed signal). The result is expressed in decibels (dB).

2. **Fraction of Reconstructed Points**:  
   This component measures the fraction of valid (non-NaN) points in the reconstructed signal relative to the target signal.

3. **Final Metric**:  
   The SNR is scaled by the fraction of reconstructed points (alpha) to produce the final score. This ensures that both the quality of reconstruction and the completeness of the reconstruction are considered.

## Formula

Given `signal_power` and `noise_power`, the Signal-to-Noise Ratio (SNR) in decibels is calculated as:

`SNR = 10 * log10(signal_power / noise_power)`

The final score is computed by scaling the mean SNR by the fraction of reconstructed points (`alpha`):

`Final Score = Mean SNR * alpha`

Where:

`alpha` represents the fraction of reconstructed (non-NaN) points in the signal.
