from __future__ import annotations

import numpy as np

from amfd.core.models import FeatureVector, SensorWindow


def extract_features(window: SensorWindow) -> FeatureVector:
    signal = np.asarray(window.vibration, dtype=float)
    if signal.size < 4:
        raise ValueError("At least four vibration samples are required.")

    centered = signal - signal.mean()
    rms = float(np.sqrt(np.mean(np.square(centered))))
    peak_to_peak = float(np.ptp(centered))
    peak = float(np.max(np.abs(centered)))
    crest_factor = float(peak / rms) if rms > 0 else 0.0

    spectrum = np.fft.rfft(centered)
    magnitudes = np.abs(spectrum)
    frequencies = np.fft.rfftfreq(signal.size, d=1 / window.sampling_rate_hz)
    dominant_index = int(np.argmax(magnitudes[1:]) + 1) if magnitudes.size > 1 else 0
    spectral_energy = float(np.sum(np.square(magnitudes)) / signal.size)

    rpm_mean = float(np.mean(window.rpm)) if window.rpm else None

    return FeatureVector(
        rms=rms,
        peak_to_peak=peak_to_peak,
        crest_factor=crest_factor,
        dominant_frequency_hz=float(frequencies[dominant_index]),
        spectral_energy=spectral_energy,
        rpm_mean=rpm_mean,
    )

