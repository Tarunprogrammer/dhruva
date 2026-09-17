"""Frequency domain (FFT / Spectral) anomaly detector."""

import io
from typing import Dict, Any, Tuple
import numpy as np
from PIL import Image

class FrequencyDetector:
    """
    Analyzes frequency domain anomalies using:
    1. 2D Fast Fourier Transform (FFT) Power Spectrum.
    2. Radial / Azimuthal Average Power Distribution (1D spectral profile).
    3. High-Frequency / Low-Frequency energy balance and spectral peak detection.
    """

    def __init__(self, target_size: Tuple[int, int] = (256, 256)):
        self.target_size = target_size

    def compute_fft_spectrum(self, image: Image.Image) -> Tuple[np.ndarray, np.ndarray, Image.Image]:
        """
        Computes 2D FFT and magnitude spectrum.
        Returns:
            - Magnitude spectrum (2D numpy float array)
            - Centered complex FFT array
            - Renderable visual spectrum (PIL.Image)
        """
        # Resize to fixed dimension for standardized frequency evaluation
        resized = image.convert("L").resize(self.target_size, Image.Resampling.BILINEAR)
        img_arr = np.asarray(resized, dtype=np.float32)

        # 2D Fast Fourier Transform
        f = np.fft.fft2(img_arr)
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift)

        # Log power spectrum for dynamic range compression
        log_spectrum = np.log1p(magnitude)

        # Normalize to 0-255 for visualization
        norm_spec = (log_spectrum - log_spectrum.min()) / (log_spectrum.max() - log_spectrum.min() + 1e-8)
        vis_arr = (norm_spec * 255.0).astype(np.uint8)
        vis_image = Image.fromarray(vis_arr, mode="L")

        return magnitude, fshift, vis_image

    def compute_azimuthal_average(self, magnitude: np.ndarray) -> np.ndarray:
        """
        Calculates the 1D radially-averaged (azimuthal) power profile from center to edges.
        """
        h, w = magnitude.shape
        cy, cx = h // 2, w // 2

        y, x = np.ogrid[:h, :w]
        r = np.hypot(x - cx, y - cy).astype(int)

        # Compute radial average
        max_r = min(cx, cy)
        radial_profile = np.zeros(max_r, dtype=np.float32)
        
        for radius in range(max_r):
            mask = (r == radius)
            if np.any(mask):
                radial_profile[radius] = np.mean(magnitude[mask])

        return radial_profile

    def detect_spectral_anomalies(self, magnitude: np.ndarray, radial_profile: np.ndarray) -> Dict[str, float]:
        """
        Evaluates spectral decay properties and high-frequency power ratios.
        Natural camera sensors exhibit smooth 1/f power law decay.
        Deepfakes & GANs frequently contain high-frequency grid artifacts or abrupt decay slopes.
        """
        total_energy = float(np.sum(magnitude)) + 1e-8
        max_r = len(radial_profile)
        
        if max_r < 10:
            return {"hf_ratio": 0.0, "decay_roughness": 0.0, "spectral_anomaly_score": 0.0}

        # Energy distribution across low, mid, and high frequency bands
        low_band = np.sum(radial_profile[: max_r // 3])
        high_band = np.sum(radial_profile[2 * max_r // 3 :])
        
        hf_ratio = float(high_band / (low_band + 1e-6))

        # Radial profile roughness / oscillation (GAN artifacts produce ripples in radial spectrum)
        diffs = np.diff(radial_profile)
        roughness = float(np.std(diffs) / (np.mean(radial_profile) + 1e-6))

        # Spectral anomaly scoring heuristic
        # Deepfake artifacts usually present as either abnormally low high-freq (due to smoothing)
        # or abnormally high localized spikes (due to checkerboard deconvolution).
        anomaly_score = 0.0
        if hf_ratio < 0.005:  # Oversmoothed / synthetic face blending
            anomaly_score += 0.45
        elif hf_ratio > 0.08:  # High-frequency checkerboard / unmasked noise
            anomaly_score += 0.40

        if roughness > 0.4:
            anomaly_score += 0.35

        anomaly_score = float(np.clip(anomaly_score, 0.0, 1.0))

        return {
            "hf_ratio": round(hf_ratio, 6),
            "decay_roughness": round(roughness, 4),
            "spectral_anomaly_score": round(anomaly_score, 4)
        }

    def analyze(self, image: Image.Image) -> Dict[str, Any]:
        """
        Runs comprehensive frequency domain analysis.
        """
        magnitude, _, vis_image = self.compute_fft_spectrum(image)
        radial_profile = self.compute_azimuthal_average(magnitude)
        metrics = self.detect_spectral_anomalies(magnitude, radial_profile)

        return {
            "frequency_manipulation_score": metrics["spectral_anomaly_score"],
            "high_frequency_ratio": metrics["hf_ratio"],
            "spectral_roughness": metrics["decay_roughness"],
            "fft_spectrum_image": vis_image,
            "radial_profile": radial_profile.tolist()
        }
