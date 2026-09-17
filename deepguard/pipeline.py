"""Unified multi-stream Deepfake Detection & Media Verification Pipeline."""

import io
from typing import Dict, Any, Union, Optional, Tuple
from PIL import Image
import numpy as np

from .detectors.spatial import SpatialDetector
from .detectors.frequency import FrequencyDetector
from .detectors.metadata import MetadataDetector
from .detectors.biological import BiologicalDetector

class DeepGuardPipeline:
    """
    Unified multi-stream pipeline integrating:
    - Spatial artifact & ELA analysis
    - Frequency 2D FFT & spectral decay analysis
    - Biological & chrominance channel coherence
    - EXIF, C2PA, and cryptographic metadata verification
    """

    def __init__(
        self,
        spatial_weight: float = 0.35,
        frequency_weight: float = 0.35,
        biological_weight: float = 0.15,
        metadata_weight: float = 0.15
    ):
        self.weights = {
            "spatial": spatial_weight,
            "frequency": frequency_weight,
            "biological": biological_weight,
            "metadata": metadata_weight
        }
        # Normalize weights
        total_w = sum(self.weights.values())
        self.weights = {k: v / total_w for k, v in self.weights.items()}

        # Initialize sub-detectors
        self.spatial_detector = SpatialDetector()
        self.frequency_detector = FrequencyDetector()
        self.metadata_detector = MetadataDetector()
        self.biological_detector = BiologicalDetector()

    def classify_verdict(self, score: float) -> Tuple[str, str]:
        """Maps continuous probability score to categorical verdict and risk level."""
        if score < 0.35:
            return "AUTHENTIC", "LOW"
        elif score < 0.65:
            return "SUSPICIOUS", "MEDIUM"
        else:
            return "HIGHLY_LIKELY_MANIPULATED", "HIGH"

    def analyze(self, image_input: Union[str, bytes, Image.Image]) -> Dict[str, Any]:
        """
        Runs complete forensic inspection across all streams.
        Args:
            image_input: File path (str), raw image bytes (bytes), or PIL.Image instance.
        Returns:
            Comprehensive forensic audit dictionary with scores, explanations, and visual assets.
        """
        raw_bytes = b""
        if isinstance(image_input, str):
            with open(image_input, "rb") as f:
                raw_bytes = f.read()
            image = Image.open(io.BytesIO(raw_bytes))
        elif isinstance(image_input, (bytes, bytearray)):
            raw_bytes = bytes(image_input)
            image = Image.open(io.BytesIO(raw_bytes))
        elif isinstance(image_input, Image.Image):
            image = image_input
            # Export to bytes if needed for metadata checks
            buf = io.BytesIO()
            fmt = image.format or "PNG"
            image.save(buf, format=fmt)
            raw_bytes = buf.getvalue()
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        # Execute stream detectors
        spatial_res = self.spatial_detector.analyze(image)
        freq_res = self.frequency_detector.analyze(image)
        bio_res = self.biological_detector.analyze(image)
        meta_res = self.metadata_detector.analyze(image, raw_bytes=raw_bytes)

        # Weighted aggregate calculation
        s_score = spatial_res["spatial_manipulation_score"]
        f_score = freq_res["frequency_manipulation_score"]
        b_score = bio_res["biological_manipulation_score"]
        m_score = meta_res["metadata_manipulation_score"]

        aggregate_score = (
            self.weights["spatial"] * s_score +
            self.weights["frequency"] * f_score +
            self.weights["biological"] * b_score +
            self.weights["metadata"] * m_score
        )
        aggregate_score = float(np.clip(aggregate_score, 0.0, 1.0))
        authenticity_confidence = round((1.0 - aggregate_score) * 100.0, 2)

        verdict, risk_level = self.classify_verdict(aggregate_score)

        # Compile human-readable forensic findings
        findings = []
        if s_score > 0.5:
            findings.append(f"Spatial: High Error Level Analysis (ELA) disparity ({spatial_res['ela_mean_error']:.2f}) indicates potential localized splicing or recompression.")
        if f_score > 0.5:
            findings.append(f"Frequency: Anomalous spectral power distribution ({freq_res['high_frequency_ratio']:.4f}) typical of GAN/diffusion upsampling artifacts.")
        if b_score > 0.5:
            findings.append(f"Biological: Chrominance channel discordance detected (RGB coherence: {bio_res['rgb_channel_coherence']:.2f}).")
        findings.extend(meta_res.get("findings", []))
        if not findings:
            findings.append("No significant spatial, spectral, or metadata manipulation anomalies detected.")

        return {
            "verdict": verdict,
            "risk_level": risk_level,
            "manipulation_probability": round(aggregate_score, 4),
            "authenticity_confidence_pct": authenticity_confidence,
            "stream_scores": {
                "spatial": round(s_score, 4),
                "frequency": round(f_score, 4),
                "biological": round(b_score, 4),
                "metadata": round(m_score, 4)
            },
            "stream_weights": self.weights,
            "detailed_metrics": {
                "spatial": {k: v for k, v in spatial_res.items() if k != "ela_image"},
                "frequency": {k: v for k, v in freq_res.items() if k != "fft_spectrum_image"},
                "biological": bio_res,
                "metadata": meta_res
            },
            "findings": findings,
            "visual_assets": {
                "ela_image": spatial_res["ela_image"],
                "fft_spectrum_image": freq_res["fft_spectrum_image"]
            }
        }
