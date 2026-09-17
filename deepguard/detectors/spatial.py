"""Spatial artifact and compression inconsistency detector."""

import io
from typing import Dict, Any, Tuple
import numpy as np
from PIL import Image, ImageChops, ImageEnhance, ImageFilter

class SpatialDetector:
    """
    Analyzes spatial domain anomalies:
    1. Error Level Analysis (ELA) for compression rate disparities
    2. High-frequency noise residuals (filtering smoothing artifacts)
    3. Gradient & Laplacian edge variance across image segments
    """

    def __init__(self, ela_quality: int = 90, ela_scale: float = 15.0):
        self.ela_quality = ela_quality
        self.ela_scale = ela_scale

    def compute_ela(self, image: Image.Image) -> Tuple[Image.Image, float, float]:
        """
        Computes Error Level Analysis (ELA).
        Returns:
            - ELA visual heatmap (PIL.Image)
            - Mean error level
            - Max error level
        """
        # Ensure image is in RGB format
        rgb_image = image.convert("RGB")
        
        # Save to buffer at known quality
        buffer = io.BytesIO()
        rgb_image.save(buffer, format="JPEG", quality=self.ela_quality)
        buffer.seek(0)
        
        # Reload compressed version
        recompressed = Image.open(buffer).convert("RGB")
        
        # Compute absolute difference
        diff = ImageChops.difference(rgb_image, recompressed)
        diff_arr = np.asarray(diff, dtype=np.float32)
        
        # Calculate statistics
        mean_err = float(np.mean(diff_arr))
        max_err = float(np.max(diff_arr))
        
        # Amplify difference for visual inspection
        extrema = diff.getextrema()
        max_diff = max([ex[1] for ex in extrema]) if extrema else 1.0
        scale = 255.0 / max(1.0, max_diff) if max_diff > 0 else 1.0
        
        enhanced_diff = ImageEnhance.Brightness(diff).enhance(scale)
        return enhanced_diff, mean_err, max_err

    def compute_noise_residual(self, image: Image.Image) -> Dict[str, float]:
        """
        Calculates high-pass residual noise characteristics.
        Synthetic media often shows unnaturally smooth regions (low residual variance)
        or inconsistent noise patches.
        """
        gray = image.convert("L")
        img_arr = np.asarray(gray, dtype=np.float32)
        
        # Gaussian blurred baseline
        blurred = gray.filter(ImageFilter.GaussianBlur(radius=2))
        blurred_arr = np.asarray(blurred, dtype=np.float32)
        
        # Residual = Original - Blurred (High-frequency component)
        residual = img_arr - blurred_arr
        
        res_std = float(np.std(residual))
        res_var = float(np.var(residual))
        
        # Split into quadrants and test variance consistency
        h, w = img_arr.shape
        quads = [
            residual[:h//2, :w//2],
            residual[:h//2, w//2:],
            residual[h//2:, :w//2],
            residual[h//2:, w//2:]
        ]
        quad_vars = [float(np.var(q)) for q in quads if q.size > 0]
        quad_disparity = float(np.std(quad_vars) / (np.mean(quad_vars) + 1e-6))
        
        return {
            "noise_std": res_std,
            "noise_var": res_var,
            "quadrant_disparity": quad_disparity
        }

    def compute_edge_inconsistencies(self, image: Image.Image) -> float:
        """
        Calculates Laplacian edge gradient variance.
        Blending seams typically introduce abnormal variance at high-contrast boundaries.
        """
        gray = image.convert("L")
        arr = np.asarray(gray, dtype=np.float32)
        
        # Approximate Laplacian kernel using numpy convolution
        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
        
        # Simple valid convolution via slicing
        lap = (
            arr[:-2, 1:-1] + arr[2:, 1:-1] +
            arr[1:-1, :-2] + arr[1:-1, 2:] -
            4 * arr[1:-1, 1:-1]
        )
        lap_var = float(np.var(lap))
        return lap_var

    def analyze(self, image: Image.Image) -> Dict[str, Any]:
        """
        Executes full spatial inspection and returns metrics and confidence score.
        """
        ela_img, mean_err, max_err = self.compute_ela(image)
        noise_metrics = self.compute_noise_residual(image)
        lap_var = self.compute_edge_inconsistencies(image)
        
        # Heuristic scoring based on ELA disparity and noise quadrant disparity
        # In deepfakes, local ELA discrepancies and quadrant noise disparities are elevated
        ela_score = min(1.0, max(0.0, (mean_err - 2.0) / 12.0))
        noise_score = min(1.0, max(0.0, (noise_metrics["quadrant_disparity"] - 0.2) / 0.8))
        
        # Combined spatial score
        spatial_score = float(np.clip(0.6 * ela_score + 0.4 * noise_score, 0.0, 1.0))
        
        return {
            "spatial_manipulation_score": round(spatial_score, 4),
            "ela_mean_error": round(mean_err, 4),
            "ela_max_error": round(max_err, 4),
            "noise_quadrant_disparity": round(noise_metrics["quadrant_disparity"], 4),
            "laplacian_edge_variance": round(lap_var, 4),
            "ela_image": ela_img
        }
