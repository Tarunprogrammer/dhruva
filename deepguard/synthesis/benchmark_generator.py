"""Defensive benchmarking module to generate controlled synthetic manipulations."""

import io
from typing import Tuple
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance

class BenchmarkGenerator:
    """
    Generates controlled manipulation benchmarks for stress-testing deepfake detectors:
    1. Regional Compression Disparity (simulates splicing/unmatched re-encoding).
    2. High-Frequency Checkerboard Noise (simulates GAN deconvolution artifacts).
    3. Blended/Feathered Patch (simulates face swapping boundary).
    4. Over-Smoothing Filter (simulates diffusion/AI denoiser skin smoothing).
    """

    @staticmethod
    def create_synthetic_test_face(size: Tuple[int, int] = (256, 256)) -> Image.Image:
        """Generates a synthetic geometric face canvas for offline benchmarking."""
        arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        # Background gradient
        for y in range(size[1]):
            arr[y, :, 0] = int(120 + 40 * (y / size[1]))
            arr[y, :, 1] = int(140 + 30 * (y / size[1]))
            arr[y, :, 2] = int(180 - 40 * (y / size[1]))
        
        # Draw central oval face shape
        cy, cx = size[1] // 2, size[0] // 2
        ry, rx = int(size[1] * 0.35), int(size[0] * 0.28)
        y_grid, x_grid = np.ogrid[:size[1], :size[0]]
        mask = (((x_grid - cx) ** 2) / (rx ** 2) + ((y_grid - cy) ** 2) / (ry ** 2)) <= 1.0
        
        # Skin tone
        arr[mask, 0] = 230
        arr[mask, 1] = 190
        arr[mask, 2] = 160
        
        img = Image.fromarray(arr, mode="RGB")
        return img

    @staticmethod
    def apply_compression_disparity(image: Image.Image, patch_quality: int = 40) -> Image.Image:
        """
        Compresses a central region at low JPEG quality and composites it back,
        creating an ELA disparity boundary.
        """
        img_rgb = image.convert("RGB")
        w, h = img_rgb.size
        
        # Central crop
        box = (w // 4, h // 4, 3 * w // 4, 3 * h // 4)
        patch = img_rgb.crop(box)
        
        # Compress patch
        buf = io.BytesIO()
        patch.save(buf, format="JPEG", quality=patch_quality)
        buf.seek(0)
        compressed_patch = Image.open(buf)
        
        # Composite back
        result = img_rgb.copy()
        result.paste(compressed_patch, box)
        return result

    @staticmethod
    def apply_frequency_grid_artifacts(image: Image.Image, intensity: float = 0.08) -> Image.Image:
        """
        Adds high-frequency 2D periodic grid artifacts simulating transposed convolution (GAN generator).
        """
        arr = np.asarray(image.convert("RGB"), dtype=np.float32)
        h, w, c = arr.shape
        
        # Checkerboard grid pattern
        y, x = np.mgrid[:h, :w]
        grid = (np.sin(x * np.pi / 2.0) * np.sin(y * np.pi / 2.0)) * 255.0 * intensity
        
        for ch in range(c):
            arr[:, :, ch] = np.clip(arr[:, :, ch] + grid, 0, 255)
            
        return Image.fromarray(arr.astype(np.uint8), mode="RGB")

    @staticmethod
    def apply_blended_patch(image: Image.Image, blur_radius: int = 4) -> Image.Image:
        """
        Simulates face-swap boundary blending and smoothing.
        """
        img_rgb = image.convert("RGB")
        w, h = img_rgb.size
        
        # Create blurred modified patch
        box = (w // 4, h // 4, 3 * w // 4, 3 * h // 4)
        patch = img_rgb.crop(box)
        patch = patch.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        
        # Enhanced brightness to simulate lighting mismatch
        patch = ImageEnhance.Brightness(patch).enhance(1.15)
        
        result = img_rgb.copy()
        result.paste(patch, box)
        return result
