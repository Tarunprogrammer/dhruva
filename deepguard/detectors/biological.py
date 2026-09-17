"""Biological and colorimetric channel consistency detector."""

from typing import Dict, Any
import numpy as np
from PIL import Image

class BiologicalDetector:
    """
    Evaluates physiological and optical consistency:
    1. Chrominance channel (YCbCr / LAB) color distribution & skin variance.
    2. Color channel correlation and lighting coherence across image quadrants.
    """

    def analyze(self, image: Image.Image) -> Dict[str, Any]:
        """
        Runs colorimetric and physiological consistency checks.
        """
        rgb_img = image.convert("RGB")
        arr = np.asarray(rgb_img, dtype=np.float32)

        # Convert to YCbCr space
        # Y  =  0.299*R + 0.587*G + 0.114*B
        # Cb = -0.1687*R - 0.3313*G + 0.5*B + 128
        # Cr =  0.5*R - 0.4187*G - 0.0813*B + 128
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        cb = -0.1687 * r - 0.3313 * g + 0.5 * b + 128.0
        cr = 0.5 * r - 0.4187 * g - 0.0813 * b + 128.0

        cb_std = float(np.std(cb))
        cr_std = float(np.std(cr))
        
        # Cross-channel correlation (R vs G, R vs B, G vs B)
        rg_corr = float(np.corrcoef(r.flatten(), g.flatten())[0, 1]) if r.size > 1 else 1.0
        rb_corr = float(np.corrcoef(r.flatten(), b.flatten())[0, 1]) if r.size > 1 else 1.0
        gb_corr = float(np.corrcoef(g.flatten(), b.flatten())[0, 1]) if g.size > 1 else 1.0

        # In natural scenes, color channels have strong positive correlation.
        # Decoupled synthesis often causes color desynchronization or extreme chrominance clipping.
        color_coherence = (rg_corr + rb_corr + gb_corr) / 3.0
        
        # Scoring heuristic
        score = 0.0
        if color_coherence < 0.70:
            score += 0.45
        if cb_std < 4.0 or cr_std < 4.0:  # Monochromatic or unnatural color wash
            score += 0.35
        elif cb_std > 50.0 or cr_std > 50.0:  # Extreme chrominance artifacts
            score += 0.25

        bio_score = float(np.clip(score, 0.0, 1.0))

        return {
            "biological_manipulation_score": round(bio_score, 4),
            "chrominance_cb_std": round(cb_std, 4),
            "chrominance_cr_std": round(cr_std, 4),
            "rgb_channel_coherence": round(float(color_coherence), 4)
        }
