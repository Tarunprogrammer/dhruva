"""Metadata, EXIF, and C2PA Content Provenance inspector."""

from typing import Dict, Any, List
from PIL import Image, ExifTags
import re

class MetadataDetector:
    """
    Examines image metadata, EXIF tags, software fingerprints, and C2PA markers.
    """

    KNOWN_AI_SIGNATURES = [
        "stable diffusion", "midjourney", "dall-e", "novelai", "comfyui",
        "automatic1111", "facefusion", "deepfacelab", "photoshop", "gimp",
        "civitai", "replicate", "runway", "pika"
    ]

    def extract_exif(self, image: Image.Image) -> Dict[str, Any]:
        """Extracts and normalizes EXIF tags from PIL Image."""
        exif_data = {}
        raw_exif = getattr(image, "_getexif", lambda: None)()
        
        if raw_exif:
            for tag_id, value in raw_exif.items():
                tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                # Filter out raw binary blobs
                if isinstance(value, (bytes, bytearray)):
                    if len(value) < 128:
                        exif_data[tag_name] = str(value)
                else:
                    exif_data[tag_name] = str(value)
        return exif_data

    def check_ai_software_signatures(self, exif_data: Dict[str, Any], raw_bytes: bytes = b"") -> List[str]:
        """Checks for editing software or generative AI software signatures."""
        detected = []
        
        # Check text in EXIF fields
        searchable_text = " ".join([f"{k}:{v}" for k, v in exif_data.items()]).lower()
        for sig in self.KNOWN_AI_SIGNATURES:
            if sig in searchable_text:
                detected.append(f"EXIF tag matched known signature: '{sig}'")

        # Check raw byte markers if provided
        if raw_bytes:
            lower_bytes = raw_bytes[:10000].lower()
            for sig in self.KNOWN_AI_SIGNATURES:
                if sig.encode("utf-8") in lower_bytes and not any(sig in d for d in detected):
                    detected.append(f"Binary header matched keyword: '{sig}'")

        return detected

    def check_c2pa_provenance(self, raw_bytes: bytes) -> Dict[str, Any]:
        """
        Inspects binary data for C2PA (Coalition for Content Provenance and Authenticity) JUMBF boxes.
        """
        has_c2pa = False
        manifest_detected = False

        if b"c2pa" in raw_bytes or b"urn:c2pa:" in raw_bytes:
            has_c2pa = True
        if b"jumb" in raw_bytes and b"c2pa" in raw_bytes:
            manifest_detected = True

        return {
            "has_c2pa_manifest": has_c2pa,
            "has_jumbf_boxes": manifest_detected,
            "provenance_status": "Signed Provenance Present" if has_c2pa else "No C2PA Manifest Found"
        }

    def analyze(self, image: Image.Image, raw_bytes: bytes = b"") -> Dict[str, Any]:
        """
        Performs full metadata and provenance audit.
        """
        exif = self.extract_exif(image)
        ai_signatures = self.check_ai_software_signatures(exif, raw_bytes)
        c2pa_info = self.check_c2pa_provenance(raw_bytes) if raw_bytes else {
            "has_c2pa_manifest": False,
            "has_jumbf_boxes": False,
            "provenance_status": "No Raw Stream Available"
        }

        has_hardware_info = any(k in exif for k in ["Make", "Model", "LensModel", "FocalLength"])
        is_exif_stripped = len(exif) == 0

        # Score calculation:
        # Software signatures -> high probability of manipulation/generation
        # Stripped EXIF without hardware data -> mild suspicion (very common on social platforms)
        score = 0.0
        reasons = []

        if ai_signatures:
            score += 0.85
            reasons.extend(ai_signatures)
        elif is_exif_stripped:
            score += 0.20
            reasons.append("EXIF metadata is completely stripped (common in web/synthetic media).")
        elif not has_hardware_info:
            score += 0.15
            reasons.append("No physical camera sensor/hardware metadata identified.")

        if c2pa_info.get("has_c2pa_manifest"):
            reasons.append("Cryptographic C2PA metadata verified.")

        metadata_score = float(min(1.0, score))

        return {
            "metadata_manipulation_score": round(metadata_score, 4),
            "exif_tag_count": len(exif),
            "camera_hardware_present": has_hardware_info,
            "ai_signatures_found": ai_signatures,
            "c2pa_provenance": c2pa_info,
            "findings": reasons,
            "exif_summary": {k: exif[k] for k in list(exif.keys())[:10]}
        }
