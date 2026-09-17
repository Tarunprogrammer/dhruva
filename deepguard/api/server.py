"""FastAPI REST API endpoints for DeepGuard Deepfake Detection."""

import io
import base64
from typing import Dict, Any
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

from ..pipeline import DeepGuardPipeline
from ..synthesis.benchmark_generator import BenchmarkGenerator

app = FastAPI(
    title="DeepGuard Media Verification API",
    description="Multi-stream REST API for deepfake detection, frequency spectrum forensics, and metadata provenance audit.",
    version="1.0.0"
)

# Enable CORS for frontend applications
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = DeepGuardPipeline()

def pil_to_base64(img: Image.Image, format: str = "PNG") -> str:
    """Encodes a PIL image as a base64 string."""
    buf = io.BytesIO()
    img.save(buf, format=format)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

@app.get("/api/v1/health")
def health_check() -> Dict[str, Any]:
    """Health and status check endpoint."""
    return {
        "status": "healthy",
        "service": "DeepGuard Forensic Verification Engine",
        "version": "1.0.0",
        "active_streams": ["spatial_ela", "frequency_fft", "biological_chrominance", "metadata_c2pa"]
    }

@app.post("/api/v1/analyze")
async def analyze_media(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Accepts an uploaded image and performs full multi-stream forensic audit.
    Returns verdict, authenticity confidence score, and detailed metric breakdown.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a valid image format.")

    try:
        content = await file.read()
        report = pipeline.analyze(content)
        
        # Remove raw PIL objects from standard response for clean JSON serialization
        clean_report = {k: v for k, v in report.items() if k != "visual_assets"}
        return clean_report
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forensic analysis failed: {str(e)}")

@app.post("/api/v1/analyze/visuals")
async def analyze_media_with_visuals(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Performs forensic audit and returns base64-encoded forensic visualization heatmaps (ELA and 2D FFT).
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a valid image format.")

    try:
        content = await file.read()
        report = pipeline.analyze(content)
        
        ela_b64 = pil_to_base64(report["visual_assets"]["ela_image"])
        fft_b64 = pil_to_base64(report["visual_assets"]["fft_spectrum_image"])

        clean_report = {k: v for k, v in report.items() if k != "visual_assets"}
        clean_report["visual_assets_base64"] = {
            "ela_heatmap_png": f"data:image/png;base64,{ela_b64}",
            "fft_spectrum_png": f"data:image/png;base64,{fft_b64}"
        }
        return clean_report
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forensic analysis failed: {str(e)}")

@app.post("/api/v1/benchmark/simulate")
def simulate_benchmark(manipulation_type: str = "compression_disparity") -> Dict[str, Any]:
    """
    Generates a synthetic manipulation benchmark and executes verification against it.
    Supported types: 'compression_disparity', 'frequency_grid', 'blended_patch'.
    """
    base_face = BenchmarkGenerator.create_synthetic_test_face()
    
    if manipulation_type == "compression_disparity":
        manipulated = BenchmarkGenerator.apply_compression_disparity(base_face)
    elif manipulation_type == "frequency_grid":
        manipulated = BenchmarkGenerator.apply_frequency_grid_artifacts(base_face)
    elif manipulation_type == "blended_patch":
        manipulated = BenchmarkGenerator.apply_blended_patch(base_face)
    else:
        raise HTTPException(status_code=400, detail=f"Unknown manipulation type: {manipulation_type}")

    report = pipeline.analyze(manipulated)
    clean_report = {k: v for k, v in report.items() if k != "visual_assets"}
    clean_report["simulation_type"] = manipulation_type
    return clean_report
