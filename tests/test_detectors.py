"""Comprehensive test suite for DeepGuard detectors, pipeline, and API endpoints."""

import io
import pytest
from PIL import Image
import numpy as np
from fastapi.testclient import TestClient

from deepguard.detectors.spatial import SpatialDetector
from deepguard.detectors.frequency import FrequencyDetector
from deepguard.detectors.metadata import MetadataDetector
from deepguard.detectors.biological import BiologicalDetector
from deepguard.pipeline import DeepGuardPipeline
from deepguard.synthesis.benchmark_generator import BenchmarkGenerator
from deepguard.api.server import app

client = TestClient(app)

@pytest.fixture
def sample_face_image():
    """Generates a clean synthetic benchmark face image."""
    return BenchmarkGenerator.create_synthetic_test_face(size=(256, 256))

def test_benchmark_generator(sample_face_image):
    assert isinstance(sample_face_image, Image.Image)
    assert sample_face_image.size == (256, 256)
    
    comp_img = BenchmarkGenerator.apply_compression_disparity(sample_face_image)
    assert comp_img.size == (256, 256)
    
    grid_img = BenchmarkGenerator.apply_frequency_grid_artifacts(sample_face_image)
    assert grid_img.size == (256, 256)
    
    blend_img = BenchmarkGenerator.apply_blended_patch(sample_face_image)
    assert blend_img.size == (256, 256)

def test_spatial_detector(sample_face_image):
    detector = SpatialDetector()
    res = detector.analyze(sample_face_image)
    
    assert "spatial_manipulation_score" in res
    assert 0.0 <= res["spatial_manipulation_score"] <= 1.0
    assert "ela_image" in res
    assert isinstance(res["ela_image"], Image.Image)
    assert "ela_mean_error" in res

def test_frequency_detector(sample_face_image):
    detector = FrequencyDetector()
    res = detector.analyze(sample_face_image)
    
    assert "frequency_manipulation_score" in res
    assert 0.0 <= res["frequency_manipulation_score"] <= 1.0
    assert "fft_spectrum_image" in res
    assert "radial_profile" in res
    assert len(res["radial_profile"]) > 0

def test_biological_detector(sample_face_image):
    detector = BiologicalDetector()
    res = detector.analyze(sample_face_image)
    
    assert "biological_manipulation_score" in res
    assert 0.0 <= res["biological_manipulation_score"] <= 1.0
    assert "rgb_channel_coherence" in res

def test_metadata_detector(sample_face_image):
    detector = MetadataDetector()
    res = detector.analyze(sample_face_image)
    
    assert "metadata_manipulation_score" in res
    assert "exif_tag_count" in res
    assert "c2pa_provenance" in res

def test_unified_pipeline(sample_face_image):
    pipeline = DeepGuardPipeline()
    report = pipeline.analyze(sample_face_image)
    
    assert "verdict" in report
    assert report["verdict"] in ["AUTHENTIC", "SUSPICIOUS", "HIGHLY_LIKELY_MANIPULATED"]
    assert "risk_level" in report
    assert "manipulation_probability" in report
    assert 0.0 <= report["manipulation_probability"] <= 1.0
    assert "authenticity_confidence_pct" in report
    assert "stream_scores" in report
    assert "findings" in report
    assert len(report["findings"]) > 0

def test_api_health():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "active_streams" in data

def test_api_benchmark_simulate():
    response = client.post("/api/v1/benchmark/simulate?manipulation_type=frequency_grid")
    assert response.status_code == 200
    data = response.json()
    assert "verdict" in data
    assert data["simulation_type"] == "frequency_grid"

def test_api_analyze_upload(sample_face_image):
    buf = io.BytesIO()
    sample_face_image.save(buf, format="PNG")
    buf.seek(0)
    
    files = {"file": ("test.png", buf, "image/png")}
    response = client.post("/api/v1/analyze", files=files)
    
    assert response.status_code == 200
    data = response.json()
    assert "verdict" in data
    assert "authenticity_confidence_pct" in data
