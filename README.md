# DeepGuard: Open-Source Deepfake Detection & Media Verification Suite

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-teal.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)

DeepGuard is a modular, multi-stream media verification framework designed to detect AI-generated faces, face swaps, GAN/Diffusion artifacts, and digital manipulation using spatial forensics, 2D Fast Fourier Transform (FFT) frequency spectrum analysis, biological/chrominance consistency, and cryptographic C2PA / EXIF metadata inspection.

---

## 🏛️ Architecture

```
                                  ┌──────────────────────────┐
                                  │   Input Media (Img/Vid)  │
                                  └─────────────┬────────────┘
                                                │
                 ┌──────────────────────────────┼──────────────────────────────┐
                 ▼                              ▼                              ▼
    ┌─────────────────────────┐   ┌─────────────────────────┐   ┌─────────────────────────┐
    │     Spatial Stream      │   │    Frequency Stream     │   │   Metadata & Provenance │
    ├─────────────────────────┤   ├─────────────────────────┤   ├─────────────────────────┤
    │ • Error Level (ELA)     │   │ • 2D FFT Power Spectrum │   │ • EXIF / Sensor Tags    │
    │ • Blending Boundaries   │   │ • High-Freq Peak Anomaly│   │ • C2PA / JUMBF Signatures│
    │ • Gradient Inconsistency│   │ • Azimuthal Profile Avg │   │ • Compression Matrices  │
    └────────────┬────────────┘   └─────────────┬───────────┘   └─────────────┬───────────┘
                 │                              │                             │
                 └──────────────────────────────┼─────────────────────────────┘
                                                │
                                  ┌─────────────▼────────────┐
                                  │ Unified Decision Engine  │
                                  │ (Weighted Multi-Score)   │
                                  └─────────────┬────────────┘
                                                │
                        ┌───────────────────────┴───────────────────────┐
                        ▼                                               ▼
          ┌───────────────────────────┐                   ┌───────────────────────────┐
          │   Streamlit Web App       │                   │   FastAPI REST Service    │
          │   Interactive Dashboard   │                   │   Automated API Pipeline  │
          └───────────────────────────┘                   └───────────────────────────┘
```

---

## ⚡ Key Features

1. **Spatial Artifact & Compression Forensics (ELA):**
   * Multi-pass Error Level Analysis (ELA) detects disparate compression rates between host images and spliced/synthesized regions.
   * High-pass residual noise analysis catches unnatural local smoothing and boundary blending seams.

2. **Frequency Domain (2D FFT) Analysis:**
   * 2D Fourier Transform extracts the radial power spectrum to uncover high-frequency periodic grid artifacts from GAN deconvolution and diffusion upsampling.
   * Calculates Azimuthal Average profiles against natural $1/f^\alpha$ power-law decay.

3. **Biological & Chrominance Coherence:**
   * Analyzes YCbCr / LAB color space channel correlations and skin chrominance distributions to detect lighting and physiological discordance.

4. **Provenance & Metadata Audit:**
   * Scans EXIF tags for camera sensor information, timestamps, and known AI generator signatures.
   * Inspects binary headers for C2PA (Coalition for Content Provenance and Authenticity) JUMBF manifests.

5. **Defensive Synthesis Benchmark Generator:**
   * Built-in sandbox for creating controlled manipulation benchmarks (Poisson blending, boundary blurring, high-frequency grid injection) for stress-testing detectors.

6. **Full-Featured User Interfaces:**
   * **Streamlit Web Dashboard:** Interactive UI with side-by-side visual inspections, heatmaps, and scorecards.
   * **FastAPI Backend:** Production-ready REST endpoints for automated verification pipelines.
   * **CLI Utility:** Command-line tool for single-file and directory batch audits.

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository and navigate into the workspace
git clone <repo-url>
cd dhruva

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch Interactive Streamlit Dashboard

```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 3. Launch FastAPI REST Server

```bash
uvicorn deepguard.api.server:app --host 0.0.0.0 --port 8000 --reload
```
API Documentation (Swagger UI) is available at `http://localhost:8000/docs`.

### 4. Run CLI Audits

```bash
# Audit a single image and export JSON report with visual heatmaps
python cli.py -i sample_test.png -o report.json --save-visuals

# Batch audit an entire directory
python cli.py -d ./sample_images/ -o batch_report.json
```

### 5. Run Test Suite

```bash
python -m pytest tests/
```

---

## 📡 REST API Reference

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/api/v1/health` | `GET` | Service status and active stream detectors |
| `/api/v1/analyze` | `POST` | Upload image file; returns forensic report JSON |
| `/api/v1/analyze/visuals` | `POST` | Upload image file; returns report + base64 ELA & FFT images |
| `/api/v1/benchmark/simulate` | `POST` | Generates a synthetic manipulation benchmark and tests it |

---

## 📊 Example JSON Output

```json
{
  "verdict": "SUSPICIOUS",
  "risk_level": "MEDIUM",
  "manipulation_probability": 0.3825,
  "authenticity_confidence_pct": 61.75,
  "stream_scores": {
    "spatial": 0.0142,
    "frequency": 0.8000,
    "biological": 0.4500,
    "metadata": 0.2000
  },
  "findings": [
    "Frequency: Anomalous spectral power distribution (0.0020) typical of GAN/diffusion upsampling artifacts.",
    "EXIF metadata is completely stripped (common in web/synthetic media)."
  ]
}
```

---

## 🛡️ License

This project is licensed under the MIT License.
