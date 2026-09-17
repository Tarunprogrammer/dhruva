"""DeepGuard: Interactive Web Forensic Dashboard."""

import io
import json
import streamlit as st
from PIL import Image

from deepguard.pipeline import DeepGuardPipeline
from deepguard.synthesis.benchmark_generator import BenchmarkGenerator

st.set_page_config(
    page_title="DeepGuard Forensic Studio",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .verdict-badge {
        font-size: 1.3rem;
        font-weight: bold;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        display: inline-block;
        margin-top: 0.5rem;
    }
    .badge-authentic { background-color: #DCFCE7; color: #166534; }
    .badge-suspicious { background-color: #FEF9C3; color: #854D0E; }
    .badge-fake { background-color: #FEE2E2; color: #991B1B; }
</style>
""", unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.title("🛡️ DeepGuard Engine")
st.sidebar.markdown("**Multi-Stream Forensic Verification**")

st.sidebar.subheader("Stream Weight Tuning")
w_spatial = st.sidebar.slider("Spatial Stream (ELA / Noise)", 0.0, 1.0, 0.35, 0.05)
w_freq = st.sidebar.slider("Frequency Stream (2D FFT)", 0.0, 1.0, 0.35, 0.05)
w_bio = st.sidebar.slider("Biological Stream (Color/Coherence)", 0.0, 1.0, 0.15, 0.05)
w_meta = st.sidebar.slider("Metadata Stream (EXIF / C2PA)", 0.0, 1.0, 0.15, 0.05)

# Initialize Pipeline with selected weights
pipeline = DeepGuardPipeline(
    spatial_weight=w_spatial,
    frequency_weight=w_freq,
    biological_weight=w_bio,
    metadata_weight=w_meta
)

# Top Bar
st.markdown("<div class='main-title'>DeepGuard Media Verification Suite</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Open-Source Deepfake Detection, Spectral Forensics, and Content Provenance Verification</div>", unsafe_allow_html=True)

# Main Navigation Tabs
tab_inspect, tab_benchmark, tab_methodology = st.tabs([
    "🔍 Live Forensic Inspection",
    "🧪 Defensive Benchmark Lab",
    "📚 Methodology & Architecture"
])

def render_forensic_results(report: dict, original_image: Image.Image):
    """Renders comprehensive analysis cards and visual comparisons."""
    verdict = report["verdict"]
    prob = report["manipulation_probability"]
    confidence = report["authenticity_confidence_pct"]
    risk = report["risk_level"]

    st.markdown("---")
    
    # Top Level Scorecard
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Authenticity Confidence", f"{confidence:.1f}%")
    with col2:
        st.metric("Manipulation Score", f"{prob:.4f}")
    with col3:
        st.metric("Risk Level", risk)
    with col4:
        if verdict == "AUTHENTIC":
            st.markdown("<div class='verdict-badge badge-authentic'>✅ AUTHENTIC</div>", unsafe_allow_html=True)
        elif verdict == "SUSPICIOUS":
            st.markdown("<div class='verdict-badge badge-suspicious'>⚠️ SUSPICIOUS</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='verdict-badge badge-fake'>🚨 MANIPULATED</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Visual Artifact Stream
    st.subheader("🖼️ Multi-Stream Visual Inspection")
    vcol1, vcol2, vcol3 = st.columns(3)
    
    with vcol1:
        st.markdown("**Original Image**")
        st.image(original_image, use_column_width=True)
        
    with vcol2:
        st.markdown("**Spatial Error Level (ELA)**")
        st.image(report["visual_assets"]["ela_image"], use_column_width=True, caption="Highlights compression disparity & splicing")

    with vcol3:
        st.markdown("**2D FFT Power Spectrum**")
        st.image(report["visual_assets"]["fft_spectrum_image"], use_column_width=True, caption="Highlights GAN/Diffusion frequency spikes")

    # Stream Breakdown
    st.markdown("---")
    st.subheader("📊 Stream Scores Breakdown")
    
    scol1, scol2, scol3, scol4 = st.columns(4)
    with scol1:
        st.markdown(f"**Spatial Score:** `{report['stream_scores']['spatial']:.3f}`")
        st.progress(report['stream_scores']['spatial'])
    with scol2:
        st.markdown(f"**Frequency Score:** `{report['stream_scores']['frequency']:.3f}`")
        st.progress(report['stream_scores']['frequency'])
    with scol3:
        st.markdown(f"**Biological Score:** `{report['stream_scores']['biological']:.3f}`")
        st.progress(report['stream_scores']['biological'])
    with scol4:
        st.markdown(f"**Metadata Score:** `{report['stream_scores']['metadata']:.3f}`")
        st.progress(report['stream_scores']['metadata'])

    # Forensic Findings List
    st.markdown("---")
    st.subheader("📋 Forensic Audit Log & Findings")
    for item in report["findings"]:
        if "High Error" in item or "Anomalous" in item or "EXIF tag matched" in item:
            st.warning(f"⚠️ {item}")
        else:
            st.info(f"ℹ️ {item}")

    # Raw JSON Download
    clean_report = {k: v for k, v in report.items() if k != "visual_assets"}
    report_json = json.dumps(clean_report, indent=2)
    st.download_button(
        label="📥 Download JSON Forensic Report",
        data=report_json,
        file_name="deepguard_forensic_report.json",
        mime="application/json"
    )

# ----------------- TAB 1: LIVE FORENSICS -----------------
with tab_inspect:
    st.subheader("Upload Media for Verification")
    uploaded_file = st.file_uploader("Upload Image (JPG, PNG, WEBP)", type=["jpg", "jpeg", "png", "webp"])
    
    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        image = Image.open(io.BytesIO(file_bytes))
        
        with st.spinner("Executing spatial, spectral, biological, and metadata audits..."):
            report = pipeline.analyze(file_bytes)
        
        render_forensic_results(report, image)
    else:
        st.info("👆 Upload an image to perform deepfake and authenticity verification.")

# ----------------- TAB 2: BENCHMARK LAB -----------------
with tab_benchmark:
    st.subheader("🧪 Synthetic Anomaly Simulator & Detector Stress-Test")
    st.markdown("""
    Use this sandbox to test how different synthetic manipulation techniques 
    (compression boundary mismatches, GAN high-frequency deconvolution grid noise, and blending seams) 
    are caught by the multi-stream engine.
    """)
    
    sim_type = st.selectbox(
        "Select Manipulation Artifact to Simulate:",
        [
            "High-Frequency Checkerboard (GAN Up-sampling)",
            "Regional Compression Disparity (Splicing / Face-Swap)",
            "Boundary Blending & Smoothing (Feathered Face Mask)"
        ]
    )

    if st.button("Generate & Run Benchmark Test"):
        base_face = BenchmarkGenerator.create_synthetic_test_face()
        
        if "Checkerboard" in sim_type:
            manipulated = BenchmarkGenerator.apply_frequency_grid_artifacts(base_face)
        elif "Compression" in sim_type:
            manipulated = BenchmarkGenerator.apply_compression_disparity(base_face)
        else:
            manipulated = BenchmarkGenerator.apply_blended_patch(base_face)
            
        with st.spinner("Analyzing simulated benchmark sample..."):
            report = pipeline.analyze(manipulated)
            
        render_forensic_results(report, manipulated)

# ----------------- TAB 3: METHODOLOGY -----------------
with tab_methodology:
    st.subheader("🔬 Detection Principles & Engineering Foundations")
    st.markdown("""
    ### 1. Spatial Error Level Analysis (ELA)
    When an image is edited (e.g. an AI face is spliced into a photo), the modified area undergoes a different number of lossy compression cycles than the background. ELA intentionally recompresses the image and measures the delta, exposing disparate compression levels across boundaries.

    ### 2. 2D Fast Fourier Transform (FFT) Power Spectrum
    Generative models (GANs, StyleGAN, Latent Diffusion) utilize strided convolutions and transposed deconvolution layers that leave periodic lattice patterns in pixel space. When transformed to the 2D frequency domain via FFT, these manifest as sharp anomalous energy spikes away from natural $1/f^\\alpha$ spectral decay.

    ### 3. Biological & Chrominance Coherence
    Skin pigments (melanin, hemoglobin) produce smooth, physiologically correlated color channel curves ($Cb, Cr$). Decoupled generative blending frequently creates color desynchronization and channel discordance.

    ### 4. Cryptographic Provenance (C2PA)
    Digital authentication standard by C2PA allows cameras and creator tools to bind cryptographically signed provenance manifests to media files.
    """)
