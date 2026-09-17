"""Command Line Interface (CLI) for DeepGuard Media Verification."""

import os
import sys
import json
import argparse
from pathlib import Path
from deepguard.pipeline import DeepGuardPipeline

def print_banner():
    print("=" * 65)
    print(" [DeepGuard] Multi-Stream Deepfake & Media Verification Engine")
    print("=" * 65)

def analyze_single_file(pipeline: DeepGuardPipeline, file_path: str, output_json: str = None, save_visuals: bool = False):
    if not os.path.exists(file_path):
        print(f"[-] Error: File not found at '{file_path}'")
        sys.exit(1)

    print(f"\n[+] Ingesting: {file_path}")
    print("[*] Running Spatial, Spectral, Biological, and Metadata audits...")
    
    report = pipeline.analyze(file_path)
    verdict = report["verdict"]
    prob = report["manipulation_probability"]
    conf = report["authenticity_confidence_pct"]
    risk = report["risk_level"]

    print("\n" + "-" * 40)
    print(f" VERDICT               : {verdict}")
    print(f" AUTHENTICITY SCORE    : {conf:.2f}%")
    print(f" MANIPULATION PROB     : {prob:.4f}")
    print(f" RISK TIER             : {risk}")
    print("-" * 40)
    print(f" > Spatial Stream Score: {report['stream_scores']['spatial']:.4f}")
    print(f" > Freq 2D FFT Score   : {report['stream_scores']['frequency']:.4f}")
    print(f" > Biological Score    : {report['stream_scores']['biological']:.4f}")
    print(f" > Metadata Score      : {report['stream_scores']['metadata']:.4f}")
    print("-" * 40)
    print(" KEY FORENSIC FINDINGS:")
    for finding in report["findings"]:
        print(f"   * {finding}")
    print("-" * 40)

    # Save visual artifacts if requested
    if save_visuals:
        base_stem = Path(file_path).stem
        parent_dir = Path(file_path).parent
        ela_path = parent_dir / f"{base_stem}_ela.png"
        fft_path = parent_dir / f"{base_stem}_fft.png"
        
        report["visual_assets"]["ela_image"].save(ela_path)
        report["visual_assets"]["fft_spectrum_image"].save(fft_path)
        print(f"[+] Saved visual ELA heatmap: {ela_path}")
        print(f"[+] Saved 2D FFT spectrum   : {fft_path}")

    # Export clean JSON
    clean_report = {k: v for k, v in report.items() if k != "visual_assets"}
    if output_json:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(clean_report, f, indent=2)
        print(f"[+] Forensic audit report written to: {output_json}")

    return clean_report

def analyze_directory(pipeline: DeepGuardPipeline, dir_path: str, output_json: str = None):
    dir_p = Path(dir_path)
    if not dir_p.is_dir():
        print(f"[-] Error: Directory not found at '{dir_path}'")
        sys.exit(1)

    extensions = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]
    files = [f for f in dir_p.iterdir() if f.suffix.lower() in extensions]

    if not files:
        print(f"[-] No valid image files found in '{dir_path}'")
        return

    print(f"\n[+] Batch auditing {len(files)} media files in '{dir_path}'...")
    batch_results = []

    for idx, f in enumerate(files, 1):
        try:
            report = pipeline.analyze(str(f))
            clean_rep = {k: v for k, v in report.items() if k != "visual_assets"}
            clean_rep["file_path"] = str(f)
            batch_results.append(clean_rep)
            print(f" [{idx}/{len(files)}] {f.name} -> {report['verdict']} (Confidence: {report['authenticity_confidence_pct']:.1f}%)")
        except Exception as e:
            print(f" [!] Failed processing {f.name}: {e}")

    if output_json:
        with open(output_json, "w", encoding="utf-8") as out_f:
            json.dump(batch_results, out_f, indent=2)
        print(f"\n[+] Batch forensic audit report exported to: {output_json}")

def main():
    print_banner()
    parser = argparse.ArgumentParser(description="DeepGuard Forensic CLI")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-i", "--input", help="Path to single image file for verification")
    group.add_argument("-d", "--dir", help="Directory path to batch process images")
    
    parser.add_argument("-o", "--output", help="Path to write JSON forensic audit report")
    parser.add_argument("--save-visuals", action="store_true", help="Save ELA heatmap and FFT spectrum images")

    args = parser.parse_args()
    pipeline = DeepGuardPipeline()

    if args.input:
        analyze_single_file(pipeline, args.input, output_json=args.output, save_visuals=args.save_visuals)
    elif args.dir:
        analyze_directory(pipeline, args.dir, output_json=args.output)

if __name__ == "__main__":
    main()
