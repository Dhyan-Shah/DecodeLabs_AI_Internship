"""
Project 4 - Path 1: Optical Character Recognition (OCR)
DecodeLabs AI Industrial Training Kit | Batch 2026

Pipeline:
  Raw Image → Grayscale → Gaussian Blur → Deskew → Adaptive Threshold → pytesseract → Output

Requirements:
    pip install pytesseract opencv-python pillow numpy
    Install Tesseract engine: https://github.com/UB-Mannheim/tesseract/wiki
"""

import cv2
import numpy as np
import pytesseract
from PIL import Image
import os
import sys

# ─── CONFIGURATION ──────────────────────────────────────────────────────────────

# If Tesseract is not on PATH, set the executable path here (Windows example):
pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

CONFIDENCE_THRESHOLD = 80  # Minimum confidence % to accept a word (per-word check)

# Page Segmentation Modes:
#   3  → Fully automatic (default, varied layouts)
#   6  → Single uniform block of text (book pages)
#   7  → Single text line (number plates, headers)
#   11 → Sparse, scattered text (invoices, forms)
PSM_MODE = 6

# ─── STEP 1: LOAD IMAGE ─────────────────────────────────────────────────────────

def load_image(image_path: str) -> np.ndarray:
    """Load image from disk and validate."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    print(f"[✓] Loaded image: {image_path}  |  Shape: {img.shape}")
    return img


# ─── STEP 2: GRAYSCALE CONVERSION ───────────────────────────────────────────────

def convert_to_grayscale(img: np.ndarray) -> np.ndarray:
    """
    Collapse the 3D RGB matrix into a 1D intensity matrix.
    Removes distracting color data so the model focuses on luminance contrast.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(f"[✓] Grayscale conversion complete  |  Shape: {gray.shape}")
    return gray


# ─── STEP 3: GAUSSIAN BLUR ──────────────────────────────────────────────────────

def apply_gaussian_blur(gray: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Smooth the image to eliminate micro-imperfections and artifact noise.
    kernel_size must be odd (3, 5, 7...). Larger = more smoothing.
    """
    blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    print(f"[✓] Gaussian blur applied  |  Kernel: {kernel_size}x{kernel_size}")
    return blurred


# ─── STEP 4: DESKEWING ──────────────────────────────────────────────────────────

def deskew(img: np.ndarray) -> np.ndarray:
    """
    Calculate rotation angle and snap tilted text back to a
    perfect horizontal baseline using the minimum bounding rectangle.
    """
    # Find all non-zero (text) pixels
    coords = np.column_stack(np.where(img > 0))
    if coords.size == 0:
        print("[!] Deskew skipped: no foreground pixels detected")
        return img

    # Get the minimum area rectangle enclosing all text pixels
    angle = cv2.minAreaRect(coords)[-1]

    # Correct angle interpretation
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    print(f"[✓] Deskew detected angle: {angle:.2f}°")

    # Rotate the image to correct the skew
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    deskewed = cv2.warpAffine(img, M, (w, h),
                              flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
    return deskewed


# ─── STEP 5: ADAPTIVE THRESHOLDING (Otsu's Method) ──────────────────────────────

def apply_threshold(img: np.ndarray) -> np.ndarray:
    """
    Force every pixel to choose a side — black or white.
    Otsu's method automatically finds the optimal cutoff intensity.

    IF pixel_intensity >= cutoff → pixel = 255 (White)
    IF pixel_intensity <  cutoff → pixel = 0   (Black)
    """
    _, thresh = cv2.threshold(img, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print(f"[✓] Otsu's thresholding applied → binary image")
    return thresh


# ─── STEP 6: OCR EXTRACTION ─────────────────────────────────────────────────────

def extract_text(processed_img: np.ndarray, psm: int = PSM_MODE) -> str:
    """
    Run pytesseract on the pre-processed binary image.
    Returns the full extracted text string.
    """
    config = f"--psm {psm} --oem 3"  # OEM 3 = Default (LSTM + legacy)
    pil_img = Image.fromarray(processed_img)
    text = pytesseract.image_to_string(pil_img, config=config)
    return text.strip()


# ─── STEP 7: CONFIDENCE FILTERING ───────────────────────────────────────────────

def extract_with_confidence(processed_img: np.ndarray,
                            psm: int = PSM_MODE,
                            threshold: int = CONFIDENCE_THRESHOLD) -> dict:
    """
    Extract text word-by-word with confidence scores.
    Only words meeting the minimum confidence threshold are kept.

    Returns:
        {
          'accepted_words': [...],
          'rejected_words': [...],
          'full_text':      "...",
          'avg_confidence': float,
          'passed':         bool
        }
    """
    config = f"--psm {psm} --oem 3"
    pil_img = Image.fromarray(processed_img)
    data = pytesseract.image_to_data(pil_img, config=config,
                                     output_type=pytesseract.Output.DICT)

    accepted_words = []
    rejected_words = []

    for i, word in enumerate(data['text']):
        word = word.strip()
        conf = int(data['conf'][i])
        if word == "" or conf == -1:
            continue
        if conf >= threshold:
            accepted_words.append((word, conf))
        else:
            rejected_words.append((word, conf))

    full_text = " ".join([w for w, _ in accepted_words])
    confidences = [c for _, c in accepted_words] if accepted_words else [0]
    avg_conf = sum(confidences) / len(confidences)

    return {
        "accepted_words": accepted_words,
        "rejected_words": rejected_words,
        "full_text": full_text,
        "avg_confidence": avg_conf,
        "passed": avg_conf >= threshold
    }


# ─── STEP 8: SAVE DEBUG OUTPUT ──────────────────────────────────────────────────

def save_debug_images(original: np.ndarray,
                      gray: np.ndarray,
                      blurred: np.ndarray,
                      thresh: np.ndarray,
                      output_dir: str = "ocr_output") -> None:
    """Save intermediate pipeline stages for visual inspection."""
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(os.path.join(output_dir, "1_original.jpg"), original)
    cv2.imwrite(os.path.join(output_dir, "2_grayscale.jpg"), gray)
    cv2.imwrite(os.path.join(output_dir, "3_blurred.jpg"), blurred)
    cv2.imwrite(os.path.join(output_dir, "4_threshold.jpg"), thresh)
    print(f"[✓] Debug images saved to: {output_dir}/")


# ─── MAIN PIPELINE ──────────────────────────────────────────────────────────────

def run_ocr_pipeline(image_path: str,
                     psm: int = PSM_MODE,
                     save_debug: bool = True) -> None:
    """
    Full OCR pipeline:
      Load → Grayscale → Blur → Deskew → Threshold → OCR → Confidence Filter
    """
    print("\n" + "═" * 60)
    print("  PROJECT 4 — PATH 1: OCR PIPELINE")
    print("═" * 60)

    # Stage 1: Load
    original = load_image(image_path)

    # Stage 2: Grayscale
    gray = convert_to_grayscale(original)

    # Stage 3: Blur
    blurred = apply_gaussian_blur(gray, kernel_size=3)

    # Stage 4: Deskew
    deskewed = deskew(blurred)

    # Stage 5: Threshold
    thresh = apply_threshold(deskewed)

    # Stage 6+7: OCR with confidence filtering
    print("\n[→] Running OCR extraction...")
    result = extract_with_confidence(thresh, psm=psm,
                                     threshold=CONFIDENCE_THRESHOLD)

    # Save debug images
    if save_debug:
        save_debug_images(original, gray, blurred, thresh)

    # ─── OUTPUT REPORT ──────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("  EXTRACTION RESULT")
    print("─" * 60)

    print(f"\n  Extracted Text:\n")
    print(f"  {result['full_text'] if result['full_text'] else '[No text detected above threshold]'}")

    print(f"\n  Accepted Words ({len(result['accepted_words'])}):")
    for word, conf in result['accepted_words']:
        bar = "█" * (conf // 10) + "░" * (10 - conf // 10)
        print(f"    [{bar}] {conf:3d}%  →  {word}")

    if result['rejected_words']:
        print(f"\n  Rejected Words (below {CONFIDENCE_THRESHOLD}% threshold):")
        for word, conf in result['rejected_words']:
            print(f"    [DROPPED] {conf:3d}%  →  {word}")

    print(f"\n  Average Confidence : {result['avg_confidence']:.1f}%")
    status = "✅ PASSED" if result['passed'] else "❌ FAILED"
    print(f"  Validation Status  : {status} (Threshold: {CONFIDENCE_THRESHOLD}%)")
    print("─" * 60 + "\n")

    return result


# ─── ENTRY POINT ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Usage: python path1_ocr.py <image_path> [psm_mode]
    # Example: python path1_ocr.py invoice.jpg 11

    if len(sys.argv) < 2:
        print("Usage: python path1_ocr.py <image_path> [psm_mode]")
        print("  psm_mode options:")
        print("    3  → Fully automatic (default)")
        print("    6  → Single block of text (books)")
        print("    7  → Single line (number plates)")
        print("    11 → Sparse text (invoices)")
        sys.exit(1)

    image_path = sys.argv[1]
    psm = int(sys.argv[2]) if len(sys.argv) > 2 else PSM_MODE

    run_ocr_pipeline(image_path, psm=psm, save_debug=True)