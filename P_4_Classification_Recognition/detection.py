"""
Project 4 - Path 2: Object Detection with MobileNet-SSD
DecodeLabs AI Industrial Training Kit | Batch 2026

Pipeline:
  Raw Image → Blob Construction (300x300) → MobileNet-SSD → Softmax →
  Confidence Filter (≥80%) → Bounding Box Overlay → Output

Requirements:
    pip install opencv-python numpy

Model Files (download before running):
    MobileNetSSD_deploy.prototxt   → https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/MobileNetSSD_deploy.prototxt
    MobileNetSSD_deploy.caffemodel → https://drive.google.com/open?id=0B3gersZ2cHIxRm5PMWRoTkdHdHc
    (or use the script's auto-download below)
"""

import cv2
import numpy as np
import os
import sys
import urllib.request

# ─── CONFIGURATION ──────────────────────────────────────────────────────────────

CONFIDENCE_THRESHOLD = 0.80   # 80% minimum — the Gatekeeper Rule
MODEL_INPUT_SIZE     = 300    # MobileNet-SSD requires 300x300 input
MEAN_SUBTRACTION     = (127.5, 127.5, 127.5)  # ImageNet mean for normalization
SCALE_FACTOR         = 1 / 127.5              # Normalize pixel values to [-1, 1]

# COCO / VOC class labels that MobileNet-SSD was trained on (21 classes)
CLASSES = [
    "background", "aeroplane", "bicycle",   "bird",  "boat",
    "bottle",     "bus",        "car",       "cat",   "chair",
    "cow",        "diningtable","dog",        "horse", "motorbike",
    "person",     "pottedplant","sheep",      "sofa",  "train",
    "tvmonitor"
]

# Assign a distinct BGR color to each class for bounding boxes
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(CLASSES), 3), dtype="uint8")

# Model file paths (place in same directory or update paths)
PROTOTXT_PATH   = "C:\\Users\\Dell\\Downloads\\deploy.prototxt"
CAFFEMODEL_PATH = "C:\\Users\\Dell\\Downloads\\mobilenet_iter_73000.caffemodel"

# ─── AUTO-DOWNLOAD MODEL FILES ───────────────────────────────────────────────────

PROTOTXT_URL = (
    "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/"
    "master/MobileNetSSD_deploy.prototxt"
)

def download_prototxt() -> None:
    """Download the prototxt config file if missing."""
    if not os.path.exists(PROTOTXT_PATH):
        print(f"[↓] Downloading prototxt from GitHub...")
        urllib.request.urlretrieve(PROTOTXT_URL, PROTOTXT_PATH)
        print(f"[✓] Saved: {PROTOTXT_PATH}")
    else:
        print(f"[✓] Found prototxt: {PROTOTXT_PATH}")

def check_model_files() -> None:
    """Validate that both model files are present."""
    download_prototxt()
    if not os.path.exists(CAFFEMODEL_PATH):
        print(f"\n[✗] Missing model weights: {CAFFEMODEL_PATH}")
        print("    Download from:")
        print("    https://github.com/chuanqi305/MobileNet-SSD")
        print("    (MobileNetSSD_deploy.caffemodel)")
        sys.exit(1)
    print(f"[✓] Found caffemodel: {CAFFEMODEL_PATH}")


# ─── STEP 1: LOAD MODEL ─────────────────────────────────────────────────────────

def load_model() -> cv2.dnn_Net:
    """Load the MobileNet-SSD Caffe model into cv2.dnn."""
    check_model_files()
    print("\n[→] Loading MobileNet-SSD model...")
    net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, CAFFEMODEL_PATH)
    print("[✓] Model loaded successfully")
    return net


# ─── STEP 2: LOAD IMAGE ─────────────────────────────────────────────────────────

def load_image(image_path: str) -> np.ndarray:
    """Load and validate the input image."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    print(f"[✓] Loaded image: {image_path}  |  Shape: {img.shape}")
    return img


# ─── STEP 3: BLOB CONSTRUCTION ───────────────────────────────────────────────────

def build_blob(img: np.ndarray) -> np.ndarray:
    """
    Convert the image into a 4D blob for the DNN input layer.

    cv2.dnn.blobFromImage performs:
      1. Resize to MODEL_INPUT_SIZE × MODEL_INPUT_SIZE (300×300)
      2. Mean subtraction  → removes lighting/color bias
      3. Scale factor      → normalizes pixel values to [-1.0, 1.0]
      4. Channel swap      → BGR to RGB (MobileNet expects RGB)

    Output shape: (1, 3, 300, 300)
    """
    blob = cv2.dnn.blobFromImage(
        image=img,
        scalefactor=SCALE_FACTOR,
        size=(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
        mean=MEAN_SUBTRACTION,
        swapRB=True,
        crop=False
    )
    print(f"[✓] Blob constructed  |  Shape: {blob.shape}")
    return blob


# ─── STEP 4: FORWARD PASS (INFERENCE) ───────────────────────────────────────────

def run_inference(net: cv2.dnn_Net, blob: np.ndarray) -> np.ndarray:
    """
    Feed the blob through MobileNet-SSD.
    The network returns detections: shape (1, 1, N, 7)

    Each detection row:
      [image_id, class_id, confidence, x_min, y_min, x_max, y_max]
      Coordinates are normalized to [0, 1] — must be scaled to pixel space.
    """
    net.setInput(blob)
    detections = net.forward()
    print(f"[✓] Inference complete  |  Raw detections: {detections.shape[2]}")
    return detections


# ─── STEP 5: CONFIDENCE FILTER + BOUNDING BOXES ──────────────────────────────────

def process_detections(img: np.ndarray,
                        detections: np.ndarray,
                        threshold: float = CONFIDENCE_THRESHOLD) -> dict:
    """
    Apply the 80% Gatekeeper Rule:
      - confidence >= 0.80 → draw_box_and_label()
      - confidence <  0.80 → drop_detection()

    Scales normalized coordinates back to actual pixel dimensions.

    Returns result dict with accepted detections and annotated image.
    """
    (h, w) = img.shape[:2]
    output_img   = img.copy()
    accepted      = []
    dropped_count = 0

    # detections shape: (1, 1, N, 7)
    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])

        # ── The Gatekeeper Rule ──────────────────────────────────────
        if confidence < threshold:
            dropped_count += 1
            continue  # drop_detection()
        # ─────────────────────────────────────────────────────────────

        class_id = int(detections[0, 0, i, 1])

        if class_id >= len(CLASSES):
            continue

        label = CLASSES[class_id]
        color = [int(c) for c in COLORS[class_id]]

        # Scale normalized [0,1] coords → actual pixel coordinates
        box   = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (x1, y1, x2, y2) = box.astype("int")

        # Clamp to image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        # ── Draw Bounding Box ───────────────────────────────────────
        cv2.rectangle(output_img, (x1, y1), (x2, y2), color, thickness=2)

        # ── Draw Label ──────────────────────────────────────────────
        text = f"{label}: {confidence * 100:.1f}%"
        (text_w, text_h), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        # Filled background rectangle for label
        label_y = max(y1 - 10, text_h + 10)
        cv2.rectangle(output_img,
                      (x1, label_y - text_h - baseline),
                      (x1 + text_w, label_y + baseline),
                      color, cv2.FILLED)
        # White text on colored background
        cv2.putText(output_img, text, (x1, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        accepted.append({
            "class_id":   class_id,
            "label":      label,
            "confidence": confidence,
            "box":        (x1, y1, x2 - x1, y2 - y1)   # (X, Y, W, H)
        })

    return {
        "accepted":      accepted,
        "dropped_count": dropped_count,
        "output_image":  output_img,
        "passed":        len(accepted) > 0
    }


# ─── STEP 6: SAVE OUTPUT ────────────────────────────────────────────────────────

def save_output(output_img: np.ndarray,
                output_path: str = "detection_output.jpg") -> None:
    """Save the annotated image with bounding boxes to disk."""
    cv2.imwrite(output_path, output_img)
    print(f"[✓] Annotated image saved: {output_path}")


# ─── MAIN PIPELINE ──────────────────────────────────────────────────────────────

def run_detection_pipeline(image_path: str) -> None:
    """
    Full Object Detection pipeline:
      Load Model → Load Image → Blob → Inference → Filter → Draw → Save
    """
    print("\n" + "═" * 60)
    print("  PROJECT 4 — PATH 2: OBJECT DETECTION PIPELINE")
    print("═" * 60)

    # Stage 1: Load model
    net = load_model()

    # Stage 2: Load image
    img = load_image(image_path)

    # Stage 3: Build blob
    blob = build_blob(img)

    # Stage 4: Forward pass
    detections = run_inference(net, blob)

    # Stage 5: Filter + annotate
    print(f"\n[→] Applying confidence threshold: {CONFIDENCE_THRESHOLD * 100:.0f}%")
    result = process_detections(img, detections, threshold=CONFIDENCE_THRESHOLD)

    # Stage 6: Save
    base_name = os.path.splitext(image_path)[0]
    output_path = f"{base_name}_detected.jpg"
    save_output(result["output_image"], output_path)

    # ─── OUTPUT REPORT ──────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("  DETECTION RESULT")
    print("─" * 60)

    if result["accepted"]:
        print(f"\n  Objects Detected: {len(result['accepted'])}")
        print()
        for det in result["accepted"]:
            (x, y, w_box, h_box) = det["box"]
            conf_pct = det["confidence"] * 100
            bar = "█" * int(conf_pct // 10) + "░" * (10 - int(conf_pct // 10))
            print(f"  [{bar}] {conf_pct:5.1f}%  →  {det['label'].upper()}")
            print(f"            Bounding Box: X={x}, Y={y}, W={w_box}, H={h_box}")
    else:
        print("\n  [!] No objects detected above the 80% confidence threshold.")

    print(f"\n  Detections dropped (below threshold): {result['dropped_count']}")
    status = "✅ PASSED" if result["passed"] else "❌ FAILED"
    print(f"  Validation Status : {status} (Threshold: {CONFIDENCE_THRESHOLD * 100:.0f}%)")
    print(f"  Output image      : {output_path}")
    print("─" * 60 + "\n")

    return result


# ─── ENTRY POINT ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Usage: python path2_detection.py <image_path>
    # Example: python path2_detection.py street.jpg

    if len(sys.argv) < 2:
        print("Usage: python path2_detection.py <image_path>")
        print("\nExample:")
        print("  python path2_detection.py street.jpg")
        print("\nMake sure these model files are in the same directory:")
        print("  - MobileNetSSD_deploy.prototxt   (auto-downloaded)")
        print("  - MobileNetSSD_deploy.caffemodel  (manual download required)")
        sys.exit(1)

    image_path = sys.argv[1]
    run_detection_pipeline(image_path)