from pathlib import Path
import cv2
from ultralytics import YOLO

# ============================================================
# SMART TRAFFIC V2
# Purpose: First make small overhead vehicles detectable.
# Then we will use the detections for tracking/turn detection.
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "yolo11n.pt"
VIDEO_PATH = BASE_DIR / "video" / "traffic.mp4"

# Standard COCO vehicle classes
VEHICLE_CLASSES = {
    "car",
    "motorcycle",
    "bus",
    "truck",
    "bicycle",
}

CONFIDENCE = 0.12
TILE_W = 960
TILE_H = 640
OVERLAP = 0.20


def make_tiles(frame):
    """Create overlapping tiles so small overhead vehicles are larger to YOLO."""
    h, w = frame.shape[:2]

    # Two columns x two rows with overlap.
    xs = [0, max(0, w - TILE_W)]
    ys = [0, max(0, h - TILE_H)]

    # If the image is smaller, use one tile.
    if w <= TILE_W:
        xs = [0]
    if h <= TILE_H:
        ys = [0]

    tiles = []
    seen = set()

    for y in ys:
        for x in xs:
            x = min(x, max(0, w - TILE_W))
            y = min(y, max(0, h - TILE_H))

            key = (x, y)
            if key in seen:
                continue
            seen.add(key)

            crop = frame[
                y:min(y + TILE_H, h),
                x:min(x + TILE_W, w)
            ]

            tiles.append((x, y, crop))

    return tiles


def detect_tiled(model, frame):
    """Run YOLO on overlapping tiles and merge duplicate detections."""
    all_boxes = []
    all_scores = []
    all_labels = []

    for offset_x, offset_y, tile in make_tiles(frame):
        results = model.predict(
            source=tile,
            imgsz=960,
            conf=CONFIDENCE,
            verbose=False,
            device="cpu",
        )

        result = results[0]

        if result.boxes is None:
            continue

        for i in range(len(result.boxes)):
            cls_id = int(result.boxes.cls[i])
            name = model.names[cls_id]

            if name not in VEHICLE_CLASSES:
                continue

            score = float(result.boxes.conf[i])
            x1, y1, x2, y2 = map(int, result.boxes.xyxy[i])

            # Convert tile coordinates back to full-frame coordinates.
            x1 += offset_x
            x2 += offset_x
            y1 += offset_y
            y2 += offset_y

            all_boxes.append([x1, y1, x2 - x1, y2 - y1])
            all_scores.append(score)
            all_labels.append(name)

    if not all_boxes:
        return []

    # Merge overlapping detections from neighboring tiles.
    indices = cv2.dnn.NMSBoxes(
        all_boxes,
        all_scores,
        score_threshold=CONFIDENCE,
        nms_threshold=0.35,
    )

    if indices is None or len(indices) == 0:
        return []

    indices = indices.flatten().tolist()

    detections = []
    for idx in indices:
        x, y, w, h = all_boxes[idx]
        detections.append({
            "box": (x, y, x + w, y + h),
            "score": all_scores[idx],
            "type": all_labels[idx],
        })

    return detections


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    if not VIDEO_PATH.exists():
        raise FileNotFoundError(
            f"Video not found: {VIDEO_PATH}"
        )

    print("Loading YOLO model...")
    model = YOLO(str(MODEL_PATH))

    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    while True:
        ret, frame = cap.read()

        if not ret:
            # Loop the short test video.
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        detections = detect_tiled(model, frame)

        for det in detections:
            x1, y1, x2, y2 = det["box"]

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            label = f'{det["type"]} {det["score"]:.2f}'

            cv2.putText(
                frame,
                label,
                (x1, max(20, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
            )

        # Large, obvious detector status.
        cv2.putText(
            frame,
            f"DETECTED VEHICLES: {len(detections)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            3,
        )

        cv2.putText(
            frame,
            "Detection test - press Q to quit",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )

        cv2.imshow("AI Smart Traffic - Detection Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()