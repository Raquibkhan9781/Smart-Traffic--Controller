from pathlib import Path
from collections import Counter

import cv2
from ultralytics import YOLO


# ============================================================
# VEHICLE DETECTION TEST - YOLO11s
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

VIDEO_PATH = BASE_DIR / "video" / "traffic.mp4"
MODEL_PATH = BASE_DIR / "yolo11s.pt"

# Standard COCO vehicle classes
VEHICLE_CLASSES = {
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
}

CONFIDENCE = 0.15

# Tiling makes small overhead vehicles easier to detect.
TILE_W = 960
TILE_H = 640


def build_tiles(frame):
    h, w = frame.shape[:2]

    xs = [0, max(0, w - TILE_W)]
    ys = [0, max(0, h - TILE_H)]

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
            crop = frame[y:min(y + TILE_H, h), x:min(x + TILE_W, w)]
            tiles.append((x, y, crop))

    return tiles


def detect_vehicles(model, frame):
    all_boxes = []
    all_scores = []
    all_names = []

    for offset_x, offset_y, tile in build_tiles(frame):
        results = model.predict(
            tile,
            imgsz=960,
            conf=CONFIDENCE,
            verbose=False,
        )

        boxes = results[0].boxes
        if boxes is None:
            continue

        for i in range(len(boxes)):
            class_id = int(boxes.cls[i])
            name = model.names[class_id]

            if name not in VEHICLE_CLASSES:
                continue

            score = float(boxes.conf[i])

            x1, y1, x2, y2 = map(
                int,
                boxes.xyxy[i]
            )

            x1 += offset_x
            x2 += offset_x
            y1 += offset_y
            y2 += offset_y

            all_boxes.append([x1, y1, x2, y2])
            all_scores.append(score)
            all_names.append(name)

    if not all_boxes:
        return []

    # Remove duplicate detections caused by overlapping tiles.
    boxes_for_nms = []
    for x1, y1, x2, y2 in all_boxes:
        boxes_for_nms.append([
            x1,
            y1,
            max(1, x2 - x1),
            max(1, y2 - y1),
        ])

    indices = cv2.dnn.NMSBoxes(
        boxes_for_nms,
        all_scores,
        CONFIDENCE,
        0.35,
    )

    if indices is None or len(indices) == 0:
        return []

    indices = indices.flatten().tolist()

    detections = []
    for idx in indices:
        detections.append({
            "box": all_boxes[idx],
            "score": all_scores[idx],
            "type": all_names[idx],
        })

    return detections


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"yolo11s.pt not found here:\n{MODEL_PATH}\n\n"
            "Put yolo11s.pt in the Smart_Traffic_Final folder."
        )

    if not VIDEO_PATH.exists():
        raise FileNotFoundError(
            f"traffic.mp4 not found here:\n{VIDEO_PATH}"
        )

    print("Loading YOLO11s...")
    model = YOLO(str(MODEL_PATH))

    print("Opening traffic video...")
    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        raise RuntimeError("Could not open traffic.mp4")

    while True:
        ret, frame = cap.read()

        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        detections = detect_vehicles(model, frame)

        counts = Counter(
            d["type"] for d in detections
        )

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
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2,
            )

        summary = " | ".join(
            f"{name}:{counts.get(name, 0)}"
            for name in sorted(VEHICLE_CLASSES)
        )

        cv2.putText(
            frame,
            f"VEHICLES: {len(detections)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            3,
        )

        cv2.putText(
            frame,
            summary,
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            "YOLO11s detection test - Q = quit",
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )

        cv2.imshow(
            "Smart Traffic - YOLO11s",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()