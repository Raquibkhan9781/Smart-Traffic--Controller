from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "yolo11s.pt"
VIDEO_PATH = BASE_DIR / "video" / "traffic.mp4"

VEHICLE_CLASSES = {"bicycle", "car", "motorcycle", "bus", "truck"}
CONFIDENCE = 0.08

# Overlapping tiles work better for the small overhead vehicles in your video.
TILES = [
    (0, 0, 960, 620),
    (480, 0, 1440, 620),
    (960, 0, 1920, 620),
    (0, 460, 960, 1080),
    (480, 460, 1440, 1080),
    (960, 460, 1920, 1080),
]

@dataclass
class Detection:
    box: Tuple[int, int, int, int]
    score: float
    kind: str

@dataclass
class Track:
    track_id: int
    box: Tuple[int, int, int, int]
    kind: str
    score: float
    missed: int = 0
    age: int = 1


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(1, ax2 - ax1) * max(1, ay2 - ay1)
    area_b = max(1, bx2 - bx1) * max(1, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def center_distance(a, b) -> float:
    ax = (a[0] + a[2]) / 2
    ay = (a[1] + a[3]) / 2
    bx = (b[0] + b[2]) / 2
    by = (b[1] + b[3]) / 2
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


class SimpleTracker:
    def __init__(self):
        self.tracks: List[Track] = []
        self.next_id = 1
        self.max_missed = 8
        self.max_distance = 130
        self.min_iou = 0.10

    def update(self, detections: List[Detection]) -> List[Track]:
        if not self.tracks:
            for d in detections:
                self.tracks.append(Track(self.next_id, d.box, d.kind, d.score))
                self.next_id += 1
            return self.tracks

        candidates = []
        for ti, tr in enumerate(self.tracks):
            for di, d in enumerate(detections):
                if tr.kind != d.kind:
                    continue
                overlap = iou(tr.box, d.box)
                dist = center_distance(tr.box, d.box)
                if overlap >= self.min_iou or dist <= self.max_distance:
                    score = overlap + max(0.0, 1.0 - dist / self.max_distance)
                    candidates.append((score, ti, di))

        candidates.sort(reverse=True)
        matched_t, matched_d = set(), set()

        for _, ti, di in candidates:
            if ti in matched_t or di in matched_d:
                continue
            tr, d = self.tracks[ti], detections[di]
            tr.box = d.box
            tr.score = d.score
            tr.kind = d.kind
            tr.missed = 0
            tr.age += 1
            matched_t.add(ti)
            matched_d.add(di)

        for ti, tr in enumerate(self.tracks):
            if ti not in matched_t:
                tr.missed += 1

        for di, d in enumerate(detections):
            if di in matched_d:
                continue
            self.tracks.append(Track(self.next_id, d.box, d.kind, d.score))
            self.next_id += 1

        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]
        return self.tracks


def detect(model, frame) -> List[Detection]:
    raw_boxes, raw_scores, raw_names = [], [], []
    h, w = frame.shape[:2]

    for x1, y1, x2, y2 in TILES:
        x1, x2 = max(0, min(x1, w)), max(0, min(x2, w))
        y1, y2 = max(0, min(y1, h)), max(0, min(y2, h))
        if x2 <= x1 or y2 <= y1:
            continue
        tile = frame[y1:y2, x1:x2]
        result = model.predict(tile, imgsz=1280, conf=CONFIDENCE, iou=0.45,
                               max_det=100, verbose=False)[0]
        if result.boxes is None:
            continue
        for i in range(len(result.boxes)):
            name = model.names[int(result.boxes.cls[i])]
            if name not in VEHICLE_CLASSES:
                continue
            score = float(result.boxes.conf[i])
            bx1, by1, bx2, by2 = map(int, result.boxes.xyxy[i])
            bx1, bx2, by1, by2 = bx1 + x1, bx2 + x1, by1 + y1, by2 + y1
            raw_boxes.append([bx1, by1, max(1, bx2 - bx1), max(1, by2 - by1)])
            raw_scores.append(score)
            raw_names.append(name)

    if not raw_boxes:
        return []

    keep = cv2.dnn.NMSBoxes(raw_boxes, raw_scores, CONFIDENCE, 0.25)
    if keep is None or len(keep) == 0:
        return []

    keep = keep.flatten().tolist()
    out = []
    for i in keep:
        x, y, bw, bh = raw_boxes[i]
        out.append(Detection((x, y, x + bw, y + bh), raw_scores[i], raw_names[i]))
    return out


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model: {MODEL_PATH}")
    if not VIDEO_PATH.exists():
        raise FileNotFoundError(f"Missing video: {VIDEO_PATH}")

    model = YOLO(str(MODEL_PATH))
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError("Could not open traffic.mp4")

    tracker = SimpleTracker()

    while True:
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        detections = detect(model, frame)
        tracks = tracker.update(detections)
        visible = [t for t in tracks if t.missed == 0]

        for tr in visible:
            x1, y1, x2, y2 = tr.box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"ID {tr.track_id} | {tr.kind}"
            cv2.putText(frame, label, (x1, max(20, y1 - 7)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        cv2.putText(frame, f"VISIBLE VEHICLES: {len(visible)}", (15, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 3)
        cv2.putText(frame, f"ACTIVE TRACKS: {len(tracker.tracks)}", (15, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(frame, "Tracking test | Q = quit", (15, 108),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        cv2.imshow("Smart Traffic - Vehicle Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()