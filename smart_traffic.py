from __future__ import annotations

import time
from dataclasses import dataclass, field
from collections import defaultdict, deque
from pathlib import Path

import cv2
from ultralytics import YOLO

# ==============================
# CONFIG
# ==============================
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "yolo11n.pt"
VIDEO_PATH = BASE_DIR / "video" / "traffic.mp4"

# Camera/video source. Change to 0 for webcam.
SOURCE = str(VIDEO_PATH)

CONFIDENCE = 0.15
IMAGE_SIZE = 1280
SECONDS_PER_VEHICLE = 2
MIN_GREEN = 5
MAX_GREEN = 60
YELLOW_SECONDS = 2

# Standard COCO classes supported by yolo11n.
# Add custom-model class names here later for Indian vehicles.
VEHICLE_CLASSES = {
    "car",
    "motorcycle",
    "bus",
    "truck",
    "bicycle",
}

# Approximate gates calibrated from the uploaded overhead traffic video.
# Adjust these numbers if a different video/camera is used.
GATES = {
    "WEST": ((40, 330), (510, 330), (40, 700), (510, 700)),
    "EAST": ((1130, 180), (1880, 180), (1130, 700), (1880, 700)),
    "NORTH": ((520, 40), (1120, 40), (520, 190), (1120, 190)),
    "SOUTH": ((520, 720), (1120, 720), (520, 1040), (1120, 1040)),
}

# A single trip is classified by entry and exit approach.
MOVEMENT_MAP = {
    ("WEST", "EAST"): "STRAIGHT",
    ("WEST", "NORTH"): "LEFT",
    ("WEST", "SOUTH"): "RIGHT",
    ("EAST", "WEST"): "STRAIGHT",
    ("EAST", "SOUTH"): "LEFT",
    ("EAST", "NORTH"): "RIGHT",
    ("NORTH", "SOUTH"): "STRAIGHT",
    ("NORTH", "EAST"): "LEFT",
    ("NORTH", "WEST"): "RIGHT",
    ("SOUTH", "NORTH"): "STRAIGHT",
    ("SOUTH", "WEST"): "LEFT",
    ("SOUTH", "EAST"): "RIGHT",
}

# Conservative non-conflicting phases for the prototype.
PHASES = {
    "NS_STRAIGHT": [("NORTH", "STRAIGHT"), ("SOUTH", "STRAIGHT")],
    "EW_STRAIGHT": [("EAST", "STRAIGHT"), ("WEST", "STRAIGHT")],
    "NS_LEFT": [("NORTH", "LEFT"), ("SOUTH", "LEFT")],
    "EW_LEFT": [("EAST", "LEFT"), ("WEST", "LEFT")],
    "NS_RIGHT": [("NORTH", "RIGHT"), ("SOUTH", "RIGHT")],
    "EW_RIGHT": [("EAST", "RIGHT"), ("WEST", "RIGHT")],
}


@dataclass
class Track:
    positions: deque = field(default_factory=lambda: deque(maxlen=60))
    entry: str | None = None
    exit: str | None = None
    movement: str | None = None
    counted: bool = False
    last_seen: int = 0


def point_in_rect(x: int, y: int, rect) -> bool:
    x1, y1, x2, y2 = rect
    return x1 <= x <= x2 and y1 <= y <= y2


def crossing_entry(prev, curr, direction: str) -> bool:
    px, py = prev
    cx, cy = curr
    if direction == "WEST":
        return px < 40 <= cx and 330 <= cy <= 700
    if direction == "EAST":
        return px > 1880 >= cx and 180 <= cy <= 700
    if direction == "NORTH":
        return py < 40 <= cy and 520 <= cx <= 1120
    if direction == "SOUTH":
        return py > 1040 >= cy and 520 <= cx <= 1120
    return False


def crossing_exit(prev, curr, direction: str) -> bool:
    px, py = prev
    cx, cy = curr
    if direction == "WEST":
        return px > 40 >= cx and 330 <= cy <= 700
    if direction == "EAST":
        return px < 1880 <= cx and 180 <= cy <= 700
    if direction == "NORTH":
        return py > 40 >= cy and 520 <= cx <= 1120
    if direction == "SOUTH":
        return py < 1040 <= cy and 520 <= cx <= 1120
    return False


def movement_for(entry: str | None, exit_: str | None) -> str | None:
    if entry is None or exit_ is None:
        return None
    return MOVEMENT_MAP.get((entry, exit_))


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n"
            "Put yolo11n.pt inside the models folder."
        )

    cap = cv2.VideoCapture(SOURCE)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video/camera: {SOURCE}")

    model = YOLO(str(MODEL_PATH))
    tracks: dict[int, Track] = {}

    # Only completed movements are counted. This prevents the same vehicle
    # from being added repeatedly on every frame.
    completed_counts = {
        d: {m: 0 for m in ("LEFT", "STRAIGHT", "RIGHT")}
        for d in ("NORTH", "EAST", "SOUTH", "WEST")
    }

    current_phase = None
    current_signal = "RED"
    remaining = 0
    last_tick = time.monotonic()
    last_phase = None

    frame_no = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_no += 1
        h, w = frame.shape[:2]

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            imgsz=IMAGE_SIZE,
            conf=CONFIDENCE,
            verbose=False,
        )

        boxes = results[0].boxes
        active_ids = set()

        if boxes is not None and len(boxes) > 0:
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i])
                cls_name = model.names[cls_id]
                if cls_name not in VEHICLE_CLASSES:
                    continue
                if boxes.id is None:
                    continue

                tid = int(boxes.id[i])
                active_ids.add(tid)
                x1, y1, x2, y2 = map(int, boxes.xyxy[i])
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                tr = tracks.setdefault(tid, Track())
                previous = tr.positions[-1] if tr.positions else None
                tr.positions.append((cx, cy))
                tr.last_seen = frame_no

                if previous is not None and tr.entry is None:
                    for direction in ("WEST", "EAST", "NORTH", "SOUTH"):
                        if crossing_entry(previous, (cx, cy), direction):
                            tr.entry = direction
                            break

                if previous is not None and tr.entry is not None and tr.exit is None:
                    for direction in ("WEST", "EAST", "NORTH", "SOUTH"):
                        if direction == tr.entry:
                            continue
                        if crossing_exit(previous, (cx, cy), direction):
                            tr.exit = direction
                            tr.movement = movement_for(tr.entry, tr.exit)
                            if tr.movement and not tr.counted:
                                completed_counts[tr.entry][tr.movement] += 1
                                tr.counted = True
                            break

                label = f"{cls_name} ID:{tid}"
                if tr.entry:
                    label += f" IN:{tr.entry}"
                if tr.movement:
                    label += f" -> {tr.movement}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, max(18, y1 - 7)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 2)

        # Remove very old tracks, but keep completed-count totals intact.
        stale = [tid for tid, tr in tracks.items() if frame_no - tr.last_seen > 45]
        for tid in stale:
            del tracks[tid]

        # Demand for each safe phase = sum of its movement queues.
        demands = {}
        for phase, pairs in PHASES.items():
            demands[phase] = sum(completed_counts[d][m] for d, m in pairs)

        now = time.monotonic()
        if current_phase is None:
            candidates = {p: c for p, c in demands.items() if c > 0}
            if last_phase in candidates and len(candidates) > 1:
                candidates.pop(last_phase, None)
            if candidates:
                current_phase = max(candidates, key=candidates.get)
                demand = demands[current_phase]
                remaining = max(MIN_GREEN, min(MAX_GREEN, demand * SECONDS_PER_VEHICLE))
                current_signal = "GREEN"
                last_tick = now
            else:
                current_signal = "RED"
                remaining = 0
        else:
            elapsed = int(now - last_tick)
            if elapsed > 0:
                remaining -= elapsed
                last_tick = now
            if remaining <= 0 and current_signal == "GREEN":
                current_signal = "YELLOW"
                remaining = YELLOW_SECONDS
                last_tick = now
            elif remaining <= 0 and current_signal == "YELLOW":
                last_phase = current_phase
                current_phase = None
                current_signal = "RED"
                remaining = 0

        # Draw dashboard text.
        cv2.putText(frame, "AI SMART TRAFFIC CONTROLLER", (25, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        lines = [
            f"NORTH  L:{completed_counts['NORTH']['LEFT']} S:{completed_counts['NORTH']['STRAIGHT']} R:{completed_counts['NORTH']['RIGHT']}",
            f"EAST   L:{completed_counts['EAST']['LEFT']} S:{completed_counts['EAST']['STRAIGHT']} R:{completed_counts['EAST']['RIGHT']}",
            f"SOUTH  L:{completed_counts['SOUTH']['LEFT']} S:{completed_counts['SOUTH']['STRAIGHT']} R:{completed_counts['SOUTH']['RIGHT']}",
            f"WEST   L:{completed_counts['WEST']['LEFT']} S:{completed_counts['WEST']['STRAIGHT']} R:{completed_counts['WEST']['RIGHT']}",
        ]
        y = 72
        for line in lines:
            cv2.putText(frame, line, (25, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 255, 255), 2)
            y += 30

        phase_text = current_phase if current_phase else "NO ACTIVE PHASE"
        status = f"PHASE: {phase_text} | {current_signal} | {remaining}s"
        color = (0, 255, 0) if current_signal == "GREEN" else (0, 255, 255) if current_signal == "YELLOW" else (0, 0, 255)
        cv2.putText(frame, status, (25, h - 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, color, 2)

        # Draw calibrated gate lines.
        cv2.line(frame, (40, 330), (40, 700), (255, 0, 255), 2)
        cv2.line(frame, (1880, 180), (1880, 700), (255, 0, 255), 2)
        cv2.line(frame, (520, 40), (1120, 40), (255, 0, 255), 2)
        cv2.line(frame, (520, 1040), (1120, 1040), (255, 0, 255), 2)

        cv2.imshow("AI Smart Traffic Controller", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
