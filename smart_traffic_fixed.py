from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
from ultralytics import YOLO

# ============================================================
# SMART TRAFFIC CONTROLLER - CLEAN WORKING MVP
# ============================================================
# Uses the existing models/yolo11n.pt and video/traffic.mp4
# in the same project folder.
#
# Main goals:
# 1) Detect vehicles and give them tracking IDs.
# 2) Show LIVE vehicle detections and LIVE approach queues.
# 3) Detect completed WEST/EAST/NORTH/SOUTH movements when
#    a tracked vehicle enters and exits the intersection.
# 4) Prevent the signal dashboard from staying at zero simply
#    because a vehicle has not completed its trip yet.
# 5) Allocate green time as queue_vehicles * 2 seconds.
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "yolo11n.pt"
VIDEO_PATH = BASE_DIR / "video" / "traffic.mp4"

SOURCE = str(VIDEO_PATH)

# Standard COCO vehicle classes used by yolo11n.
# Auto/e-rickshaw/tractor need a custom Indian-traffic model later.
VEHICLE_NAMES = {"bicycle", "car", "motorcycle", "bus", "truck"}
VEHICLE_CLASS_IDS = [1, 2, 3, 5, 7]

CONFIDENCE = 0.10
IMAGE_SIZE = 1280
TRACKER = "bytetrack.yaml"

SECONDS_PER_VEHICLE = 2
MIN_GREEN = 5
MAX_GREEN = 60
YELLOW_SECONDS = 2

# ------------------------------------------------------------
# Calibration for the uploaded overhead 1920x1080 video.
# These are APPROACH QUEUE ROIs, not screen quadrants.
# ------------------------------------------------------------
APPROACH_ZONES = {
    "WEST": (0, 320, 720, 735),
    "EAST": (1120, 150, 1919, 735),
    "NORTH": (780, 0, 1300, 350),
    "SOUTH": (500, 730, 1200, 1079),
}

# Gates are placed closer to the intersection than the frame edge.
# Crossing direction determines entry/exit approach.
GATE_X_WEST = 720
GATE_X_EAST = 1120
GATE_Y_NORTH = 350
GATE_Y_SOUTH = 730

GATE_Y_MIN = 300
GATE_Y_MAX = 730
GATE_X_MIN = 480
GATE_X_MAX = 1250

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

@dataclass
class Track:
    positions: deque = field(default_factory=lambda: deque(maxlen=50))
    entry: str | None = None
    exit: str | None = None
    movement: str | None = None
    counted_movement: bool = False
    last_seen_frame: int = 0


def inside_rect(x: int, y: int, rect: tuple[int, int, int, int]) -> bool:
    x1, y1, x2, y2 = rect
    return x1 <= x <= x2 and y1 <= y <= y2


def crossed_entry(prev: tuple[int, int], curr: tuple[int, int], direction: str) -> bool:
    px, py = prev
    cx, cy = curr

    if direction == "WEST":
        return px < GATE_X_WEST <= cx and GATE_Y_MIN <= cy <= GATE_Y_MAX
    if direction == "EAST":
        return px > GATE_X_EAST >= cx and GATE_Y_MIN <= cy <= GATE_Y_MAX
    if direction == "NORTH":
        return py < GATE_Y_NORTH <= cy and GATE_X_MIN <= cx <= GATE_X_MAX
    if direction == "SOUTH":
        return py > GATE_Y_SOUTH >= cy and GATE_X_MIN <= cx <= GATE_X_MAX
    return False


def crossed_exit(prev: tuple[int, int], curr: tuple[int, int], direction: str) -> bool:
    px, py = prev
    cx, cy = curr

    if direction == "WEST":
        return px > GATE_X_WEST >= cx and GATE_Y_MIN <= cy <= GATE_Y_MAX
    if direction == "EAST":
        return px < GATE_X_EAST <= cx and GATE_Y_MIN <= cy <= GATE_Y_MAX
    if direction == "NORTH":
        return py > GATE_Y_NORTH >= cy and GATE_X_MIN <= cx <= GATE_X_MAX
    if direction == "SOUTH":
        return py < GATE_Y_SOUTH <= cy and GATE_X_MIN <= cx <= GATE_X_MAX
    return False


def movement_for(entry: str | None, exit_direction: str | None) -> str | None:
    if entry is None or exit_direction is None:
        return None
    return MOVEMENT_MAP.get((entry, exit_direction))


def color_for_signal(signal: str) -> tuple[int, int, int]:
    if signal == "GREEN":
        return (0, 255, 0)
    if signal == "YELLOW":
        return (0, 255, 255)
    return (0, 0, 255)


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n"
            "Put the working yolo11n.pt inside the models folder."
        )

    if not VIDEO_PATH.exists():
        raise FileNotFoundError(
            f"Video not found: {VIDEO_PATH}\n"
            "Put traffic.mp4 inside the video folder."
        )

    cap = cv2.VideoCapture(SOURCE)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {SOURCE}")

    model = YOLO(str(MODEL_PATH))
    tracks: dict[int, Track] = {}

    # Completed movement totals for historical display.
    completed_counts = {
        d: {m: 0 for m in ("LEFT", "STRAIGHT", "RIGHT")}
        for d in ("NORTH", "EAST", "SOUTH", "WEST")
    }

    current_phase: str | None = None
    signal = "RED"
    remaining = 0
    last_phase: str | None = None
    last_tick = time.monotonic()
    frame_no = 0

    # Phase is based on LIVE approach queues so the controller reacts
    # before vehicles finish their whole trip.
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_no += 1
        h, w = frame.shape[:2]

        results = model.track(
            frame,
            persist=True,
            tracker=TRACKER,
            imgsz=IMAGE_SIZE,
            conf=CONFIDENCE,
            classes=VEHICLE_CLASS_IDS,
            verbose=False,
        )

        boxes = results[0].boxes
        active_ids: set[int] = set()
        live_vehicle_centers: dict[int, tuple[int, int]] = {}

        if boxes is not None and len(boxes) > 0:
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i])
                cls_name = model.names[cls_id]
                if cls_name not in VEHICLE_NAMES:
                    continue
                if boxes.id is None:
                    continue

                track_id = int(boxes.id[i])
                active_ids.add(track_id)

                x1, y1, x2, y2 = map(int, boxes.xyxy[i])
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                live_vehicle_centers[track_id] = (cx, cy)

                tr = tracks.setdefault(track_id, Track())
                previous = tr.positions[-1] if tr.positions else None
                tr.positions.append((cx, cy))
                tr.last_seen_frame = frame_no

                # Entry detection.
                if previous is not None and tr.entry is None:
                    for d in ("WEST", "EAST", "NORTH", "SOUTH"):
                        if crossed_entry(previous, (cx, cy), d):
                            tr.entry = d
                            break

                # Exit/movement detection.
                if previous is not None and tr.entry is not None and tr.exit is None:
                    for d in ("WEST", "EAST", "NORTH", "SOUTH"):
                        if d == tr.entry:
                            continue
                        if crossed_exit(previous, (cx, cy), d):
                            tr.exit = d
                            tr.movement = movement_for(tr.entry, tr.exit)
                            if tr.movement and not tr.counted_movement:
                                completed_counts[tr.entry][tr.movement] += 1
                                tr.counted_movement = True
                            break

                # Draw every accepted detection so we can visually verify
                # whether YOLO is actually seeing the vehicles.
                label = f"{cls_name} ID:{track_id}"
                if tr.entry:
                    label += f" {tr.entry}"
                if tr.movement:
                    label += f"->{tr.movement}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    label,
                    (x1, max(18, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (0, 255, 255),
                    2,
                )
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

        # --------------------------------------------------------
        # LIVE APPROACH QUEUES
        # --------------------------------------------------------
        queue_counts = {d: 0 for d in APPROACH_ZONES}
        for track_id, center in live_vehicle_centers.items():
            cx, cy = center
            for d, rect in APPROACH_ZONES.items():
                if inside_rect(cx, cy, rect):
                    queue_counts[d] += 1
                    break

        total_live = sum(queue_counts.values())

        # --------------------------------------------------------
        # SMART SIGNAL PHASE USING LIVE QUEUE DEMAND
        # --------------------------------------------------------
        ns_demand = queue_counts["NORTH"] + queue_counts["SOUTH"]
        ew_demand = queue_counts["EAST"] + queue_counts["WEST"]
        demands = {"NS": ns_demand, "EW": ew_demand}

        now = time.monotonic()
        elapsed = int(now - last_tick)
        if elapsed > 0:
            remaining -= elapsed
            last_tick = now

        if current_phase is None:
            candidates = {p: c for p, c in demands.items() if c > 0}
            if last_phase in candidates and len(candidates) > 1:
                candidates.pop(last_phase, None)

            if candidates:
                current_phase = max(candidates, key=candidates.get)
                demand = demands[current_phase]
                remaining = max(MIN_GREEN, min(MAX_GREEN, demand * SECONDS_PER_VEHICLE))
                signal = "GREEN"
                last_tick = now
            else:
                signal = "RED"
                remaining = 0
        elif remaining <= 0 and signal == "GREEN":
            signal = "YELLOW"
            remaining = YELLOW_SECONDS
            last_tick = now
        elif remaining <= 0 and signal == "YELLOW":
            last_phase = current_phase
            current_phase = None
            signal = "RED"
            remaining = 0
            last_tick = now

        # --------------------------------------------------------
        # VISUAL CALIBRATION
        # --------------------------------------------------------
        # Scale the calibration coordinates if a different frame size
        # is encountered.
        sx = w / 1920.0
        sy = h / 1080.0

        def X(v: int) -> int:
            return int(v * sx)

        def Y(v: int) -> int:
            return int(v * sy)

        # Gate lines.
        cv2.line(frame, (X(GATE_X_WEST), Y(GATE_Y_MIN)), (X(GATE_X_WEST), Y(GATE_Y_MAX)), (255, 0, 255), 2)
        cv2.line(frame, (X(GATE_X_EAST), Y(GATE_Y_MIN)), (X(GATE_X_EAST), Y(GATE_Y_MAX)), (255, 0, 255), 2)
        cv2.line(frame, (X(GATE_X_MIN), Y(GATE_Y_NORTH)), (X(GATE_X_MAX), Y(GATE_Y_NORTH)), (255, 0, 255), 2)
        cv2.line(frame, (X(GATE_X_MIN), Y(GATE_Y_SOUTH)), (X(GATE_X_MAX), Y(GATE_Y_SOUTH)), (255, 0, 255), 2)

        # --------------------------------------------------------
        # DASHBOARD
        # --------------------------------------------------------
        cv2.putText(frame, "AI SMART TRAFFIC CONTROLLER", (25, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        cv2.putText(frame, f"LIVE DETECTED VEHICLES: {len(live_vehicle_centers)}",
                    (25, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

        y = 105
        for d in ("NORTH", "EAST", "SOUTH", "WEST"):
            txt = f"{d}: {queue_counts[d]} vehicles"
            cv2.putText(frame, txt, (25, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            y += 28

        # Movement history.
        cv2.putText(frame, "COMPLETED MOVEMENTS", (25, y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        y += 38
        for d in ("NORTH", "EAST", "SOUTH", "WEST"):
            c = completed_counts[d]
            txt = f"{d}: L {c['LEFT']}  S {c['STRAIGHT']}  R {c['RIGHT']}"
            cv2.putText(frame, txt, (25, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 220, 0), 2)
            y += 25

        phase_label = current_phase if current_phase else "NONE"
        status = f"PHASE: {phase_label} | {signal} | {remaining}s"
        cv2.putText(frame, status, (25, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.72,
                    color_for_signal(signal), 2)

        cv2.imshow("AI Smart Traffic Controller", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()