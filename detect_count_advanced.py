"""
Advanced Vehicle Detection, ByteTrack Tracking, and Bidirectional Counting
Features:
- Road-plane tire contact patch tracking ((x1+x2)//2, y2)
- Hysteresis deadband to eliminate threshold oscillation
- Temporal class smoothing (majority voting) to prevent flickering labels
- Tuned ByteTrack configuration for traffic occlusion handling
- Motion trajectory trails
- Semi-transparent modern HUD analytics dashboard
- Stale track memory pruning for 24/7 long-running streams
"""

from collections import Counter, defaultdict, deque
from pathlib import Path
import argparse
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = (
    ROOT / "weights" / "vehicle_yolov8s_best.pt"
    if (ROOT / "weights" / "vehicle_yolov8s_best.pt").exists()
    else ROOT / "runs" / "vehicle" / "roboflow_v2_enhanced" / "weights" / "best.pt"
    if (ROOT / "runs" / "vehicle" / "roboflow_v2_enhanced" / "weights" / "best.pt").exists()
    else ROOT / "runs" / "vehicle" / "roboflow_v2" / "weights" / "best.pt"
)
DEFAULT_TRACKER = (
    ROOT / "configs" / "tracker_traffic.yaml"
    if (ROOT / "configs" / "tracker_traffic.yaml").exists()
    else ROOT / "tracker_traffic.yaml"
)

CLASS_NAMES = ("bus", "car", "motorbike", "truck")
CLASS_COLORS = {
    "bus": (255, 128, 0),        # Orange-blue
    "car": (0, 220, 255),        # Cyan-yellow
    "motorbike": (255, 0, 200),  # Magenta
    "truck": (0, 255, 128),      # Green-mint
}


def draw_hud(frame, counts_ab, counts_ba, class_counts, active_vehicles, fps, total_counted):
    """Draw a semi-transparent HUD overlay with traffic analytics."""
    h, w = frame.shape[:2]
    panel_w, panel_h = 360, 225
    x0, y0 = 15, 15

    # Create semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(
        overlay, (x0, y0), (x0 + panel_w, y0 + panel_h),
        (20, 24, 30), -1
    )
    cv2.rectangle(
        overlay, (x0, y0), (x0 + panel_w, y0 + panel_h),
        (80, 100, 120), 1
    )
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Title & Live Stats
    cv2.putText(
        frame, "VEHICLE AI ANALYTICS", (x0 + 12, y0 + 26),
        cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 230, 255), 2, cv2.LINE_AA
    )
    cv2.putText(
        frame, f"FPS: {fps:.1f}  |  Active: {active_vehicles}",
        (x0 + 12, y0 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
        (180, 200, 210), 1, cv2.LINE_AA
    )

    # Direction Flow
    cv2.line(frame, (x0 + 10, y0 + 60), (x0 + panel_w - 10, y0 + 60), (60, 75, 90), 1)
    cv2.putText(
        frame, f"Inbound (A->B): {counts_ab}",
        (x0 + 12, y0 + 82), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
        (50, 255, 150), 2, cv2.LINE_AA
    )
    cv2.putText(
        frame, f"Outbound (B->A): {counts_ba}",
        (x0 + 12, y0 + 106), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
        (80, 180, 255), 2, cv2.LINE_AA
    )

    # Per-class counts
    cv2.line(frame, (x0 + 10, y0 + 116), (x0 + panel_w - 10, y0 + 116), (60, 75, 90), 1)
    y_cls = y0 + 138
    for cname in CLASS_NAMES:
        col = CLASS_COLORS.get(cname, (200, 200, 200))
        cv2.circle(frame, (x0 + 18, y_cls - 4), 4, col, -1)
        cnt = class_counts[cname]
        cv2.putText(
            frame, f"{cname.capitalize()}: {cnt}",
            (x0 + 30, y_cls), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
            (240, 240, 240), 1, cv2.LINE_AA
        )
        y_cls += 20

    # Total Footer
    cv2.putText(
        frame, f"Total Crossings: {total_counted}",
        (x0 + 190, y0 + 208), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
        (0, 255, 255), 2, cv2.LINE_AA
    )


def main():
    parser = argparse.ArgumentParser(
        description="High-Accuracy Vehicle Detection, Tracking, and Directional Counting"
    )
    parser.add_argument("--source", required=True, help="Path to input video or webcam index (0)")
    parser.add_argument(
        "--weights", type=str, default=str(DEFAULT_WEIGHTS),
        help="Path to trained YOLO weights (*.pt)"
    )
    parser.add_argument(
        "--tracker", type=str, default=str(DEFAULT_TRACKER),
        help="Path to tracker YAML configuration"
    )
    parser.add_argument(
        "--line", type=float, default=0.60,
        help="Virtual counting line as a fraction of frame height (0.0 - 1.0)"
    )
    parser.add_argument(
        "--band", type=float, default=0.02,
        help="Hysteresis band half-width as fraction of frame height (default 0.02)"
    )
    parser.add_argument(
        "--imgsz", type=int, default=960,
        help="Inference image resolution (default 960 for 1080p source)"
    )
    parser.add_argument(
        "--conf", type=float, default=0.25,
        help="Detection confidence threshold (default 0.25)"
    )
    parser.add_argument(
        "--trails", action="store_true", default=True,
        help="Draw smooth trajectory motion trails behind vehicles"
    )
    parser.add_argument(
        "--no-trails", dest="trails", action="store_false",
        help="Disable trajectory trails"
    )
    parser.add_argument(
        "--no-show", action="store_true",
        help="Run headless without interactive preview window"
    )
    parser.add_argument(
        "--output", type=str, default="",
        help="Custom output video path (defaults to <source_stem>_advanced_counted.mp4)"
    )
    args = parser.parse_args()

    weights_path = Path(args.weights)
    if not weights_path.is_file():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    # Open video source (file or webcam)
    source_is_cam = args.source.isdigit()
    source_target = int(args.source) if source_is_cam else args.source
    cap = cv2.VideoCapture(source_target)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {args.source}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    line_y = int(height * args.line)
    band_px = max(2, int(height * args.band))
    upper_y = max(0, line_y - band_px)
    lower_y = min(height - 1, line_y + band_px)

    if args.output:
        out_path = Path(args.output)
    else:
        stem = "webcam" if source_is_cam else Path(args.source).stem
        out_path = ROOT / f"{stem}_advanced_counted.mp4"

    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps, (width, height)
    )

    device = 0 if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if device == 0 else "CPU"
    print("=" * 65)
    print(" ADVANCED VEHICLE DETECTION & BIDIRECTIONAL TRACKER")
    print("=" * 65)
    print(f"Device:            {device_name}")
    print(f"Model Weights:     {weights_path.name}")
    print(f"Tracker Config:    {Path(args.tracker).name}")
    print(f"Resolution:        {args.imgsz}px")
    print(f"Counting Line:     y = {line_y} (band: [{upper_y}, {lower_y}])")
    print(f"Output Video:      {out_path.name}")
    print("=" * 65)

    model = YOLO(str(weights_path))

    # State tracking
    trajectories = defaultdict(lambda: deque(maxlen=30))
    class_history = defaultdict(lambda: deque(maxlen=20))
    last_stable_side = {}       # track_id -> -1 (above) or +1 (below)
    counted_directions = set()  # set of (track_id, "A->B" | "B->A")
    last_seen_frame = {}

    direction_counts = Counter()
    class_crossing_counts = Counter()
    total_frames = 0
    t_start = time.time()
    rolling_fps = fps

    try:
        while True:
            t_frame_start = time.time()
            ok, frame = cap.read()
            if not ok:
                break

            total_frames += 1

            # Run tracking with tuned ByteTrack config
            results = model.track(
                source=frame,
                persist=True,
                tracker=str(args.tracker),
                conf=args.conf,
                imgsz=args.imgsz,
                device=device,
                verbose=False
            )
            result = results[0]

            # Draw counting line and hysteresis zone
            cv2.line(frame, (0, line_y), (width, line_y), (0, 255, 255), 2, cv2.LINE_AA)
            cv2.line(frame, (0, upper_y), (width, upper_y), (255, 180, 0), 1, cv2.LINE_AA)
            cv2.line(frame, (0, lower_y), (width, lower_y), (255, 180, 0), 1, cv2.LINE_AA)

            boxes = result.boxes
            active_count = 0

            if boxes is not None and boxes.id is not None:
                coords = boxes.xyxy.cpu().numpy()
                tids = boxes.id.int().cpu().tolist()
                cids = boxes.cls.int().cpu().tolist()
                confs = boxes.conf.cpu().tolist()
                active_count = len(tids)

                for box, tid, cid, conf in zip(coords, tids, cids, confs):
                    x1, y1, x2, y2 = map(int, box)
                    
                    # 1. GROUND CONTACT PATCH: bottom-center (where tires meet pavement)
                    contact_x = (x1 + x2) // 2
                    contact_y = y2

                    # 2. TEMPORAL CLASS SMOOTHING: majority voting
                    detected_name = result.names[cid]
                    class_history[tid].append(detected_name)
                    stable_name = Counter(class_history[tid]).most_common(1)[0][0]

                    # 3. Update trajectory
                    trajectories[tid].append((contact_x, contact_y))
                    last_seen_frame[tid] = total_frames

                    # 4. Hysteresis State Transition
                    if contact_y < upper_y:
                        current_side = -1  # Above band (Zone A)
                    elif contact_y > lower_y:
                        current_side = 1   # Below band (Zone B)
                    else:
                        current_side = None  # Inside deadband (retain prior state)

                    if current_side is not None:
                        prev_side = last_stable_side.get(tid)
                        if prev_side is not None and prev_side != current_side:
                            direction = "A->B" if (prev_side == -1 and current_side == 1) else "B->A"
                            count_key = (tid, direction)

                            if count_key not in counted_directions:
                                counted_directions.add(count_key)
                                direction_counts[direction] += 1
                                class_crossing_counts[stable_name] += 1

                        last_stable_side[tid] = current_side

                    # 5. Visual Styling
                    is_counted = any(t == tid for t, _ in counted_directions)
                    box_color = (0, 255, 100) if is_counted else CLASS_COLORS.get(stable_name, (0, 200, 255))

                    # Bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                    
                    # Ground contact indicator
                    cv2.circle(frame, (contact_x, contact_y), 4, (0, 0, 255), -1, cv2.LINE_AA)

                    # Label pill
                    badge_text = f"{stable_name} #{tid} {conf:.2f}"
                    (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                    cv2.rectangle(
                        frame, (x1, max(0, y1 - th - 8)),
                        (x1 + tw + 8, y1), box_color, -1
                    )
                    cv2.putText(
                        frame, badge_text, (x1 + 4, max(th + 2, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (10, 15, 20), 1, cv2.LINE_AA
                    )

                    # 6. Motion Trajectory Trail
                    if args.trails and len(trajectories[tid]) > 1:
                        pts = np.array(list(trajectories[tid]), dtype=np.int32).reshape((-1, 1, 2))
                        cv2.polylines(frame, [pts], isClosed=False, color=box_color, thickness=2, lineType=cv2.LINE_AA)

            # 7. Memory management: prune stale tracks older than 120 frames
            if total_frames % 60 == 0:
                stale_ids = [
                    tid for tid, f in last_seen_frame.items()
                    if total_frames - f > 120
                ]
                for tid in stale_ids:
                    trajectories.pop(tid, None)
                    class_history.pop(tid, None)
                    last_stable_side.pop(tid, None)
                    last_seen_frame.pop(tid, None)

            # Compute rolling FPS
            frame_time = time.time() - t_frame_start
            if frame_time > 0:
                rolling_fps = 0.9 * rolling_fps + 0.1 * (1.0 / frame_time)

            # Render Analytics Dashboard HUD
            total_crossings = direction_counts["A->B"] + direction_counts["B->A"]
            draw_hud(
                frame=frame,
                counts_ab=direction_counts["A->B"],
                counts_ba=direction_counts["B->A"],
                class_counts=class_crossing_counts,
                active_vehicles=active_count,
                fps=rolling_fps,
                total_counted=total_crossings
            )

            writer.write(frame)

            if not args.no_show:
                cv2.imshow("Vehicle AI - Advanced Tracker", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("\nProcessing stopped early by user.")
                    break

    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

    elapsed = time.time() - t_start
    avg_fps = total_frames / elapsed if elapsed > 0 else 0

    print("\n" + "=" * 65)
    print("              PROCESSING SUMMARY")
    print("=" * 65)
    print(f"Total Frames Processed:   {total_frames}")
    print(f"Elapsed Time:             {elapsed:.2f} seconds ({avg_fps:.1f} FPS)")
    print(f"Inbound Flow (A -> B):    {direction_counts['A->B']}")
    print(f"Outbound Flow (B -> A):   {direction_counts['B->A']}")
    print(f"Total Line Crossings:     {len(counted_directions)}")
    print("-" * 65)
    print("Per-Class Vehicle Crossings:")
    for cname in CLASS_NAMES:
        print(f"  • {cname.capitalize():<12}: {class_crossing_counts[cname]}")
    print("=" * 65)
    print(f"Annotated output saved to: {out_path.resolve()}\n")


if __name__ == "__main__":
    main()
