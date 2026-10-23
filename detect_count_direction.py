
from pathlib import Path
import argparse
from collections import Counter

import cv2
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
MODEL_PATH = (
    ROOT / "runs" / "vehicle" / "roboflow_v2"
    / "weights" / "best.pt"
)

CLASSES = ("bus", "car", "motorbike", "truck")


def main():
    parser = argparse.ArgumentParser(
        description="Vehicle tracking with robust, bidirectional line counting"
    )
    parser.add_argument("--source", required=True, help="Input video path")
    parser.add_argument(
        "--line", type=float, default=0.60,
        help="Line height as a fraction of video height (default: 0.60)"
    )
    parser.add_argument(
        "--band", type=float, default=0.015,
        help="Hysteresis half-width as a fraction of frame height (default: 0.015)"
    )
    parser.add_argument(
        "--conf", type=float, default=0.35,
        help="Detection confidence threshold"
    )
    parser.add_argument(
        "--no-show", action="store_true",
        help="Save output without displaying the video window"
    )
    args = parser.parse_args()

    if not 0 < args.line < 1:
        parser.error("--line must be between 0 and 1")
    if not 0 <= args.band < min(args.line, 1 - args.line):
        parser.error("--band must be non-negative and fit around the line")
    if not 0 < args.conf < 1:
        parser.error("--conf must be between 0 and 1")

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    source_path = Path(args.source)
    if not source_path.is_file():
        raise FileNotFoundError(f"Video not found: {source_path}")

    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {source_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError("Video has invalid dimensions")

    if fps <= 0:
        fps = 25.0

    line_y = int(height * args.line)
    band_px = max(1, int(height * args.band))
    upper_y = max(0, line_y - band_px)
    lower_y = min(height - 1, line_y + band_px)

    output_path = ROOT / f"{source_path.stem}_direction_counted.mp4"
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot create output video: {output_path}")

    device = 0 if torch.cuda.is_available() else "cpu"
    print("Model:", MODEL_PATH)
    print("Device:", torch.cuda.get_device_name(0) if device == 0 else "CPU")
    print("Video:", source_path)
    print("Counting line:", line_y, "| Hysteresis half-width:", band_px, "px")
    print("Output:", output_path)
    print("Press Q in the preview window to stop early.")

    model = YOLO(str(MODEL_PATH))

    # track_id -> last stable side: -1 = A (above), +1 = B (below)
    last_stable_side = {}

    # Each ID can be counted once in A->B and once in B->A.
    counted_track_directions = set()

    # Direction totals and class-specific crossing totals.
    direction_totals = Counter()
    class_direction_totals = Counter()

    frames_processed = 0
    seen_track_classes = {}

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frames_processed += 1

            result = model.track(
                source=frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=args.conf,
                imgsz=640,
                device=device,
                verbose=False,
            )[0]

            # Draw the line and its deadband (hysteresis zone).
            cv2.line(frame, (0, line_y), (width, line_y), (0, 255, 255), 2)
            cv2.line(frame, (0, upper_y), (width, upper_y), (255, 200, 0), 1)
            cv2.line(frame, (0, lower_y), (width, lower_y), (255, 200, 0), 1)

            boxes = result.boxes

            if boxes is not None and boxes.id is not None:
                coords = boxes.xyxy.cpu().numpy()
                ids = boxes.id.int().cpu().tolist()
                class_ids = boxes.cls.int().cpu().tolist()
                confs = boxes.conf.cpu().tolist()

                for box, track_id, class_id, confidence in zip(
                    coords, ids, class_ids, confs
                ):
                    x1, y1, x2, y2 = map(int, box)
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2

                    class_name = result.names[class_id]
                    seen_track_classes[track_id] = class_name

                    # Hysteresis: don't update side while inside the band.
                    if center_y < upper_y:
                        current_stable_side = -1
                    elif center_y > lower_y:
                        current_stable_side = 1
                    else:
                        current_stable_side = None

                    if current_stable_side is not None:
                        previous_side = last_stable_side.get(track_id)

                        if previous_side is not None and previous_side != current_stable_side:
                            if previous_side == -1 and current_stable_side == 1:
                                direction = "A->B"
                            else:
                                direction = "B->A"

                            key = (track_id, direction)

                            # Suppress duplicate counts for the same ID/direction.
                            if key not in counted_track_directions:
                                counted_track_directions.add(key)
                                direction_totals[direction] += 1
                                class_direction_totals[(class_name, direction)] += 1

                        last_stable_side[track_id] = current_stable_side

                    color = (0, 220, 0) if any(
                        tid == track_id for tid, _ in counted_track_directions
                    ) else (255, 160, 0)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = f"{class_name} ID:{track_id} {confidence:.2f}"
                    cv2.putText(
                        frame, label, (x1, max(20, y1 - 7)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                    )
                    cv2.circle(
                        frame, (center_x, center_y), 4, (0, 0, 255), -1
                    )

            # Display current cumulative counts.
            panel_bottom = 160
            cv2.rectangle(frame, (8, 8), (390, panel_bottom), (25, 25, 25), -1)
            cv2.putText(
                frame, f"A -> B: {direction_totals['A->B']}",
                (18, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2
            )
            cv2.putText(
                frame, f"B -> A: {direction_totals['B->A']}",
                (18, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2
            )

            y_text = 88
            for name in CLASSES:
                a_to_b = class_direction_totals[(name, "A->B")]
                b_to_a = class_direction_totals[(name, "B->A")]
                cv2.putText(
                    frame, f"{name}: A->B {a_to_b} | B->A {b_to_a}",
                    (18, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                    (255, 255, 255), 1
                )
                y_text += 17

            writer.write(frame)

            if not args.no_show:
                cv2.imshow("Vehicle Direction Counting", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Stopped early by user.")
                    break

    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()
    print("\n========== TRACKING SUMMARY ==========")

    seen_class_counts = Counter(seen_track_classes.values())

    for name in CLASSES:
        print(f"{name}: {seen_class_counts[name]} unique tracked IDs")

    print("\n========== COUNTING SUMMARY ==========")
    print("Frames processed:", frames_processed)
    print("A -> B (above to below):", direction_totals["A->B"])
    print("B -> A (below to above):", direction_totals["B->A"])
    print("\nPer-class crossing totals:")
    for name in CLASSES:
        print(
            f"{name}: A->B={class_direction_totals[(name, 'A->B')]}, "
            f"B->A={class_direction_totals[(name, 'B->A')]}"
        )

    print("\nSaved annotated video:", output_path)


if __name__ == "__main__":
    main()
