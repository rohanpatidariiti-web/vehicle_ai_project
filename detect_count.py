
from pathlib import Path
import argparse
from collections import Counter

import cv2
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "runs" / "vehicle" / "roboflow_v2" / "weights" / "best.pt"


def main():
    parser = argparse.ArgumentParser(
        description="Vehicle detection, ByteTrack tracking, and line-crossing counting"
    )
    parser.add_argument(
        "--source", required=True,
        help="Path to a traffic video, e.g. C:\\Videos\\traffic.mp4"
    )
    parser.add_argument(
        "--line", type=float, default=0.60,
        help="Horizontal counting-line position as a fraction of frame height (default 0.60)"
    )
    parser.add_argument(
        "--conf", type=float, default=0.25,
        help="Detection confidence threshold (default 0.25)"
    )
    parser.add_argument(
        "--no-show", action="store_true",
        help="Save output without displaying a video window"
    )
    args = parser.parse_args()

    if not 0 < args.line < 1:
        raise ValueError("--line must be between 0 and 1")

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")

    video_path = Path(args.source)
    if not video_path.is_file():
        raise FileNotFoundError(f"Input video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError("Video has invalid dimensions")

    if fps <= 0:
        fps = 25.0

    output_path = ROOT / f"{video_path.stem}_counted.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(output_path), fourcc, fps, (width, height)
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create output video: {output_path}")

    device = 0 if torch.cuda.is_available() else "cpu"
    print("Model:", MODEL_PATH)
    print("Device:", torch.cuda.get_device_name(0) if device == 0 else "CPU")
    print("Input:", video_path)
    print("Output:", output_path)
    print("Press Q in the video window to stop early.")

    model = YOLO(str(MODEL_PATH))

    # Keep the last side of the line for each active track ID.
    previous_side = {}
    counted_ids = set()
    class_counts = Counter()
    total_frames = 0
    line_y = int(height * args.line)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            total_frames += 1

            result = model.track(
                source=frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=args.conf,
                imgsz=640,
                device=device,
                verbose=False
            )[0]

            # Draw the counting line.
            cv2.line(
                frame, (0, line_y), (width, line_y),
                (0, 255, 255), 2
            )
            cv2.putText(
                frame, "COUNTING LINE",
                (10, max(25, line_y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 255), 2
            )

            boxes = result.boxes
            if boxes is not None and boxes.id is not None:
                xyxy = boxes.xyxy.cpu().numpy()
                track_ids = boxes.id.int().cpu().tolist()
                class_ids = boxes.cls.int().cpu().tolist()
                confidences = boxes.conf.cpu().tolist()

                for box, track_id, class_id, confidence in zip(
                    xyxy, track_ids, class_ids, confidences
                ):
                    x1, y1, x2, y2 = map(int, box)
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2

                    class_name = result.names[class_id]
                    side = -1 if center_y < line_y else 1

                    # Count an ID once, on its first observed line crossing.
                    if track_id in previous_side:
                        crossed = previous_side[track_id] != side
                        if crossed and track_id not in counted_ids:
                            counted_ids.add(track_id)
                            class_counts[class_name] += 1
                    previous_side[track_id] = side

                    color = (0, 255, 0) if track_id in counted_ids else (255, 160, 0)
                    cv2.rectangle(
                        frame, (x1, y1), (x2, y2), color, 2
                    )
                    label = f"{class_name} ID:{track_id} {confidence:.2f}"
                    cv2.putText(
                        frame, label, (x1, max(20, y1 - 7)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                    )
                    cv2.circle(
                        frame, (center_x, center_y), 4, (0, 0, 255), -1
                    )

            # Display cumulative line-crossing counts.
            cv2.rectangle(frame, (8, 8), (290, 145), (25, 25, 25), -1)
            cv2.putText(
                frame, f"Total counted: {len(counted_ids)}",
                (18, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2
            )

            y_text = 62
            for name in ("bus", "car", "motorbike", "truck"):
                cv2.putText(
                    frame, f"{name}: {class_counts[name]}",
                    (18, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1
                )
                y_text += 20

            writer.write(frame)

            if not args.no_show:
                cv2.imshow("Vehicle Tracking and Counting", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Stopped by user.")
                    break

    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

    print("\n--- Counting summary ---")
    print("Frames processed:", total_frames)
    print("Total unique track IDs counted:", len(counted_ids))
    for name in ("bus", "car", "motorbike", "truck"):
        print(f"{name}: {class_counts[name]}")
    print("Saved annotated video:", output_path)


if __name__ == "__main__":
    main()
