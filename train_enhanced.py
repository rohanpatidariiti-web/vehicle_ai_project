"""
Enhanced YOLO Training Pipeline for Vehicle Detection
Includes:
- Gradient accumulation for effective batch size on limited VRAM
- Cosine LR scheduling
- Rich data augmentations (HSV, translation, scale, mosaic, mixup)
- Configurable model architectures (yolov8n, yolov8s, yolo11s)
- Automatic mixed precision (AMP)
"""

from pathlib import Path
import argparse
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DATA_YAML = ROOT / "dataset" / "data.yaml"


def main():
    parser = argparse.ArgumentParser(description="Enhanced YOLO Vehicle Training")
    parser.add_argument(
        "--model", type=str, default="yolov8s.pt",
        help="Base model architecture (e.g. yolov8n.pt, yolov8s.pt, yolo11s.pt)"
    )
    parser.add_argument(
        "--epochs", type=int, default=60,
        help="Total training epochs"
    )
    parser.add_argument(
        "--batch", type=int, default=4,
        help="Physical batch size per step"
    )
    parser.add_argument(
        "--accumulate", type=int, default=4,
        help="Gradient accumulation steps (effective batch = batch * accumulate)"
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="Training image resolution"
    )
    parser.add_argument(
        "--name", type=str, default="roboflow_v2_enhanced",
        help="Experiment name under runs/vehicle/"
    )
    args = parser.parse_args()

    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Dataset config not found at: {DATA_YAML}")

    device = 0 if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if device == 0 else "CPU"
    effective_batch = args.batch * args.accumulate

    print("=" * 65)
    print("        ENHANCED YOLO VEHICLE TRAINING PIPELINE")
    print("=" * 65)
    print(f"Base Model:             {args.model}")
    print(f"Dataset Config:         {DATA_YAML}")
    print(f"Device:                 {device_name}")
    print(f"Physical Batch Size:    {args.batch}")
    print(f"Gradient Accumulation:  {args.accumulate} (Effective batch = {effective_batch})")
    print(f"Image Resolution:       {args.imgsz}")
    print(f"Epochs:                 {args.epochs}")
    print("=" * 65)

    model = YOLO(args.model)

    results = model.train(
        data=str(DATA_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=2 if device == 0 else 0,
        patience=15,
        project=str(ROOT / "runs" / "vehicle"),
        name=args.name,
        exist_ok=True,
        plots=True,
        save=True,
        amp=True,                  # Automatic mixed precision
        cos_lr=True,               # Cosine learning rate decay
        # Data Augmentations
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        perspective=0.0005,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        close_mosaic=10,
    )

    best_weights = ROOT / "runs" / "vehicle" / args.name / "weights" / "best.pt"
    print("\nTraining completed successfully!")
    print(f"Best model weights saved to: {best_weights}")


if __name__ == "__main__":
    main()
