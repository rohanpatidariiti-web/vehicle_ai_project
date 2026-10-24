from pathlib import Path
import argparse
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent

DEFAULT_MODEL_PATH = (
    ROOT / "runs" / "vehicle" / "roboflow_v2_enhanced" / "weights" / "best.pt"
    if (ROOT / "runs" / "vehicle" / "roboflow_v2_enhanced" / "weights" / "best.pt").exists()
    else ROOT / "runs" / "vehicle" / "roboflow_v2" / "weights" / "best.pt"
)
DATA_YAML = ROOT / "dataset" / "data.yaml"
OUTPUT_DIR = ROOT / "runs" / "vehicle" / "evaluation"


def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLO Vehicle Model")
    parser.add_argument(
        "--weights", type=str, default=str(DEFAULT_MODEL_PATH),
        help="Path to model weights (*.pt)"
    )
    parser.add_argument(
        "--split", type=str, default="test",
        help="Data split to evaluate on (test, val, train)"
    )
    args = parser.parse_args()

    model_path = Path(args.weights)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")

    if not DATA_YAML.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {DATA_YAML}")

    device = 0 if torch.cuda.is_available() else "cpu"

    print("Model:", model_path)
    print("Dataset:", DATA_YAML)
    print(f"Evaluation split: {args.split}")
    print("Device:", torch.cuda.get_device_name(0) if device == 0 else "CPU")

    model = YOLO(str(model_path))

    exp_name = f"{model_path.parent.parent.name}_{args.split}"
    metrics = model.val(
        data=str(DATA_YAML),
        split=args.split,
        imgsz=640,
        batch=2,
        device=device,
        workers=0,
        plots=True,
        project=str(OUTPUT_DIR),
        name=exp_name,
        exist_ok=True,
    )

    print("\n========== EVALUATION RESULTS ==========")
    print(f"Precision:  {metrics.box.mp:.4f}")
    print(f"Recall:     {metrics.box.mr:.4f}")
    print(f"mAP@50:     {metrics.box.map50:.4f}")
    print(f"mAP@50-95:  {metrics.box.map:.4f}")

    print("\nPer-class mAP@50-95:")
    for class_id, ap in enumerate(metrics.box.maps):
        print(f"  {metrics.names[class_id]:<12}: {ap:.4f}")

    print("\nResults folder:")
    print(OUTPUT_DIR / exp_name)


if __name__ == "__main__":
    main()
