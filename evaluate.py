
from pathlib import Path
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent

MODEL_PATH = (
    ROOT / "runs" / "vehicle" / "roboflow_v2"
    / "weights" / "best.pt"
)
DATA_YAML = ROOT / "dataset" / "data.yaml"
OUTPUT_DIR = ROOT / "runs" / "vehicle" / "evaluation"


def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not DATA_YAML.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {DATA_YAML}")

    device = 0 if torch.cuda.is_available() else "cpu"

    print("Model:", MODEL_PATH)
    print("Dataset:", DATA_YAML)
    print("Evaluation split: test")
    print("Device:", torch.cuda.get_device_name(0)
          if device == 0 else "CPU")

    model = YOLO(str(MODEL_PATH))

    metrics = model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=640,
        batch=2,
        device=device,
        workers=0,
        plots=True,
        project=str(OUTPUT_DIR),
        name="roboflow_v2_test",
        exist_ok=True,
    )

    print("\n========== TEST RESULTS ==========")
    print(f"Precision:  {metrics.box.mp:.4f}")
    print(f"Recall:     {metrics.box.mr:.4f}")
    print(f"mAP@50:     {metrics.box.map50:.4f}")
    print(f"mAP@50-95:  {metrics.box.map:.4f}")

    print("\nPer-class mAP@50-95:")
    for class_id, ap in enumerate(metrics.box.maps):
        print(f"{metrics.names[class_id]}: {ap:.4f}")

    print("\nResults folder:")
    print(OUTPUT_DIR / "roboflow_v2_test")


if __name__ == "__main__":
    main()
