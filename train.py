
from pathlib import Path
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DATA_YAML = ROOT / "dataset" / "data.yaml"

def main():
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Dataset config not found: {DATA_YAML}")

    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        device = 0
    else:
        print("WARNING: GPU unavailable; using CPU.")
        device = "cpu"

    model = YOLO("yolov8n.pt")

    model.train(
        data=str(DATA_YAML),
        epochs=50,
        imgsz=640,
        batch=2,  # Conservative setting for 4 GB VRAM
        device=device,
        workers=0,
        patience=10,
        project=str(ROOT / "runs" / "vehicle"),
        name="roboflow_v2",
        exist_ok=True,
        plots=True,
        save=True,
    )

    print("Training finished.")
    print("Best model:", ROOT / "runs" / "vehicle" /
          "roboflow_v2" / "weights" / "best.pt")

if __name__ == "__main__":
    main()
