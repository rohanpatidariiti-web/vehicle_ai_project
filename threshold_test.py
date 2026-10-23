from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent

MODEL_PATH = ROOT / "runs" / "vehicle" / "roboflow_v2" / "weights" / "best.pt"
DATA_YAML = ROOT / "dataset" / "data.yaml"

model = YOLO(str(MODEL_PATH))

thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

print("\nConfidence threshold experiment")
print("=" * 70)

for conf in thresholds:

    metrics = model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=640,
        batch=2,
        device=0,
        workers=0,
        conf=conf,
        plots=False,
        verbose=False,
    )

    print(
        f"conf={conf:.2f} | "
        f"Precision={metrics.box.mp:.4f} | "
        f"Recall={metrics.box.mr:.4f} | "
        f"mAP50={metrics.box.map50:.4f} | "
        f"mAP50-95={metrics.box.map:.4f}"
    )