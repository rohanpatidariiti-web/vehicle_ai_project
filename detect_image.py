
from pathlib import Path
import argparse
import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "runs" / "vehicle" / "roboflow_v2" / "weights" / "best.pt"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Path to JPG, JPEG or PNG")
    args = parser.parse_args()

    if not MODEL.exists():
        raise FileNotFoundError(f"Model not found: {MODEL}")

    image_path = Path(args.source)
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    model = YOLO(str(MODEL))
    results = model.predict(source=image, conf=0.25, save=False, verbose=False)
    result = results[0]

    output = result.plot()
    output_path = ROOT / f"{image_path.stem}_detected.jpg"
    if not cv2.imwrite(str(output_path), output):
        raise RuntimeError("Could not save output image")

    print("Detections by class:")
    for class_id, count in result.names.items():
        n = int((result.boxes.cls == class_id).sum()) if result.boxes else 0
        print(f"  {count}: {n}")

    print(f"Total vehicles detected: {len(result.boxes)}")
    print(f"Saved image: {output_path}")

if __name__ == "__main__":
    main()
