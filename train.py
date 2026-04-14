"""Training entrypoint for the AI-Based Tyre Health Monitoring System."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.classification import train_classifier
from src.dataset_loader import ensure_dataset_folders
from src.detection import train_yolo_detector
from src.lifespan import train_lifespan_model


BASE_DIR = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Train all tyre-health models.")
    parser.add_argument("--skip-yolo", action="store_true", help="Skip YOLO training if annotations are unavailable.")
    parser.add_argument("--yolo-epochs", type=int, default=50, help="YOLOv8 training epochs.")
    parser.add_argument("--classifier-epochs", type=int, default=30, help="Classifier training epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Classifier training batch size.")
    args = parser.parse_args()

    ensure_dataset_folders(BASE_DIR / "data")

    summary = {}

    summary["classifier"] = train_classifier(
        data_root=BASE_DIR / "data",
        epochs=args.classifier_epochs,
        batch_size=args.batch_size,
    )

    summary["lifespan"] = train_lifespan_model()

    if args.skip_yolo:
        summary["yolo"] = {"status": "skipped", "reason": "Requested by --skip-yolo"}
    else:
        try:
            summary["yolo"] = train_yolo_detector(epochs=args.yolo_epochs)
        except Exception as exc:
            summary["yolo"] = {
                "status": "skipped",
                "reason": str(exc),
                "hint": "Prepare YOLO labels under data/annotations/train and data/annotations/val.",
            }

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
