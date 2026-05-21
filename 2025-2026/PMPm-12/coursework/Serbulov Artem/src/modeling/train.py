import os
from pathlib import Path

from loguru import logger
import typer
from ultralytics import YOLO
import yaml

from ultralytics import settings

# --- MLflow Setup ---
# Force enable MLflow logging in Ultralytics
settings.update({"mlflow": True})

# Import paths from the project config
from src.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


def create_yolo_config(data_path: Path, yaml_path: Path) -> Path:
    """
    Generate the dataset.yaml file required by YOLOv8.
    """
    class_names = [
        "pedestrian", "people", "bicycle", "car", "van",
        "truck", "tricycle", "awning-tricycle", "bus", "motor",
    ]

    config = {
        "path": str(data_path.absolute()), 
        "train": "train/images", 
        "val": "val/images", 
        "test": "test_dev/images", 
        "names": {i: name for i, name in enumerate(class_names)},
    }

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, sort_keys=False)

    logger.success(f"Config created: {yaml_path}")
    return yaml_path


@app.command()
def main(
    data_dir: Path = typer.Option(
        PROCESSED_DATA_DIR / "visdrone_yolo",
        help="Path to YOLO dataset root (expects train/val/test_dev splits).",
    ),
    epochs: int = typer.Option(1, help="Number of epochs"),
    patience: int = typer.Option(10, help="Early stopping number"),

    batch_size: int = typer.Option(16, help="Batch size"),

    img_size: int = typer.Option(640, help="Image size (Optimized for Small Objects)"),
    device: str = typer.Option("0", help="Device to use: 'cpu', '0', '0,1' etc."),
    
    yaml_arch: str = typer.Option("yolov8s-p2.yaml", help="Architecture with P2 Head"),
    pretrained_weights: str = typer.Option("yolov8s.pt", help="Pretrained weights"),

    experiment_name: str = typer.Option("EEML_2026_Detection", help="Experiment Name"),
    run_name: str = typer.Option("run_02", help="Run Name"),
):
    logger.info(f"Starting Training: Arch={yaml_arch}, Weights={pretrained_weights}...")

    os.environ["MLFLOW_EXPERIMENT_NAME"] = experiment_name
    os.environ["MLFLOW_RUN"] = run_name

    # 1. Create YAML config
    yaml_path = data_dir / "dataset.yaml"
    create_yolo_config(data_dir, yaml_path)

    # 2. Load model (THE MAGIC TRICK)
    model = YOLO(yaml_arch).load(pretrained_weights)
    logger.info("Model loaded successfully with P2 head and pretrained weights.")

    # 3. Run training
    # Note: project and name arguments determine where local weights are saved.
    # By aligning them with MLflow names, we keep local and remote tracking synchronized.
    local_project_dir = str(MODELS_DIR / experiment_name)

    model.train(
        data=str(yaml_path),
        epochs=epochs,
        patience=patience,
        imgsz=img_size,
        batch=batch_size,
        device=device,

        fraction=0.2,
        cache='ram',

        mosaic=0.5,
        mixup=0.0,
        copy_paste=0.0,

        optimizer="AdamW",
        lr0=0.001,
        warmup_epochs=2.0,
        close_mosaic=5,

        project=local_project_dir,
        name=run_name,
        exist_ok=True, 
        verbose=True,
        workers=32
    )

    logger.success(
        f"Training complete! Best model saved to: {MODELS_DIR}/yolov8s_p2_wiou/weights/best.pt"
    )

    # 4. Validation
    logger.info("Running validation...")
    metrics = model.val(data=str(yaml_path), imgsz=img_size, batch=batch_size, device=device)
    logger.info(f"mAP50: {metrics.box.map50}")


if __name__ == "__main__":
    app()