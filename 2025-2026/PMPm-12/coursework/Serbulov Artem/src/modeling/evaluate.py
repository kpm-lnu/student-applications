import typer
from loguru import logger
from pathlib import Path
from typing import Any
from ultralytics import YOLO

# Import paths from project config
from src.config import PROCESSED_DATA_DIR

app = typer.Typer()


def _evaluate_split(
    model: YOLO,
    data_config: Path,
    split: str,
    img_size: int,
    batch_size: int,
    device: str,
    save_json: bool,
) -> tuple[float, float, Path]:
    """Run Ultralytics evaluation for a single dataset split and return key metrics."""
    results: Any = model.val(
        data=str(data_config),
        split=split,
        imgsz=img_size,
        batch=batch_size,
        device=device,
        save_json=save_json,
        verbose=True,
    )
    return results.box.map50, results.box.map, Path(results.save_dir)

@app.command()
def main(
    model_path: Path = typer.Option(
        ..., 
        "--model", "-m",
        help="Path to the trained YOLO model weights (e.g., best.pt)."
    ),
    data_config: Path = typer.Option(
        PROCESSED_DATA_DIR / "visdrone_yolo_tiled" / "dataset.yaml",
        "--data", "-d",
        help="Path to the dataset.yaml file."
    ),
    img_size: int = typer.Option(
        1280, 
        "--imgsz", 
        help="Image size used for inference (should match training resolution)."
    ),
    batch_size: int = typer.Option(
        8, 
        "--batch", 
        help="Batch size for evaluation."
    ),
    device: str = typer.Option(
        "0", 
        "--device", 
        help="Device to run evaluation on (e.g., '0' or 'cpu')."
    ),
    save_json: bool = typer.Option(
        True, 
        help="Save results to a JSON file for further analysis."
    )
):
    """
    Evaluate a trained YOLO model on both VAL and TEST splits.
    This script provides final metrics (mAP50, mAP50-95) for research reporting.
    """
    
    if not model_path.exists():
        logger.error(f"Model weights not found at: {model_path}")
        raise typer.Exit(code=1)

    if not data_config.exists():
        logger.error(f"Dataset config not found at: {data_config}")
        raise typer.Exit(code=1)

    logger.info(f"🧐 Starting evaluation for model: {model_path.name}")
    logger.info(f"Using dataset config: {data_config}")

    # 1. Load the trained model
    # We use the specific weights obtained from the training phase
    model = YOLO(str(model_path))

    # 2. Evaluate on the validation split for model selection diagnostics.
    val_map50, val_map, val_save_dir = _evaluate_split(
        model=model,
        data_config=data_config,
        split="val",
        img_size=img_size,
        batch_size=batch_size,
        device=device,
        save_json=save_json,
    )

    logger.success("--- VALIDATION RESULTS (VAL SET) ---")
    logger.info(f"mAP@50:      {val_map50:.4f}")
    logger.info(f"mAP@50-95:   {val_map:.4f}")
    logger.info(f"VAL artifacts saved to: {val_save_dir}")

    # 3. Evaluate on the independent test split for final reporting.
    # Setting split='test' is crucial for academic integrity to ensure
    # we are not evaluating on data seen during hyperparameter tuning.
    test_map50, test_map, test_save_dir = _evaluate_split(
        model=model,
        data_config=data_config,
        split="test",
        img_size=img_size,
        batch_size=batch_size,
        device=device,
        save_json=save_json,
    )

    # 4. Log high-level metrics for the Research Abstract
    # These metrics (mAP50 and mAP50-95) are standard for VisDrone benchmarks.
    logger.success("--- FINAL EVALUATION RESULTS (TEST SET) ---")
    logger.info(f"mAP@50:      {test_map50:.4f}")
    logger.info(f"mAP@50-95:   {test_map:.4f}")
    logger.info(f"TEST artifacts saved to: {test_save_dir}")

    logger.success("--- SUMMARY (VAL vs TEST) ---")
    logger.info(f"VAL  mAP@50:    {val_map50:.4f} | mAP@50-95: {val_map:.4f}")
    logger.info(f"TEST mAP@50:    {test_map50:.4f} | mAP@50-95: {test_map:.4f}")
    
    # Extracting class-specific results can be useful for detailed analysis [cite: 623]
    logger.info("Class-wise mAP@50 performance logged in the results directory.")

    # 5. Cleanup/Final Notification
    logger.success("Evaluation complete. All plots and metrics saved to the directories above.")

if __name__ == "__main__":
    app()