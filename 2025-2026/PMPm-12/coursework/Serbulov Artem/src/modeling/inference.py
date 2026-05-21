from pathlib import Path

import cv2
import typer
from loguru import logger
from ultralytics import YOLO

app = typer.Typer()


def run_inference_and_save(
    model_path: Path,
    image_path: Path,
    img_size: int,
    conf_threshold: float,
    output_dir: Path,
    model_name: str,
):
    """Helper function to run inference and save results."""
    logger.info(f"Loading {model_name}: {model_path}")
    model = YOLO(str(model_path))

    logger.info(f"Running inference with {model_name}...")
    results = model(
        str(image_path), imgsz=img_size, conf=conf_threshold, verbose=False
    )

    result = results[0]
    logger.info(f"  → {len(result.boxes)} objects found")

    # Plot bounding boxes
    annotated_frame = result.plot()

    # Save result
    output_path = (
        output_dir
        / f"detection_{model_name}_{image_path.stem}.jpg"
    )
    cv2.imwrite(str(output_path), annotated_frame)
    logger.success(f"  → Result saved: {output_path}")

    # Print detection info
    if len(result.boxes) > 0:
        logger.info(f"  → Detection details ({model_name}):")
        for i, box in enumerate(result.boxes):
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            logger.info(f"     Object {i+1}: Class={cls_id}, Confidence={conf:.2f}")
    
    return output_path


@app.command()
def main(
    image_path: Path = typer.Option(
        "data/processed/visdrone_yolo/val/images/0000001_02999_d_0000005.jpg",
        help="Path to the image for inference.",
    ),
    pretrained_model: Path = typer.Option(
        "yolov8s.pt",
        help="Path to the pretrained YOLO model.",
    ),
    trained_model: Path = typer.Option(
        "models/EEML_2026_Detection/run_02/weights/best.pt",
        help="Path to the trained YOLO model.",
    ),
    img_size: int = typer.Option(
        640,
        help="Image size for inference.",
    ),
    conf_threshold: float = typer.Option(
        0.5,
        help="Confidence threshold for detections.",
    ),
    output_dir: Path = typer.Option(
        "reports/figures",
        help="Directory to save the results.",
    ),
):
    """
    Run YOLO inference on a single image using both pretrained and trained models.
    Saves results as separate images with bounding boxes.
    """
    if not image_path.exists():
        logger.error(f"Image not found: {image_path}")
        raise typer.Exit(code=1)

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Processing: {image_path.name}\n")

    # Run inference with pretrained model
    if pretrained_model.exists():
        pretrained_output = run_inference_and_save(
            pretrained_model,
            image_path,
            img_size,
            conf_threshold,
            output_dir,
            "pretrained",
        )
    else:
        logger.warning(f"Pretrained model not found: {pretrained_model}")

    logger.info("")

    # Run inference with trained model
    if trained_model.exists():
        trained_output = run_inference_and_save(
            trained_model,
            image_path,
            img_size,
            conf_threshold,
            output_dir,
            "trained",
        )
    else:
        logger.warning(f"Trained model not found: {trained_model}")

    logger.info("\n✓ Inference complete!")


if __name__ == "__main__":
    app()
