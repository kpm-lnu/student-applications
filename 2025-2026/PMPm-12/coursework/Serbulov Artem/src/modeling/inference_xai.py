from pathlib import Path

import cv2
import numpy as np
import torch
import typer
from loguru import logger
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from ultralytics import YOLO

class YOLOv8Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # YOLOv8 returns a tuple: (predictions, intermediate_features)
        # We just return the first element (the tensor) so pytorch_grad_cam doesn't crash
        return self.model(x)[0]

app = typer.Typer()

@app.command()
def main(
    model_path: Path = typer.Option(
        "yolov8s.pt", 
        help="Path to the trained YOLO model."
    ),
    image_path: Path = typer.Option(
        "data/processed/visdrone_yolo/val/images/0000001_02999_d_0000005.jpg", 
        help="Path to the test image."
    ),
    img_size: int = typer.Option(
        640, 
        help="Image size for inference."
    ),
    target_layer_idx: int = typer.Option(
        22, 
        help="Index of the target C2f layer for XAI (usually 22 or 24 for YOLOv8s)."
    ),
    output_dir: Path = typer.Option(
        "reports/figures", 
        help="Directory to save the generated visualizations."
    )
):
    """
    Generate Explainable AI (XAI) heatmaps using EigenCAM for YOLO object detection.
    """
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        raise typer.Exit(code=1)
        
    if not image_path.exists():
        logger.error(f"Image not found: {image_path}")
        raise typer.Exit(code=1)

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load the model
    logger.info(f"Loading model: {model_path.name}")
    model = YOLO(str(model_path))

    # 2. Read and preprocess the image
    logger.info(f"Processing image: {image_path.name}")
    img = cv2.imread(str(image_path))
    img = cv2.resize(img, (img_size, img_size))
    rgb_img = img.copy()[:, :, ::-1] / 255.0

    # 3. Define the target layer for XAI
    # We dynamically select the layer based on the CLI argument
    try:
        target_layers = [model.model.model[target_layer_idx]]
        logger.info(f"Target layer selected: index {target_layer_idx}")
    except IndexError:
        logger.error(f"Target layer index {target_layer_idx} is out of bounds for this model.")
        raise typer.Exit(code=1)

    # 4. Initialize EigenCAM and prepare tensor
    wrapped_model = YOLOv8Wrapper(model.model)
    cam = EigenCAM(model=wrapped_model, target_layers=target_layers)
    input_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    # 5. Generate the heatmap
    logger.info("Generating EigenCAM heatmap...")
    grayscale_cam = cam(input_tensor=input_tensor)[0, :]
    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    # 6. Run standard YOLO inference for comparison
    logger.info("Running standard YOLO inference for comparison...")
    results = model(str(image_path), imgsz=img_size, verbose=False)
    res_plotted = results[0].plot()

    res_plotted_resized = cv2.resize(res_plotted, (img_size, img_size))

    # 7. Combine images side-by-side and save
    logger.info("Combining XAI heatmap and YOLO boxes...")

    cam_image_bgr = cam_image[:, :, ::-1]
    combined_image = np.hstack((cam_image_bgr, res_plotted_resized))
    
    combined_out = output_dir / f"combined_xai_{target_layer_idx}.jpg"
    cv2.imwrite(str(combined_out), combined_image)
    logger.success(f"Combined XAI image saved to: {combined_out}")


if __name__ == "__main__":
    app()