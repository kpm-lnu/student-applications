from pathlib import Path
import time

import typer
from loguru import logger
from ultralytics import YOLO

from src.config import PROCESSED_DATA_DIR

app = typer.Typer()

@app.command()
def main(
    model_path: Path = typer.Option(
        "models/yolov8s_p2_wiou/weights/best.pt", 
        help="Path to your trained model weights."
    ),
    data_config: Path = typer.Option(
        PROCESSED_DATA_DIR / "visdrone_yolo_tiled" / "dataset.yaml",
        "--data", "-d",
        help="Path to the dataset.yaml file."
    ),
    img_size: int = typer.Option(
        640, 
        help="Image size for Edge deployment (typically smaller, e.g., 640 or 960)."
    ),
    workspace: int = typer.Option(
        8, 
        help="Allocate memory (in GB) for TensorRT—increase for better optimization (8-16 for batch=1).",
    ),
    device: str = typer.Option(
        "0", 
        help="CUDA device index for export."
    ),
    use_int8: bool = typer.Option(
        False,
        "--int8/--no-int8",
        help="Quantize to INT8 instead of FP16 (smaller, faster on some edge devices)."
    ),
    validate: bool = typer.Option(
        True,
        "--validate/--no-validate",
        help="Run quick latency check after export."
    ),
    test_image: Path = typer.Option(
        "data/processed/visdrone_yolo/val/images/0000001_02999_d_0000005.jpg",
        help="Test image for post-export validation."
    )
):
    """
    Export a trained YOLO model to TensorRT for Edge deployment (real-time batch=1 inference).
    
    Key optimizations for edge deployment:
    - Fixed batch=1 (no dynamic shapes) to minimize latency
    - Larger workspace for better kernel selection
    - Optional INT8 quantization for smaller model + lower latency on Jetson
    """
    if not model_path.exists():
        logger.error(f"Model weights not found at: {model_path}")
        raise typer.Exit(code=1)

    logger.info(f"Loading model from: {model_path}")
    model = YOLO(str(model_path))

    # --- Export to TensorRT ---
    precision = "INT8" if use_int8 else "FP16"
    logger.info(
        f"Exporting to TensorRT {precision} (batch=1, fixed shape)...\n"
        f"  Image size: {img_size}×{img_size}\n"
        f"  Workspace: {workspace} GB\n"
        f"  This may take 3-10 minutes."
    )
    
    export_start = time.time()
    exported_path = model.export(
        format="engine",
        data=data_config,
        half=not use_int8,      # FP16 (or False forces full precision, INT8 handled below)
        int8=use_int8,          # Enable INT8 quantization if requested
        workspace=workspace,    # Critical: larger workspace=better kernel optimization for batch=1
        imgsz=img_size,
        device=device,
        dynamic=False,          # **KEY**: Disable dynamic shapes; lock to batch=1, fixed size
    )
    export_time = time.time() - export_start
    
    logger.success(f"✓ TensorRT Export completed in {export_time:.1f}s")
    logger.info(f"  Engine saved to: {exported_path}")
    logger.info(f"  Model size: {Path(exported_path).stat().st_size / (1024*1024):.1f} MB")

    # --- Post-export validation: quick latency check ---
    if validate and test_image.exists():
        logger.info("\n--- Running Post-Export Latency Check ---")
        try:
            import statistics
            
            # Load exported engine
            trt_model = YOLO(str(exported_path), task="detect")
            
            # Quick warmup
            for _ in range(5):
                trt_model(str(test_image), imgsz=img_size, verbose=False)
            
            # Timed inference (10 runs for stability)
            times_ms = []
            for _ in range(10):
                start = time.perf_counter()
                result = trt_model(str(test_image), imgsz=img_size, verbose=False)
                end = time.perf_counter()
                times_ms.append((end - start) * 1000.0)
            
            median_ms = statistics.median(times_ms)
            fps = 1000.0 / median_ms
            
            logger.success(f"✓ Latency check passed:")
            logger.info(f"  Median latency: {median_ms:.2f} ms")
            logger.info(f"  Estimated FPS: {fps:.2f}")
            logger.info(f"  Range: {min(times_ms):.2f}–{max(times_ms):.2f} ms\n")
            
        except Exception as e:
            logger.warning(f"Post-export validation failed (non-blocking): {e}")
    
    logger.info(
        "✓ Edge-optimized engine ready for deployment!\n"
        "  Next: Use `make benchmark` to compare with PyTorch baseline."
    )

if __name__ == "__main__":
    app()