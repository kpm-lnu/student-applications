#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = object_detection_tracking
PYTHON_VERSION = 3.12
PYTHON_INTERPRETER = python

# Dataset roots (Ultralytics expects a dataset root containing train/val/test_dev)
FULL_DATASET_ROOT = data/processed/visdrone_yolo
TILED_DATASET_ROOT = data/processed/visdrone_yolo_tiled

# Default training dataset root (override with `make train DATASET_ROOT=...`)
DATASET_ROOT = $(FULL_DATASET_ROOT)

# Raw VisDrone split to convert (expects `images/` + `annotations/`)
#RAW_SPLIT_DIR = data/raw/VisDrone2019-DET-train
#RAW_SPLIT_DIR = data/raw/VisDrone2019-DET-val
RAW_SPLIT_DIR = data/raw/VisDrone2019-DET-test-dev

# Output split dir for conversion (creates `images/` + `labels/` inside this dir)
#OUT_SPLIT_DIR = $(FULL_DATASET_ROOT)/train
#OUT_SPLIT_DIR = $(FULL_DATASET_ROOT)/val
OUT_SPLIT_DIR = $(FULL_DATASET_ROOT)/test_dev

# Tiling defaults (balanced preset)
TILE_SIZE = 640
TILE_OVERLAP = 0.25
TILE_MIN_INTERSECTION = 0.30
TILE_KEEP_EMPTY_PROB = 0.10
TILE_SEED = 1337

# Training defaults
EPOCHS = 50
BATCH = 32
IMG_SIZE = 640
DEVICE = 0

# Benchmark defaults (override via CLI, e.g. `make benchmark BENCH_TIMED_RUNS=200`)
BENCH_WARMUP_RUNS = 20
BENCH_TIMED_RUNS = 150
BENCH_REPEATS = 10
# Use one of: --discard-first-repeat or --keep-first-repeat
BENCH_STEADY_STATE_FLAG = --discard-first-repeat

# Path to your best model (override via CLI: make evaluate MODEL_PATH=...)
MODEL_PATH = models/EEML_2026_Detection/run_02/weights/best.pt
TRT_MODEL_PATH = models/EEML_2026_Detection/run_02/weights/best.engine
TEST_IMAGE = data/processed/visdrone_yolo/val/images/0000001_02999_d_0000005.jpg
# XAI_LAYER = 15
XAI_LAYER = 18

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## $(PYTHON_INTERPRETER) -m pip install ./ultralytics_src
## Install Python dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install --upgrade pip
	$(PYTHON_INTERPRETER) -m pip install -e .

## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format

## Run tests
.PHONY: test
test:
	python -m pytest tests

## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	$(PYTHON_INTERPRETER) -m venv .venv
	@echo ">>> New virtual environment created. Activate with:"
	@echo ">>> Windows: .\.venv\Scripts\activate"
	@echo ">>> Unix/macOS: source ./.venv/bin/activate"


#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

## Make dataset
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) -m src.dataset --help
	@echo ""
	@echo ">>> Use one of:"
	@echo ">>>   make convert RAW_SPLIT_DIR=... OUT_SPLIT_DIR=..."
	@echo ">>>   make tile"

## Convert one VisDrone split (raw -> YOLO) into a split dir under the dataset root
.PHONY: convert
convert: requirements
	$(PYTHON_INTERPRETER) -m src.dataset main \
		--input-dir $(RAW_SPLIT_DIR) \
		--output-dir $(OUT_SPLIT_DIR)

## Create offline tiled training split (train tiles only; val/test_dev full frames via symlink)
.PHONY: tile
tile: requirements
	$(PYTHON_INTERPRETER) -m src.dataset tile-train-split \
		--source-dataset $(FULL_DATASET_ROOT) \
		--output-dataset $(TILED_DATASET_ROOT) \
		--tile-size $(TILE_SIZE) \
		--overlap $(TILE_OVERLAP) \
		--min-intersection-ratio $(TILE_MIN_INTERSECTION) \
		--keep-empty-prob $(TILE_KEEP_EMPTY_PROB) \
		--seed $(TILE_SEED)

## Train a model
.PHONY: train
train: requirements
	$(PYTHON_INTERPRETER) -m src.modeling.train \
		--data-dir $(DATASET_ROOT) \
		--epochs $(EPOCHS) \
		--batch-size $(BATCH) \
		--img-size $(IMG_SIZE) \
		--device $(DEVICE)

## Train on tiled dataset (requires `make tile` first)
.PHONY: train_tiled
train_tiled: requirements
	$(PYTHON_INTERPRETER) -m src.modeling.train \
		--data-dir $(TILED_DATASET_ROOT) \
		--epochs $(EPOCHS) \
		--batch-size $(BATCH) \
		--img-size $(IMG_SIZE) \
		--device $(DEVICE)

## Evaluate the trained model on the independent TEST split
.PHONY: evaluate
evaluate: requirements
	$(PYTHON_INTERPRETER) -m src.modeling.evaluate \
		--model $(TRT_MODEL_PATH) \
		--data $(TILED_DATASET_ROOT)/dataset.yaml \
		--imgsz $(IMG_SIZE) \
		--batch $(BATCH) \
		--device $(DEVICE)

## Export the trained model to TensorRT (FP16) for Edge deployment
.PHONY: export_edge
export_edge: requirements
	$(PYTHON_INTERPRETER) -m src.modeling.export_edge \
		--model-path $(MODEL_PATH) \
		--img-size $(IMG_SIZE) \
		--device $(DEVICE) \
		--int8

## Run inference benchmark (Latency & FPS) for PyTorch and TensorRT
.PHONY: benchmark
benchmark: requirements
	$(PYTHON_INTERPRETER) -m src.modeling.benchmark \
		--pt-model $(MODEL_PATH) \
		--trt-model $(TRT_MODEL_PATH) \
		--test-image $(TEST_IMAGE) \
		--img-size $(IMG_SIZE) \
		--warmup-runs $(BENCH_WARMUP_RUNS) \
		--timed-runs $(BENCH_TIMED_RUNS) \
		--repeats $(BENCH_REPEATS) \
		$(BENCH_STEADY_STATE_FLAG)

## Generate Explainable AI (XAI) heatmaps using EigenCAM
.PHONY: xai
xai: requirements
	$(PYTHON_INTERPRETER) -m src.modeling.inference_xai \
		--model-path $(MODEL_PATH) \
		--image-path $(TEST_IMAGE) \
		--img-size $(IMG_SIZE) \
		--target-layer-idx $(XAI_LAYER)

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)