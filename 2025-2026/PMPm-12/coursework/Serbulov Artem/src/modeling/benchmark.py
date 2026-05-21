from pathlib import Path
import math
import statistics
import time
from dataclasses import dataclass

import typer
from loguru import logger
import torch
from ultralytics import YOLO

app = typer.Typer()

@dataclass
class BenchmarkSummary:
    """Aggregated benchmark metrics for one model."""

    name: str
    model_path: str
    wall_times_ms: list[float]
    preprocess_ms: list[float]
    inference_ms: list[float]
    postprocess_ms: list[float]
    repeat_medians_ms: list[float]
    analysis_wall_times_ms: list[float]
    analysis_preprocess_ms: list[float]
    analysis_inference_ms: list[float]
    analysis_postprocess_ms: list[float]
    analysis_repeat_medians_ms: list[float]
    analysis_scope: str


def _sync_if_cuda() -> None:
    """Synchronize CUDA device so wall-clock measurements include GPU work."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _percentile(values: list[float], pct: float) -> float:
    """Return percentile using linear interpolation between sorted samples."""
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]

    sorted_values = sorted(values)
    rank = (len(sorted_values) - 1) * (pct / 100.0)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return sorted_values[low]
    weight = rank - low
    return sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight


def _run_single_pass(
    model: YOLO,
    test_image: str,
    img_size: int,
    warmup_runs: int,
    timed_runs: int,
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Run one warmup+timed pass and return collected latency samples."""
    for _ in range(warmup_runs):
        model(test_image, imgsz=img_size, verbose=False)

    wall_times_ms: list[float] = []
    preprocess_ms: list[float] = []
    inference_ms: list[float] = []
    postprocess_ms: list[float] = []

    for _ in range(timed_runs):
        _sync_if_cuda()
        start = time.perf_counter()
        result = model(test_image, imgsz=img_size, verbose=False)
        _sync_if_cuda()
        end = time.perf_counter()

        wall_times_ms.append((end - start) * 1000.0)

        speed = result[0].speed
        preprocess_ms.append(float(speed.get("preprocess", 0.0)))
        inference_ms.append(float(speed.get("inference", 0.0)))
        postprocess_ms.append(float(speed.get("postprocess", 0.0)))

    return wall_times_ms, preprocess_ms, inference_ms, postprocess_ms


def _measure_one_inference(
    model: YOLO,
    test_image: str,
    img_size: int,
) -> tuple[float, float, float, float]:
    """Measure one inference and return wall/pre/inference/post timings in milliseconds."""
    _sync_if_cuda()
    start = time.perf_counter()
    result = model(test_image, imgsz=img_size, verbose=False)
    _sync_if_cuda()
    end = time.perf_counter()

    speed = result[0].speed
    return (
        (end - start) * 1000.0,
        float(speed.get("preprocess", 0.0)),
        float(speed.get("inference", 0.0)),
        float(speed.get("postprocess", 0.0)),
    )


def run_benchmark(
    model_path: str,
    name: str,
    test_image: str,
    img_size: int,
    warmup_runs: int,
    timed_runs: int,
    repeats: int,
    discard_first_repeat: bool,
) -> BenchmarkSummary:
    """Run a stabilized benchmark and return robust latency statistics."""
    logger.info(f"--- Running Benchmark for {name} ({model_path}) ---")
    model = YOLO(model_path, task="detect")

    wall_times_ms: list[float] = []
    preprocess_ms: list[float] = []
    inference_ms: list[float] = []
    postprocess_ms: list[float] = []
    repeat_medians_ms: list[float] = []

    for repeat_idx in range(1, repeats + 1):
        run_wall, run_pre, run_inf, run_post = _run_single_pass(
            model=model,
            test_image=test_image,
            img_size=img_size,
            warmup_runs=warmup_runs,
            timed_runs=timed_runs,
        )

        wall_times_ms.extend(run_wall)
        preprocess_ms.extend(run_pre)
        inference_ms.extend(run_inf)
        postprocess_ms.extend(run_post)

        repeat_median = statistics.median(run_wall)
        repeat_medians_ms.append(repeat_median)
        logger.info(
            f"  Repeat {repeat_idx}/{repeats} median latency: {repeat_median:.2f} ms"
        )

    drop_first = discard_first_repeat and repeats > 1
    if discard_first_repeat and repeats == 1:
        logger.warning(
            "`discard_first_repeat=True` but `repeats=1`; using all samples for analysis."
        )

    sample_offset = timed_runs if drop_first else 0
    analysis_scope = "repeats 2..N (steady-state)" if drop_first else "all repeats"

    summary = BenchmarkSummary(
        name=name,
        model_path=model_path,
        wall_times_ms=wall_times_ms,
        preprocess_ms=preprocess_ms,
        inference_ms=inference_ms,
        postprocess_ms=postprocess_ms,
        repeat_medians_ms=repeat_medians_ms,
        analysis_wall_times_ms=wall_times_ms[sample_offset:],
        analysis_preprocess_ms=preprocess_ms[sample_offset:],
        analysis_inference_ms=inference_ms[sample_offset:],
        analysis_postprocess_ms=postprocess_ms[sample_offset:],
        analysis_repeat_medians_ms=repeat_medians_ms[1:] if drop_first else repeat_medians_ms,
        analysis_scope=analysis_scope,
    )

    _log_summary(summary)
    return summary


def run_paired_benchmark(
    pt_model_path: str,
    trt_model_path: str,
    test_image: str,
    img_size: int,
    warmup_runs: int,
    timed_runs: int,
    repeats: int,
    discard_first_repeat: bool,
) -> tuple[BenchmarkSummary, BenchmarkSummary]:
    """Run a paired benchmark with alternating PT/TRT inferences to reduce temporal drift bias."""
    logger.info("--- Running Paired Benchmark (alternating PT/TRT per repeat) ---")
    pt_model = YOLO(pt_model_path, task="detect")
    trt_model = YOLO(trt_model_path, task="detect")

    pt_wall: list[float] = []
    pt_pre: list[float] = []
    pt_inf: list[float] = []
    pt_post: list[float] = []
    pt_repeat_medians: list[float] = []

    trt_wall: list[float] = []
    trt_pre: list[float] = []
    trt_inf: list[float] = []
    trt_post: list[float] = []
    trt_repeat_medians: list[float] = []

    for repeat_idx in range(1, repeats + 1):
        for _ in range(warmup_runs):
            pt_model(test_image, imgsz=img_size, verbose=False)
            trt_model(test_image, imgsz=img_size, verbose=False)

        pt_run_wall: list[float] = []
        trt_run_wall: list[float] = []

        for _ in range(timed_runs):
            # Alternate order across repeats to reduce first-run bias.
            if repeat_idx % 2 == 1:
                pt_w, pt_p, pt_i, pt_po = _measure_one_inference(pt_model, test_image, img_size)
                trt_w, trt_p, trt_i, trt_po = _measure_one_inference(trt_model, test_image, img_size)
            else:
                trt_w, trt_p, trt_i, trt_po = _measure_one_inference(trt_model, test_image, img_size)
                pt_w, pt_p, pt_i, pt_po = _measure_one_inference(pt_model, test_image, img_size)

            pt_wall.append(pt_w)
            pt_pre.append(pt_p)
            pt_inf.append(pt_i)
            pt_post.append(pt_po)
            pt_run_wall.append(pt_w)

            trt_wall.append(trt_w)
            trt_pre.append(trt_p)
            trt_inf.append(trt_i)
            trt_post.append(trt_po)
            trt_run_wall.append(trt_w)

        pt_repeat_median = statistics.median(pt_run_wall)
        trt_repeat_median = statistics.median(trt_run_wall)
        pt_repeat_medians.append(pt_repeat_median)
        trt_repeat_medians.append(trt_repeat_median)
        logger.info(
            f"  Repeat {repeat_idx}/{repeats} medians -> PT: {pt_repeat_median:.2f} ms | TRT: {trt_repeat_median:.2f} ms"
        )

    drop_first = discard_first_repeat and repeats > 1
    if discard_first_repeat and repeats == 1:
        logger.warning(
            "`discard_first_repeat=True` but `repeats=1`; using all samples for analysis."
        )

    sample_offset = timed_runs if drop_first else 0
    analysis_scope = "repeats 2..N (steady-state)" if drop_first else "all repeats"

    pt_summary = BenchmarkSummary(
        name="PyTorch FP32",
        model_path=pt_model_path,
        wall_times_ms=pt_wall,
        preprocess_ms=pt_pre,
        inference_ms=pt_inf,
        postprocess_ms=pt_post,
        repeat_medians_ms=pt_repeat_medians,
        analysis_wall_times_ms=pt_wall[sample_offset:],
        analysis_preprocess_ms=pt_pre[sample_offset:],
        analysis_inference_ms=pt_inf[sample_offset:],
        analysis_postprocess_ms=pt_post[sample_offset:],
        analysis_repeat_medians_ms=pt_repeat_medians[1:] if drop_first else pt_repeat_medians,
        analysis_scope=analysis_scope,
    )

    trt_summary = BenchmarkSummary(
        name="TensorRT INT8",
        model_path=trt_model_path,
        wall_times_ms=trt_wall,
        preprocess_ms=trt_pre,
        inference_ms=trt_inf,
        postprocess_ms=trt_post,
        repeat_medians_ms=trt_repeat_medians,
        analysis_wall_times_ms=trt_wall[sample_offset:],
        analysis_preprocess_ms=trt_pre[sample_offset:],
        analysis_inference_ms=trt_inf[sample_offset:],
        analysis_postprocess_ms=trt_post[sample_offset:],
        analysis_repeat_medians_ms=trt_repeat_medians[1:] if drop_first else trt_repeat_medians,
        analysis_scope=analysis_scope,
    )

    _log_summary(pt_summary)
    _log_summary(trt_summary)
    return pt_summary, trt_summary


def _log_summary(summary: BenchmarkSummary) -> None:
    """Print robust benchmark metrics for one model."""
    samples = summary.analysis_wall_times_ms
    median_ms = statistics.median(samples)
    mean_ms = statistics.fmean(samples)
    std_ms = statistics.stdev(samples) if len(samples) > 1 else 0.0
    p90_ms = _percentile(samples, 90)
    p95_ms = _percentile(samples, 95)
    cv_pct = (std_ms / mean_ms * 100.0) if mean_ms > 0 else 0.0

    pre_med = statistics.median(summary.analysis_preprocess_ms) if summary.analysis_preprocess_ms else 0.0
    inf_med = statistics.median(summary.analysis_inference_ms) if summary.analysis_inference_ms else 0.0
    post_med = statistics.median(summary.analysis_postprocess_ms) if summary.analysis_postprocess_ms else 0.0

    fps_median = 1000.0 / median_ms if median_ms > 0 else 0.0

    logger.success(f"{summary.name} Benchmark Results:")
    logger.info(f"  Analysis scope:{summary.analysis_scope}")
    logger.info(f"  Samples:       {len(samples)}")
    logger.info(f"  Median:        {median_ms:.2f} ms")
    logger.info(f"  Mean:          {mean_ms:.2f} ms")
    logger.info(f"  Std dev:       {std_ms:.2f} ms (CV: {cv_pct:.2f}%)")
    logger.info(f"  P90 / P95:     {p90_ms:.2f} / {p95_ms:.2f} ms")
    logger.info(f"  Min / Max:     {min(samples):.2f} / {max(samples):.2f} ms")
    logger.info(f"  Median FPS:    {fps_median:.2f}")
    logger.info(f"  Median breakdown from model.speed (ms):")
    logger.info(f"    Pre-process: {pre_med:.2f}")
    logger.info(f"    Inference:   {inf_med:.2f}")
    logger.info(f"    Post-process:{post_med:.2f}")
    logger.info(
        f"  Repeat medians (all): {[round(v, 2) for v in summary.repeat_medians_ms]}"
    )
    logger.info(
        f"  Repeat medians (analysis): {[round(v, 2) for v in summary.analysis_repeat_medians_ms]}\n"
    )


def _report_regression(baseline: BenchmarkSummary, candidate: BenchmarkSummary) -> None:
    """Report latency change with repeat-level confidence interval."""
    base_median = statistics.median(baseline.analysis_wall_times_ms)
    cand_median = statistics.median(candidate.analysis_wall_times_ms)

    if base_median <= 0:
        logger.warning("Baseline median latency is non-positive; skipping regression report.")
        return

    delta_ms = cand_median - base_median
    delta_pct = (delta_ms / base_median) * 100.0
    baseline_spread_ms = _percentile(baseline.analysis_wall_times_ms, 95) - base_median
    exceeds_noise = abs(delta_ms) > baseline_spread_ms

    paired_repeat_count = min(
        len(baseline.analysis_repeat_medians_ms),
        len(candidate.analysis_repeat_medians_ms),
    )
    repeat_deltas = [
        candidate.analysis_repeat_medians_ms[i] - baseline.analysis_repeat_medians_ms[i]
        for i in range(paired_repeat_count)
    ]
    mean_repeat_delta = statistics.fmean(repeat_deltas) if repeat_deltas else 0.0
    if len(repeat_deltas) > 1:
        repeat_delta_sd = statistics.stdev(repeat_deltas)
        ci_half_width = 1.96 * (repeat_delta_sd / math.sqrt(len(repeat_deltas)))
    else:
        repeat_delta_sd = 0.0
        ci_half_width = 0.0

    ci_low = mean_repeat_delta - ci_half_width
    ci_high = mean_repeat_delta + ci_half_width
    ci_excludes_zero = ci_low > 0 or ci_high < 0

    logger.info("--- Relative Comparison (Candidate vs Baseline) ---")
    logger.info(f"  Analysis scope:   {baseline.analysis_scope}")
    logger.info(f"  Baseline median:  {base_median:.2f} ms")
    logger.info(f"  Candidate median: {cand_median:.2f} ms")
    logger.info(f"  Delta:            {delta_ms:+.2f} ms ({delta_pct:+.2f}%)")
    logger.info(f"  Baseline noise band (P95 - median): {baseline_spread_ms:.2f} ms")
    logger.info(f"  Exceeds baseline variability: {'YES' if exceeds_noise else 'NO'}")
    if repeat_deltas:
        logger.info(
            f"  Repeat delta mean ±95% CI: {mean_repeat_delta:+.2f} ms [{ci_low:+.2f}, {ci_high:+.2f}] (n={len(repeat_deltas)})"
        )
        logger.info(f"  CI excludes zero (stable conclusion): {'YES' if ci_excludes_zero else 'NO'}\n")
    else:
        logger.info("  Repeat-level CI unavailable (no paired repeat medians).\n")


@app.command()
def main(
    pt_model: Path = typer.Option(
        "yolov8s.pt", 
        help="Path to the PyTorch FP32 model (e.g., best.pt)."
    ),
    trt_model: str = typer.Option(
        "", 
        help="Path to the TensorRT INT8 model (e.g., best.engine). Leave empty to skip."
    ),
    test_image: Path = typer.Option(
        "data/processed/visdrone_yolo/val/images/0000001_02999_d_0000001.jpg", 
        help="Path to a test image for inference."
    ),
    img_size: int = typer.Option(
        640, 
        help="Image size used for inference."
    ),
    warmup_runs: int = typer.Option(
        20,
        min=1,
        help="Number of warm-up inferences before timed runs (per repeat).",
    ),
    timed_runs: int = typer.Option(
        100,
        min=5,
        help="Number of timed inferences to collect (per repeat).",
    ),
    repeats: int = typer.Option(
        3,
        min=1,
        help="How many independent warmup+timed passes to run.",
    ),
    discard_first_repeat: bool = typer.Option(
        True,
        "--discard-first-repeat/--keep-first-repeat",
        help="Use repeats 2..N as primary metrics to avoid startup drift.",
    ),
    paired: bool = typer.Option(
        True,
        "--paired/--sequential",
        help="Alternate PT/TRT inferences within each repeat to reduce drift bias.",
    )
):
    """
    Evaluate and compare inference speed (Latency/FPS) between models.
    """
    if not pt_model.exists():
        logger.error(f"PyTorch model not found: {pt_model}")
        raise typer.Exit(code=1)
        
    if not test_image.exists():
        logger.error(f"Test image not found: {test_image}")
        raise typer.Exit(code=1)

    # Benchmark TensorRT model if the path is provided
    if trt_model:
        trt_path = Path(trt_model)
        if trt_path.exists():
            if paired:
                baseline, candidate = run_paired_benchmark(
                    pt_model_path=str(pt_model),
                    trt_model_path=str(trt_path),
                    test_image=str(test_image),
                    img_size=img_size,
                    warmup_runs=warmup_runs,
                    timed_runs=timed_runs,
                    repeats=repeats,
                    discard_first_repeat=discard_first_repeat,
                )
            else:
                baseline = run_benchmark(
                    model_path=str(pt_model),
                    name="PyTorch FP32",
                    test_image=str(test_image),
                    img_size=img_size,
                    warmup_runs=warmup_runs,
                    timed_runs=timed_runs,
                    repeats=repeats,
                    discard_first_repeat=discard_first_repeat,
                )
                candidate = run_benchmark(
                    model_path=str(trt_path),
                    name="TensorRT INT8",
                    test_image=str(test_image),
                    img_size=img_size,
                    warmup_runs=warmup_runs,
                    timed_runs=timed_runs,
                    repeats=repeats,
                    discard_first_repeat=discard_first_repeat,
                )
            _report_regression(baseline, candidate)
        else:
            logger.warning(f"TensorRT model not found at {trt_model}. Skipping TRT benchmark.")
    else:
        # Single-model benchmark path
        run_benchmark(
            model_path=str(pt_model),
            name="PyTorch FP32",
            test_image=str(test_image),
            img_size=img_size,
            warmup_runs=warmup_runs,
            timed_runs=timed_runs,
            repeats=repeats,
            discard_first_repeat=discard_first_repeat,
        )

if __name__ == "__main__":
    app()