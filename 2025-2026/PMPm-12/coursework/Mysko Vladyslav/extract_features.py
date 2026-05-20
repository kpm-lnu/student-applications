from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from tqdm import tqdm


# =========================
# CONFIG
# =========================

INPUT_CSV = Path("dataset/labels_balanced.csv")
OUTPUT_CSV = Path("dataset/features_balanced.csv")

FRAGMENT_SIZE = 1024


# =========================
# FEATURE FUNCTIONS
# =========================

def shannon_entropy(data: np.ndarray) -> float:
    probs = np.bincount(data, minlength=256) / len(data)
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def printable_ratio(data: np.ndarray) -> float:
    return float(np.mean((data >= 32) & (data <= 126)))


def byte_histogram(data: np.ndarray) -> np.ndarray:
    return np.bincount(data, minlength=256) / len(data)


def safe_stat(data: np.ndarray, func) -> float:
    val = func(data)
    if np.isnan(val) or np.isinf(val):
        return 0.0
    return float(val)


def block_entropies(data: np.ndarray, block_size: int = 64) -> np.ndarray:
    values = []

    for i in range(0, len(data), block_size):
        block = data[i:i + block_size]
        if len(block) < 2:
            continue

        probs = np.bincount(block, minlength=256) / len(block)
        probs = probs[probs > 0]
        values.append(-np.sum(probs * np.log2(probs)))

    if not values:
        return np.array([0.0])

    return np.array(values, dtype=float)

import zlib


def chi_square_uniform(data: np.ndarray) -> float:
    counts = np.bincount(data, minlength=256)
    expected = len(data) / 256.0
    return float(np.sum((counts - expected) ** 2 / expected))


def compression_ratio(data: np.ndarray) -> float:
    raw = data.tobytes()
    compressed = zlib.compress(raw)
    return float(len(compressed) / len(raw))


def unique_byte_ratio(data: np.ndarray) -> float:
    return float(len(np.unique(data)) / 256.0)


def ascii_control_ratio(data: np.ndarray) -> float:
    return float(np.mean(data < 32))


def high_byte_ratio(data: np.ndarray) -> float:
    return float(np.mean(data >= 128))


def byte_diff_std(data: np.ndarray) -> float:
    diffs = np.abs(np.diff(data.astype(np.int16)))
    return float(np.std(diffs))


def half_entropy_features(data: np.ndarray):
    mid = len(data) // 2
    return shannon_entropy(data[:mid]), shannon_entropy(data[mid:])

def bit_features(data: np.ndarray):
    bits = np.unpackbits(data)

    ones_ratio = float(bits.mean())
    imbalance = float(abs(ones_ratio - 0.5))

    runs = int(np.sum(bits[1:] != bits[:-1]) + 1)
    mean_run = float(len(bits) / runs)

    return ones_ratio, imbalance, runs, mean_run


def byte_diff_mean(data: np.ndarray) -> float:
    diffs = np.abs(np.diff(data.astype(np.int16)))
    return float(np.mean(diffs))


def zero_ratio(data: np.ndarray) -> float:
    return float(np.mean(data == 0))

import sp80022suite


def safe_nist_value(x):
    if isinstance(x, (tuple, list)):
        vals = []
        for v in x:
            try:
                vals.append(float(v))
            except Exception:
                pass
        return vals
    try:
        return [float(x)]
    except Exception:
        return [0.0]


def nist_features(data: np.ndarray) -> dict:
    bits = np.unpackbits(data).astype(np.int8)

    features = {}

    tests = {
        "nist_frequency": lambda b: sp80022suite.frequency(b),
        "nist_block_frequency": lambda b: sp80022suite.blockFrequency(b),
        "nist_runs": lambda b: sp80022suite.runs(b),
        "nist_cusum": lambda b: sp80022suite.cusum(b),
        "nist_approx_entropy": lambda b: sp80022suite.approximateEntropy(b),
        "nist_serial": lambda b: sp80022suite.serial(b),
    }

    for name, fn in tests.items():
        try:
            vals = safe_nist_value(fn(bits))
        except Exception:
            vals = [0.0]

        for i, v in enumerate(vals):
            features[f"{name}_{i}"] = v

    return features

def fips_monobit_features(data: np.ndarray) -> dict:
    bits = np.unpackbits(data)
    ones = int(bits.sum())
    n = len(bits)

    return {
        "fips_monobit_ones": ones,
        "fips_monobit_ratio": float(ones / n),
        "fips_monobit_deviation": float(abs(ones / n - 0.5)),
    }


def fips_poker_feature(data: np.ndarray) -> dict:
    # FIPS poker uses 4-bit chunks
    bits = np.unpackbits(data)
    nibbles = bits[:len(bits) // 4 * 4].reshape(-1, 4)

    values = (
        nibbles[:, 0] * 8 +
        nibbles[:, 1] * 4 +
        nibbles[:, 2] * 2 +
        nibbles[:, 3]
    )

    counts = np.bincount(values, minlength=16)
    m = len(values)

    poker_stat = (16.0 / m) * np.sum(counts ** 2) - m

    return {
        "fips_poker_stat": float(poker_stat),
        "fips_poker_max_count": float(np.max(counts)),
        "fips_poker_min_count": float(np.min(counts)),
        "fips_poker_std_count": float(np.std(counts)),
    }


def fips_run_features(data: np.ndarray) -> dict:
    bits = np.unpackbits(data)

    run_counts = np.zeros(6, dtype=np.int32)
    current = 1

    for i in range(1, len(bits)):
        if bits[i] == bits[i - 1]:
            current += 1
        else:
            idx = min(current, 6) - 1
            run_counts[idx] += 1
            current = 1

    idx = min(current, 6) - 1
    run_counts[idx] += 1

    total_runs = int(run_counts.sum())

    features = {
        "fips_total_runs": total_runs,
        "fips_longest_run_bin": int(np.max(np.where(run_counts > 0)[0]) + 1),
    }

    for i in range(6):
        features[f"fips_runs_len_{i + 1}"] = int(run_counts[i])
        features[f"fips_runs_len_{i + 1}_ratio"] = float(run_counts[i] / max(total_runs, 1))

    return features


def fips_longest_run_features(data: np.ndarray) -> dict:
    bits = np.unpackbits(data)

    max_run = 1
    current = 1

    for i in range(1, len(bits)):
        if bits[i] == bits[i - 1]:
            current += 1
            max_run = max(max_run, current)
        else:
            current = 1

    return {
        "fips_longest_run": int(max_run),
        "fips_longest_run_norm": float(max_run / len(bits)),
    }


def fips_continuous_run_features(data: np.ndarray) -> dict:
    # repeated 16-byte chunks
    chunks = [
        bytes(data[i:i + 16])
        for i in range(0, len(data) - 15, 16)
    ]

    unique_chunks = len(set(chunks))
    total_chunks = len(chunks)

    return {
        "fips_16byte_chunk_unique_ratio": float(unique_chunks / max(total_chunks, 1)),
        "fips_16byte_chunk_repeat_count": int(total_chunks - unique_chunks),
    }


def fips_features(data: np.ndarray) -> dict:
    features = {}
    features.update(fips_monobit_features(data))
    features.update(fips_poker_feature(data))
    features.update(fips_run_features(data))
    features.update(fips_longest_run_features(data))
    features.update(fips_continuous_run_features(data))
    return features

def extract_features(data: bytes) -> dict:
    arr = np.frombuffer(data, dtype=np.uint8)

    if len(arr) != FRAGMENT_SIZE:
        raise ValueError(f"Expected {FRAGMENT_SIZE} bytes, got {len(arr)}")

    features = {}

    features["entropy"] = shannon_entropy(arr)
    features["mean"] = float(np.mean(arr))
    features["std"] = float(np.std(arr))
    features["skew"] = safe_stat(arr, skew)
    features["kurtosis"] = safe_stat(arr, kurtosis)
    features["printable_ratio"] = printable_ratio(arr)

    ones_ratio, imbalance, runs, mean_run = bit_features(arr)
    features["bit_ones_ratio"] = ones_ratio
    features["bit_imbalance"] = imbalance
    features["bit_runs"] = runs
    features["bit_mean_run"] = mean_run

    features["byte_diff_mean"] = byte_diff_mean(arr)

    local_entropy = block_entropies(arr, block_size=64)
    features["local_entropy_mean"] = float(np.mean(local_entropy))
    features["local_entropy_std"] = float(np.std(local_entropy))
    features["local_entropy_min"] = float(np.min(local_entropy))
    features["local_entropy_max"] = float(np.max(local_entropy))
    features["chi_square_uniform"] = chi_square_uniform(arr)
    features["compression_ratio"] = compression_ratio(arr)
    features["unique_byte_ratio"] = unique_byte_ratio(arr)
    features["ascii_control_ratio"] = ascii_control_ratio(arr)
    features["high_byte_ratio"] = high_byte_ratio(arr)
    features["byte_diff_std"] = byte_diff_std(arr)

    h1, h2 = half_entropy_features(arr)
    features["entropy_first_half"] = h1
    features["entropy_second_half"] = h2
    features["entropy_half_diff"] = abs(h1 - h2)
    features.update(nist_features(arr))
    features.update(fips_features(arr))

    hist = byte_histogram(arr)
    for i, value in enumerate(hist):
        features[f"byte_hist_{i}"] = float(value)

    return features


# =========================
# MAIN
# =========================

def main():
    df = pd.read_csv(INPUT_CSV)

    rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
        file_path = Path(row["file"])
        label = row["label"]

        data = file_path.read_bytes()

        if len(data) != FRAGMENT_SIZE:
            print(f"[skip] {file_path}: size={len(data)}")
            continue

        features = extract_features(data)

        output_row = {
            "file": str(file_path),
            "label": label,
        }

        if "original_file" in row:
            output_row["original_file"] = row["original_file"]

        output_row.update(features)
        rows.append(output_row)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUTPUT_CSV, index=False)

    print("\nDONE")
    print(f"Saved features to: {OUTPUT_CSV}")
    print(f"Rows: {len(out_df)}")
    print(f"Columns: {len(out_df.columns)}")

    print("\nClass distribution:")
    print(out_df["label"].value_counts())


if __name__ == "__main__":
    main()