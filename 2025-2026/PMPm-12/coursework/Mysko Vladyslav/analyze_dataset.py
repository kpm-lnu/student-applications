from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# CONFIG
# =========================

INPUT_CSV = Path("dataset/features_balanced.csv")
OUTPUT_DIR = Path("results_dataset_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_COL = "label"

KEY_FEATURES = [
    "entropy",
    "printable_ratio",
    "compression_ratio",
    "chi_square_uniform",
    "local_entropy_std",
    "unique_byte_ratio",
    "byte_diff_mean",
    "byte_diff_std",
]

OPTIONAL_FEATURES = [
    "fips_poker_stat",
    "fips_longest_run",
    "fips_monobit_deviation",
    "fips_16byte_chunk_unique_ratio",
]


# =========================
# PLOTS
# =========================

def plot_class_distribution(df: pd.DataFrame):
    counts = df[LABEL_COL].value_counts().sort_index()

    plt.figure(figsize=(8, 5))
    plt.bar(counts.index, counts.values)
    plt.xlabel("Class")
    plt.ylabel("Number of fragments")
    plt.title("Class Distribution")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "class_distribution.png", dpi=300)
    plt.close()


CLASS_LABELS_UA = {
    "compressed": "архіви / стиснуті",
    "executable": "виконувані файли",
    "image": "світлини",
    "pdf": "PDF",
    "text": "текст",
}


def plot_boxplot(df: pd.DataFrame, feature: str):
    labels = sorted(df[LABEL_COL].unique())
    data = [df[df[LABEL_COL] == label][feature].dropna().values for label in labels]

    ua_labels = [CLASS_LABELS_UA.get(label, label) for label in labels]

    plt.figure(figsize=(9, 5))
    plt.boxplot(data, labels=ua_labels, showfliers=False)

    plt.xlabel("Клас")
    plt.ylabel(feature)
    plt.title(f"Розподіл ознаки '{feature}' за класами")
    plt.xticks(rotation=30, ha="right")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"boxplot_{feature}.png", dpi=300)
    plt.close()


def plot_selected_boxplots(df: pd.DataFrame):
    features = []

    for feature in KEY_FEATURES + OPTIONAL_FEATURES:
        if feature in df.columns:
            features.append(feature)

    for feature in features:
        plot_boxplot(df, feature)


def plot_correlation_heatmap(df: pd.DataFrame):
    selected = [
        feature for feature in KEY_FEATURES + OPTIONAL_FEATURES
        if feature in df.columns
    ]

    if len(selected) < 2:
        print("[warning] Not enough features for correlation heatmap")
        return

    corr = df[selected].corr()

    plt.figure(figsize=(10, 8))
    plt.imshow(corr.values, aspect="auto")
    plt.colorbar(label="Correlation")

    ticks = np.arange(len(selected))
    plt.xticks(ticks, selected, rotation=45, ha="right")
    plt.yticks(ticks, selected)

    plt.title("Correlation Heatmap of Selected Features")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "selected_feature_correlation_heatmap.png", dpi=300)
    plt.close()

    corr.to_csv(OUTPUT_DIR / "selected_feature_correlation_matrix.csv")


def plot_average_byte_histograms(df: pd.DataFrame):
    hist_cols = [f"byte_hist_{i}" for i in range(256)]

    if not all(col in df.columns for col in hist_cols):
        print("[warning] Byte histogram features not found, skipping average histogram plot")
        return

    labels = sorted(df[LABEL_COL].unique())

    for label in labels:
        class_df = df[df[LABEL_COL] == label]
        mean_hist = class_df[hist_cols].mean().values

        plt.figure(figsize=(10, 4))
        plt.bar(np.arange(256), mean_hist)
        plt.xlabel("Byte value")
        plt.ylabel("Average relative frequency")
        plt.title(f"Average Byte Histogram — {label}")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"average_byte_histogram_{label}.png", dpi=300)
        plt.close()


def plot_entropy_vs_compression(df: pd.DataFrame):
    if "entropy" not in df.columns or "compression_ratio" not in df.columns:
        return

    labels = sorted(df[LABEL_COL].unique())

    plt.figure(figsize=(8, 6))

    for label in labels:
        class_df = df[df[LABEL_COL] == label]
        plt.scatter(
            class_df["entropy"],
            class_df["compression_ratio"],
            s=8,
            alpha=0.5,
            label=label,
        )

    plt.xlabel("Shannon entropy")
    plt.ylabel("Compression ratio")
    plt.title("Entropy vs Compression Ratio")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "scatter_entropy_vs_compression_ratio.png", dpi=300)
    plt.close()


# =========================
# TABLES
# =========================

def save_basic_statistics(df: pd.DataFrame):
    counts = df[LABEL_COL].value_counts().sort_index()
    counts.to_csv(OUTPUT_DIR / "class_distribution.csv", header=["count"])

    selected = [
        feature for feature in KEY_FEATURES + OPTIONAL_FEATURES
        if feature in df.columns
    ]

    summary = (
        df.groupby(LABEL_COL)[selected]
        .agg(["mean", "std", "min", "median", "max"])
    )

    summary.to_csv(OUTPUT_DIR / "feature_summary_by_class.csv")


def save_dataset_overview(df: pd.DataFrame):
    overview = {
        "total_fragments": len(df),
        "num_classes": df[LABEL_COL].nunique(),
        "classes": ", ".join(sorted(df[LABEL_COL].unique())),
        "num_columns": len(df.columns),
    }

    overview_df = pd.DataFrame([overview])
    overview_df.to_csv(OUTPUT_DIR / "dataset_overview.csv", index=False)


# =========================
# MAIN
# =========================

def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)

    if LABEL_COL not in df.columns:
        raise ValueError(f"Column '{LABEL_COL}' not found in {INPUT_CSV}")

    print("Dataset shape:", df.shape)
    print("\nClass distribution:")
    print(df[LABEL_COL].value_counts().sort_index())

    save_dataset_overview(df)
    save_basic_statistics(df)

    plot_class_distribution(df)
    plot_selected_boxplots(df)
    plot_correlation_heatmap(df)
    plot_average_byte_histograms(df)
    plot_entropy_vs_compression(df)

    print("\nDONE")
    print(f"Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()