from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


INPUT_CSV = Path("dataset/features_balanced.csv")
OUTPUT_DIR = Path("results_dataset_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DROP_COLUMNS = {
    "file",
    "label",
    "original_file",
}

CLASS_LABELS_UA = {
    "compressed": "архіви / стиснуті",
    "executable": "виконувані файли",
    "image": "зображення",
    "pdf": "PDF",
    "text": "текст",
}

RANDOM_SEED = 42
MAX_POINTS_PER_CLASS = 1000


def main():
    df = pd.read_csv(INPUT_CSV)

    feature_cols = [
        col for col in df.columns
        if col not in DROP_COLUMNS
    ]

    sampled_parts = []

    for label, group in df.groupby("label"):
        n = min(len(group), MAX_POINTS_PER_CLASS)
        sampled_parts.append(
            group.sample(n=n, random_state=RANDOM_SEED)
        )

    plot_df = pd.concat(sampled_parts, ignore_index=True)

    X = plot_df[feature_cols].values.astype(np.float32)
    y = plot_df["label"].values

    X_scaled = StandardScaler().fit_transform(X)

    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    X_pca = pca.fit_transform(X_scaled)

    explained = pca.explained_variance_ratio_

    pca_df = pd.DataFrame({
        "PC1": X_pca[:, 0],
        "PC2": X_pca[:, 1],
        "label": y,
    })

    pca_df["label_ua"] = (
        pca_df["label"]
        .map(CLASS_LABELS_UA)
        .fillna(pca_df["label"])
    )

    pca_df.to_csv(
        OUTPUT_DIR / "pca_2d_coordinates.csv",
        index=False,
    )

    plt.figure(figsize=(9, 7))

    for label in sorted(pca_df["label"].unique()):
        class_df = pca_df[pca_df["label"] == label]
        label_ua = CLASS_LABELS_UA.get(label, label)

        plt.scatter(
            class_df["PC1"],
            class_df["PC2"],
            s=10,
            alpha=0.6,
            label=label_ua,
        )

    plt.xlabel(f"ГК1 ({explained[0] * 100:.2f}% дисперсії)")
    plt.ylabel(f"ГК2 ({explained[1] * 100:.2f}% дисперсії)")
    plt.title("Проєкція ознак файлових фрагментів методом головних компонент")

    plt.legend(title="Клас")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = OUTPUT_DIR / "pca_2d_projection.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print("DONE")
    print(f"Saved PCA plot to: {output_path}")
    print(f"Saved PCA coordinates to: {OUTPUT_DIR / 'pca_2d_coordinates.csv'}")
    print(f"Explained variance PC1: {explained[0]:.4f}")
    print(f"Explained variance PC2: {explained[1]:.4f}")
    print(f"Total explained variance: {explained.sum():.4f}")


if __name__ == "__main__":
    main()