from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


INPUT_CSV = Path("dataset/labels_balanced.csv")
OUTPUT_DIR = Path("results_dataset_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FRAGMENT_SIZE = 1024
IMAGE_SIZE = 32
RANDOM_SEED = 42


def load_random_fragment(df: pd.DataFrame, label: str) -> tuple[np.ndarray, Path]:
    class_df = df[df["label"] == label]

    if class_df.empty:
        raise ValueError(f"No fragments found for label: {label}")

    row = class_df.sample(n=1, random_state=RANDOM_SEED).iloc[0]
    file_path = Path(row["file"])

    data = file_path.read_bytes()

    if len(data) != FRAGMENT_SIZE:
        raise ValueError(f"Expected {FRAGMENT_SIZE} bytes, got {len(data)}: {file_path}")

    arr = np.frombuffer(data, dtype=np.uint8)
    image = arr.reshape(IMAGE_SIZE, IMAGE_SIZE)

    return image, file_path


def main():
    df = pd.read_csv(INPUT_CSV)

    text_image, text_path = load_random_fragment(df, "text")
    compressed_image, compressed_path = load_random_fragment(df, "compressed")

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    axes[0].imshow(text_image, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Текстовий фрагмент")
    axes[0].axis("off")

    axes[1].imshow(compressed_image, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("Стиснений фрагмент")
    axes[1].axis("off")

    plt.suptitle("1024-байтові фрагменти у вигляді 32×32 grayscale-зображень")
    plt.tight_layout()

    output_path = OUTPUT_DIR / "byte_image_text_vs_compressed.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print("DONE")
    print(f"Text sample: {text_path}")
    print(f"Compressed sample: {compressed_path}")
    print(f"Saved plot to: {output_path}")


if __name__ == "__main__":
    main()