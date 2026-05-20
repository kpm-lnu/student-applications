from pathlib import Path
import time
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# =========================
# CONFIG
# =========================

INPUT_CSV = Path("dataset/labels_balanced.csv")
OUTPUT_DIR = Path("results_bytecnn")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FRAGMENT_SIZE = 1024
IMAGE_SIZE = 32

PATIENCE = 4
MIN_DELTA = 0.002
TEST_SIZE = 0.2
VAL_SIZE = 0.15

BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 3e-4
PATIENCE = 7
RANDOM_SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# REPRODUCIBILITY
# =========================

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================
# DATASET
# =========================

class FragmentDataset(Dataset):
    def __init__(self, df: pd.DataFrame, labels: np.ndarray):
        self.df = df.reset_index(drop=True)
        self.labels = labels

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_path = Path(self.df.loc[idx, "file"])
        data = file_path.read_bytes()

        if len(data) != FRAGMENT_SIZE:
            raise ValueError(f"Expected {FRAGMENT_SIZE} bytes, got {len(data)}: {file_path}")

        arr = np.frombuffer(data, dtype=np.uint8).astype(np.float32)
        arr = arr / 255.0

        image = arr.reshape(1, IMAGE_SIZE, IMAGE_SIZE)
        x = torch.tensor(image, dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)

        return x, y


# =========================
# MODEL
# =========================

class ByteCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),      # 32 -> 16
            nn.Dropout2d(0.05),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),      # 16 -> 8
            nn.Dropout2d(0.10),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),      # 8 -> 4
            nn.Dropout2d(0.15),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x

# =========================
# TRAIN / EVAL
# =========================

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    for x, y in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)

        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(y.detach().cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")

    return avg_loss, acc, macro_f1


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    for x, y in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)

        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")

    return avg_loss, acc, macro_f1, np.array(all_labels), np.array(all_preds)


def plot_training_curves(history):
    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], label="Train loss")
    plt.plot(history["val_loss"], label="Validation loss")
    # Перший графік: Втрати (Loss)
    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], label="Втрати на тренуванні")
    plt.plot(history["val_loss"], label="Втрати на валідації")
    plt.xlabel("Епоха")
    plt.ylabel("Втрати (Loss)")
    plt.title("Втрати під час тренування та валідації ByteCNN")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "bytecnn_loss_curve.png", dpi=300)
    plt.close()

    # Другий графік: Точність (Accuracy)
    plt.figure(figsize=(8, 5))
    plt.plot(history["train_acc"], label="Точність на тренуванні")
    plt.plot(history["val_acc"], label="Точність на валідації")
    plt.xlabel("Епоха")
    plt.ylabel("Точність (Accuracy)")
    plt.title("Точність під час тренування та валідації ByteCNN")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "bytecnn_accuracy_curve.png", dpi=300)
    plt.close()

CLASS_LABELS_UA = {
    "compressed": "архіви / стиснуті",
    "executable": "виконувані файли",
    "image": "зображення",
    "pdf": "PDF",
    "text": "текст",
}
def plot_confusion_matrix(cm, class_names):
    ua_class_names = [CLASS_LABELS_UA[name] for name in class_names]

    plt.figure(figsize=(8, 6))
    plt.imshow(cm)

    plt.title("Матриця плутанини — 1D ByteCNN")
    plt.colorbar()

    ticks = np.arange(len(class_names))

    plt.xticks(ticks, ua_class_names, rotation=20, ha="right")
    plt.yticks(ticks, ua_class_names)

    plt.xlabel("Передбачений клас")
    plt.ylabel("Справжній клас")

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix_1dcnn.png", dpi=300)
    plt.close()



# =========================
# MAIN
# =========================

def main():
    set_seed(RANDOM_SEED)

    print(f"Device: {DEVICE}")

    df = pd.read_csv(INPUT_CSV)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["label"].values)
    class_names = list(label_encoder.classes_)

    print("Classes:", class_names)
    print("Dataset size:", len(df))
    print("\nClass distribution:")
    print(df["label"].value_counts())

    train_val_df, test_df, y_train_val, y_test = train_test_split(
        df,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    train_df, val_df, y_train, y_val = train_test_split(
        train_val_df,
        y_train_val,
        test_size=VAL_SIZE,
        random_state=RANDOM_SEED,
        stratify=y_train_val,
    )

    train_dataset = FragmentDataset(train_df, y_train)
    val_dataset = FragmentDataset(val_df, y_val)
    test_dataset = FragmentDataset(test_df, y_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    model = ByteCNN(num_classes=len(class_names)).to(DEVICE)

    criterion = nn.CrossEntropyLoss(
        label_smoothing=0.1
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-4,
        weight_decay=1e-3,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )

    history = {
        "train_loss": [],
        "train_acc": [],
        "train_f1": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
    }

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc, train_f1 = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
        )

        val_loss, val_acc, val_f1, _, _ = evaluate(
            model,
            val_loader,
            criterion,
        )
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["train_f1"].append(train_f1)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, train_f1={train_f1:.4f} | "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, val_f1={val_f1:.4f}"
        )

        if val_loss < best_val_loss - MIN_DELTA:
            best_val_loss = val_loss
            best_state = model.state_dict()
            patience_counter = 0
            torch.save(best_state, OUTPUT_DIR / "bytecnn_best.pt")
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

    train_time = time.time() - start_time

    if best_state is not None:
        model.load_state_dict(torch.load(OUTPUT_DIR / "bytecnn_best.pt", map_location=DEVICE))

    test_loss, test_acc, test_f1, y_true, y_pred = evaluate(
        model,
        test_loader,
        criterion,
    )

    print("\n==============================")
    print("TEST RESULTS")
    print("==============================")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"Macro-F1: {test_f1:.4f}")
    print(f"Train time: {train_time:.3f} sec")

    print("\nClassification report:")
    report_text = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0,
    )
    print(report_text)

    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(y_true, y_pred)

    plot_training_curves(history)
    plot_confusion_matrix(cm, class_names)

    results = {
        "model": "ByteCNN",
        "accuracy": test_acc,
        "macro_f1": test_f1,
        "test_loss": test_loss,
        "train_time_sec": train_time,
        "classes": class_names,
        "epochs_trained": len(history["train_loss"]),
        "fragment_size": FRAGMENT_SIZE,
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
    }

    with (OUTPUT_DIR / "bytecnn_results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    with (OUTPUT_DIR / "bytecnn_classification_report.json").open("w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=4)

    pd.DataFrame(history).to_csv(OUTPUT_DIR / "bytecnn_training_history.csv", index=False)

    print("\nSaved:")
    print(OUTPUT_DIR / "bytecnn_best.pt")
    print(OUTPUT_DIR / "bytecnn_results.json")
    print(OUTPUT_DIR / "bytecnn_loss_curve.png")
    print(OUTPUT_DIR / "bytecnn_accuracy_curve.png")
    print(OUTPUT_DIR / "confusion_matrix_bytecnn.png")


if __name__ == "__main__":
    main()