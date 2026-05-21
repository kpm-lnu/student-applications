from pathlib import Path
import time
import json
import math
import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
    matthews_corrcoef,
)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# CONFIG
# ============================================================

INPUT_CSV = Path("dataset/labels_balanced.csv")
OUTPUT_DIR = Path("results_byte2image_full")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FRAGMENT_SIZE = 1024

# Paper-style Byte2Image parameters
NGRAM = 8
SHIFT_COUNT = 8
EMBED_DIM = 32

TEST_SIZE = 0.2
VAL_SIZE = 0.15

# For CPU, 16 or 32 is safer.
# If you use GPU, try 64.
BATCH_SIZE = 32
EPOCHS = 50

LEARNING_RATE = 3e-4
MIN_LR_RATIO = 0.05
WEIGHT_DECAY = 2e-2

LABEL_SMOOTHING = 0.08

PATIENCE = 8
MIN_DELTA = 0.002

BYTE_MASK_PROB = 0.03
WARMUP_EPOCHS = 2

RANDOM_SEED = 42

# On Windows/WSL CPU, num_workers=0 is safer.
# On Linux/GPU, you may try 2 or 4.
NUM_WORKERS = 0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True


# ============================================================
# BYTE2IMAGE TRANSFORM
# ============================================================

def create_shifted_byte_matrix(data: bytes) -> np.ndarray:
    """
    Create Ns x 8 shifted byte matrix.

    Input:
        data: 1024 raw bytes

    Output:
        xin: shape [1024, 8]

    Column 0:
        original byte sequence

    Columns 1..7:
        bit-shifted byte sequences, crossing byte boundaries.
        This exposes intra-byte information.
    """
    arr = np.frombuffer(data, dtype=np.uint8)

    if len(arr) != FRAGMENT_SIZE:
        raise ValueError(f"Expected {FRAGMENT_SIZE} bytes, got {len(arr)}")

    x = arr.astype(np.uint16)

    # next byte is needed because shifted windows cross byte boundaries
    x_next = np.roll(x, -1)
    x_next[-1] = 0

    shifted_cols = []

    for shift in range(SHIFT_COUNT):
        if shift == 0:
            shifted = x
        else:
            shifted = ((x << shift) & 0xFF) | (x_next >> (8 - shift))

        shifted_cols.append(shifted.astype(np.uint8))

    # Shape: [8, Ns] -> [Ns, 8]
    xin = np.stack(shifted_cols, axis=1)

    return xin


def byte2image_ngram(data: bytes, ngram: int = NGRAM) -> np.ndarray:
    """
    Paper-style Byte2Image conversion.

    Step 1:
        Raw byte sequence -> shifted byte matrix [Ns, 8]

    Step 2:
        Intra-byte n-gram stacking:
        [Ns, 8] -> [Ns - ngram + 1, 8 * ngram]

    Step 3:
        Treat as grayscale image:
        [1, H, W]

    For FRAGMENT_SIZE=1024 and NGRAM=16:
        H = 1009
        W = 128

    Output:
        image tensor as numpy array: [1, 1009, 128]
    """
    xin = create_shifted_byte_matrix(data)  # [1024, 8]

    # sliding windows over byte positions
    # windows shape from sliding_window_view: [H, 8, ngram]
    windows = np.lib.stride_tricks.sliding_window_view(
        xin,
        window_shape=ngram,
        axis=0,
    )

    # Convert [H, 8, ngram] -> [H, ngram, 8] -> [H, ngram * 8]
    xngram = windows.transpose(0, 2, 1).reshape(windows.shape[0], ngram * SHIFT_COUNT)

    # Normalize to [0, 1]
    xngram = xngram.astype(np.float32) / 255.0

    # [1, H, W]
    return xngram[None, :, :]


def raw_byte_sequence(data: bytes) -> np.ndarray:
    """
    Raw 1D byte sequence branch.

    Output:
        [1024]
    """
    arr = np.frombuffer(data, dtype=np.uint8).astype(np.float32)
    return arr / 255.0


# ============================================================
# DATASET
# ============================================================

class Byte2ImageFullDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        labels: np.ndarray,
        augment: bool = False,
        byte_mask_prob: float = 0.0,
    ):
        self.df = df.reset_index(drop=True)
        self.labels = labels
        self.augment = augment
        self.byte_mask_prob = byte_mask_prob

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_path = Path(self.df.loc[idx, "file"])
        data = file_path.read_bytes()

        if len(data) != FRAGMENT_SIZE:
            raise ValueError(
                f"Expected {FRAGMENT_SIZE} bytes, got {len(data)}: {file_path}"
            )

        # Light byte-level augmentation.
        # We only use it for train set.
        if self.augment and self.byte_mask_prob > 0.0:
            arr = np.frombuffer(data, dtype=np.uint8).copy()
            mask = np.random.rand(arr.shape[0]) < self.byte_mask_prob
            arr[mask] = 0
            data = arr.tobytes()

        img = byte2image_ngram(data, ngram=NGRAM)
        seq = raw_byte_sequence(data)

        img_tensor = torch.from_numpy(img).float()
        seq_tensor = torch.from_numpy(seq).float()
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)

        return img_tensor, seq_tensor, label_tensor


# ============================================================
# MODEL: RESNET-STYLE BACKBONE
# ============================================================

class BasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, dropout: float = 0.0):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.relu(out)

        return out


class SmallResNetBackbone(nn.Module):
    """
    ResNet18-style image backbone, but smaller than full ResNet18.

    Reason:
        Full ResNet18 may overfit strongly on 10k coursework samples,
        especially on CPU and with fragment-level data.

    Input:
        [B, 1, H, EMBED_DIM]

    Output:
        feature vector [B, 256]
    """
    def __init__(self, in_channels: int = 1, base_channels: int = 16):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                base_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )

        self.layer1 = self._make_layer(
            in_channels=base_channels,
            out_channels=base_channels,
            blocks=1,
            stride=1,
            dropout=0.05,
        )

        self.layer2 = self._make_layer(
            in_channels=base_channels,
            out_channels=base_channels * 2,
            blocks=1,
            stride=2,
            dropout=0.10,
        )

        self.layer3 = self._make_layer(
            in_channels=base_channels * 2,
            out_channels=base_channels * 4,
            blocks=1,
            stride=2,
            dropout=0.15,
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.out_dim = base_channels * 4

    def _make_layer(
        self,
        in_channels: int,
        out_channels: int,
        blocks: int,
        stride: int,
        dropout: float,
    ):
        layers = [
            BasicBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                dropout=dropout,
            )
        ]

        for _ in range(1, blocks):
            layers.append(
                BasicBlock(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    stride=1,
                    dropout=dropout,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.pool(x)
        x = torch.flatten(x, 1)

        return x


class Byte2ImageFullFusionNet(nn.Module):
    """
    Full Byte2Image-style fusion model.

    Branch 1:
        Byte2Image n-gram grayscale image
        -> wide convolution
        -> reshape
        -> ResNet-style CNN backbone

    Branch 2:
        Raw 1D byte sequence
        -> shallow fully-connected byte branch

    Fusion:
        concatenate image features and byte features
        -> classifier
    """
    def __init__(self, num_classes: int):
        super().__init__()

        # Input image:
        # [B, 1, H, 8 * NGRAM]
        #
        # For 1024 bytes and NGRAM=16:
        # [B, 1, 1009, 128]
        #
        # Wide conv maps n-gram width into EMBED_DIM feature channels.
        self.wide_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=(1, SHIFT_COUNT * NGRAM),
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.05),
        )

        # After wide_conv:
        # [B, EMBED_DIM, H, 1]
        #
        # Then reshape to:
        # [B, 1, H, EMBED_DIM]
        self.image_backbone = SmallResNetBackbone(
            in_channels=1,
            base_channels=12,
        )

        image_feat_dim = self.image_backbone.out_dim

        # Raw byte sequence branch.
        # Paper's idea: this branch keeps inter-byte/co-occurrence information.
        self.byte_branch = nn.Sequential(
            nn.Linear(FRAGMENT_SIZE, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.50),
        )

        byte_feat_dim = 32

        self.classifier = nn.Sequential(
            nn.Linear(image_feat_dim + byte_feat_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.55),

            nn.Linear(64, num_classes),
        )

    def forward(self, img, seq):
        # img: [B, 1, H, 8 * NGRAM]
        x = self.wide_conv(img)

        # [B, EMBED_DIM, H, 1] -> [B, EMBED_DIM, H]
        x = x.squeeze(-1)

        # [B, EMBED_DIM, H] -> [B, H, EMBED_DIM]
        x = x.permute(0, 2, 1)

        # [B, H, EMBED_DIM] -> [B, 1, H, EMBED_DIM]
        x = x.unsqueeze(1)

        img_feat = self.image_backbone(x)

        seq_feat = self.byte_branch(seq)

        fused = torch.cat([img_feat, seq_feat], dim=1)

        logits = self.classifier(fused)

        return logits


# ============================================================
# TRAIN / EVAL
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    for img, seq, y in loader:
        img = img.to(DEVICE, non_blocking=True)
        seq = seq.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits = model(img, seq)
        loss = criterion(logits, y)

        loss.backward()

        # Additional stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()

        total_loss += loss.item() * y.size(0)

        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(y.detach().cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    mcc = matthews_corrcoef(all_labels, all_preds)

    return avg_loss, acc, macro_f1, bal_acc, mcc


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    for img, seq, y in loader:
        img = img.to(DEVICE, non_blocking=True)
        seq = seq.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        logits = model(img, seq)
        loss = criterion(logits, y)

        total_loss += loss.item() * y.size(0)

        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    mcc = matthews_corrcoef(all_labels, all_preds)

    return (
        avg_loss,
        acc,
        macro_f1,
        bal_acc,
        mcc,
        np.array(all_labels),
        np.array(all_preds),
    )


# ============================================================
# LR SCHEDULER
# ============================================================

def get_warmup_cosine_lr(epoch: int, base_lr: float, total_epochs: int) -> float:
    """
    Epoch is 1-based.
    """
    min_lr = base_lr * MIN_LR_RATIO

    if epoch <= WARMUP_EPOCHS:
        return base_lr * epoch / max(1, WARMUP_EPOCHS)

    progress = (epoch - WARMUP_EPOCHS) / max(1, total_epochs - WARMUP_EPOCHS)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))

    return min_lr + (base_lr - min_lr) * cosine


def set_optimizer_lr(optimizer, lr: float):
    for group in optimizer.param_groups:
        group["lr"] = lr


# ============================================================
# PLOTS
# ============================================================

def plot_training_curves(history):
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    best_epoch = int(np.argmax(history["val_f1"])) + 1

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="Train loss")
    plt.plot(epochs, history["train_eval_loss"], label="Train eval loss")
    plt.plot(epochs, history["val_loss"], label="Validation loss")
    plt.axvline(best_epoch, linestyle="--", label=f"Best epoch: {best_epoch}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Full Byte2Image Fusion Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "byte2image_full_loss_curve.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_acc"], label="Train accuracy")
    plt.plot(epochs, history["train_eval_acc"], label="Train eval accuracy")
    plt.plot(epochs, history["val_acc"], label="Validation accuracy")
    plt.axvline(best_epoch, linestyle="--", label=f"Best epoch: {best_epoch}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Full Byte2Image Fusion Training and Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "byte2image_full_accuracy_curve.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_f1"], label="Train macro-F1")
    plt.plot(epochs, history["train_eval_f1"], label="Train eval macro-F1")
    plt.plot(epochs, history["val_f1"], label="Validation macro-F1")
    plt.axvline(best_epoch, linestyle="--", label=f"Best epoch: {best_epoch}")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.title("Full Byte2Image Fusion Training and Validation Macro-F1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "byte2image_full_f1_curve.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["lr"], label="Learning rate")
    plt.xlabel("Epoch")
    plt.ylabel("LR")
    plt.title("Learning Rate Schedule")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "byte2image_full_lr_curve.png", dpi=300)
    plt.close()


def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm)
    plt.title("Confusion Matrix — Full Byte2Image Fusion")
    plt.colorbar()

    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)

    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix_byte2image_full.png", dpi=300)
    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():
    set_seed(RANDOM_SEED)

    print(f"Device: {DEVICE}")
    print(f"Output directory: {OUTPUT_DIR}")

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

    print("\nSplit sizes:")
    print(f"Train: {len(train_df)}")
    print(f"Val:   {len(val_df)}")
    print(f"Test:  {len(test_df)}")

    train_dataset = Byte2ImageFullDataset(
        train_df,
        y_train,
        augment=True,
        byte_mask_prob=BYTE_MASK_PROB,
    )

    # Clean train eval dataset:
    # no augmentation, used only to measure real train performance.
    train_eval_dataset = Byte2ImageFullDataset(
        train_df,
        y_train,
        augment=False,
        byte_mask_prob=0.0,
    )

    val_dataset = Byte2ImageFullDataset(
        val_df,
        y_val,
        augment=False,
        byte_mask_prob=0.0,
    )

    test_dataset = Byte2ImageFullDataset(
        test_df,
        y_test,
        augment=False,
        byte_mask_prob=0.0,
    )

    pin_memory = DEVICE == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )

    train_eval_loader = DataLoader(
        train_eval_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )

    model = Byte2ImageFullFusionNet(
        num_classes=len(class_names),
    ).to(DEVICE)

    print("\nModel:")
    print(model)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nTotal params:     {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")

    criterion = nn.CrossEntropyLoss(
        label_smoothing=LABEL_SMOOTHING,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    history = {
        "train_loss": [],
        "train_acc": [],
        "train_f1": [],
        "train_bal_acc": [],
        "train_mcc": [],

        "train_eval_loss": [],
        "train_eval_acc": [],
        "train_eval_f1": [],
        "train_eval_bal_acc": [],
        "train_eval_mcc": [],

        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
        "val_bal_acc": [],
        "val_mcc": [],

        "lr": [],
    }

    best_val_f1 = -1.0
    best_state = None
    patience_counter = 0

    best_model_path = OUTPUT_DIR / "byte2image_full_best.pt"

    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        lr = get_warmup_cosine_lr(
            epoch=epoch,
            base_lr=LEARNING_RATE,
            total_epochs=EPOCHS,
        )
        set_optimizer_lr(optimizer, lr)

        train_loss, train_acc, train_f1, train_bal_acc, train_mcc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
        )

        train_eval_loss, train_eval_acc, train_eval_f1, train_eval_bal_acc, train_eval_mcc, _, _ = evaluate(
            model,
            train_eval_loader,
            criterion,
        )

        val_loss, val_acc, val_f1, val_bal_acc, val_mcc, _, _ = evaluate(
            model,
            val_loader,
            criterion,
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["train_f1"].append(train_f1)
        history["train_bal_acc"].append(train_bal_acc)
        history["train_mcc"].append(train_mcc)

        history["train_eval_loss"].append(train_eval_loss)
        history["train_eval_acc"].append(train_eval_acc)
        history["train_eval_f1"].append(train_eval_f1)
        history["train_eval_bal_acc"].append(train_eval_bal_acc)
        history["train_eval_mcc"].append(train_eval_mcc)

        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        history["val_bal_acc"].append(val_bal_acc)
        history["val_mcc"].append(val_mcc)

        history["lr"].append(lr)

        print(
            f"Epoch {epoch:02d} | "
            f"lr={lr:.6f} | "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, train_f1={train_f1:.4f} | "
            f"train_eval_acc={train_eval_acc:.4f}, train_eval_f1={train_eval_f1:.4f} | "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, val_f1={val_f1:.4f}, "
            f"val_bal_acc={val_bal_acc:.4f}, val_mcc={val_mcc:.4f}"
        )

        if val_f1 > best_val_f1 + MIN_DELTA:
            best_val_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            torch.save(best_state, best_model_path)
            print(f"  New best model saved. best_val_f1={best_val_f1:.4f}")
        else:
            patience_counter += 1
            print(f"  No significant improvement. patience={patience_counter}/{PATIENCE}")

        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

    train_time = time.time() - start_time

    if best_state is not None:
        model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    else:
        print("Warning: no best state was saved. Using final model.")

    test_loss, test_acc, test_f1, test_bal_acc, test_mcc, y_true, y_pred = evaluate(
        model,
        test_loader,
        criterion,
    )

    print("\n==============================")
    print("TEST RESULTS")
    print("==============================")
    print(f"Test loss:          {test_loss:.4f}")
    print(f"Accuracy:           {test_acc:.4f}")
    print(f"Macro-F1:           {test_f1:.4f}")
    print(f"Balanced Accuracy:  {test_bal_acc:.4f}")
    print(f"MCC:                {test_mcc:.4f}")
    print(f"Best val Macro-F1:  {best_val_f1:.4f}")
    print(f"Train time:         {train_time:.3f} sec")

    report_text = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0,
    )

    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    print("\nClassification report:")
    print(report_text)

    cm = confusion_matrix(y_true, y_pred)

    plot_training_curves(history)
    plot_confusion_matrix(cm, class_names)

    results = {
        "model": "Full Byte2Image Fusion Network",
        "description": "Sliding byte window + intra-byte n-gram image + wide convolution + ResNet-style image branch + raw byte branch",
        "accuracy": test_acc,
        "macro_f1": test_f1,
        "balanced_accuracy": test_bal_acc,
        "mcc": test_mcc,
        "test_loss": test_loss,
        "best_val_f1": best_val_f1,
        "train_time_sec": train_time,
        "classes": class_names,
        "epochs_trained": len(history["train_loss"]),

        "fragment_size": FRAGMENT_SIZE,
        "ngram": NGRAM,
        "shift_count": SHIFT_COUNT,
        "embed_dim": EMBED_DIM,

        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "label_smoothing": LABEL_SMOOTHING,
        "byte_mask_prob": BYTE_MASK_PROB,
        "warmup_epochs": WARMUP_EPOCHS,
        "patience": PATIENCE,
        "min_delta": MIN_DELTA,
        "total_params": total_params,
        "trainable_params": trainable_params,
    }

    with (OUTPUT_DIR / "byte2image_full_results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    with (OUTPUT_DIR / "byte2image_full_classification_report.json").open("w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=4)

    pd.DataFrame(history).to_csv(
        OUTPUT_DIR / "byte2image_full_training_history.csv",
        index=False,
    )

    print("\nSaved:")
    print(best_model_path)
    print(OUTPUT_DIR / "byte2image_full_results.json")
    print(OUTPUT_DIR / "byte2image_full_classification_report.json")
    print(OUTPUT_DIR / "byte2image_full_training_history.csv")
    print(OUTPUT_DIR / "byte2image_full_loss_curve.png")
    print(OUTPUT_DIR / "byte2image_full_accuracy_curve.png")
    print(OUTPUT_DIR / "byte2image_full_f1_curve.png")
    print(OUTPUT_DIR / "byte2image_full_lr_curve.png")
    print(OUTPUT_DIR / "confusion_matrix_byte2image_full.png")


if __name__ == "__main__":
    main()