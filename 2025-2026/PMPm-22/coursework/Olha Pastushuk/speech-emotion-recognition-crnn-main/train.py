import os
import math
import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataset import EmotionVADDataset
from model import EmotionCRNN

# Config
CSV_TRAIN = "data/labels_train.csv"
CSV_VAL = "data/labels_val.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
EPOCHS = 40
LR = 1e-3
WEIGHT_DECAY = 1e-4  # L2 regularization
CLIP_GRAD_NORM = 5.0
SAVE_PATH = "emotion_crnn_best.pth"

# Метрики
def rmse(a, b):
    return torch.sqrt(torch.mean((a-b)**2)).item()

def mae(a, b):
    return torch.mean(torch.abs(a-b)).item()

def pcc(a, b):
    """
    Pearson Correlation Coefficient для кожного вектору
    a, b: torch.Tensor розміру [N, 2] (Valence, Arousal)
    повертає: tuple (pcc_valence, pcc_arousal)
    """
    a_mean = torch.mean(a, dim=0)
    b_mean = torch.mean(b, dim=0)
    cov = torch.sum((a - a_mean) * (b - b_mean), dim=0)
    std_a = torch.sqrt(torch.sum((a - a_mean)**2, dim=0))
    std_b = torch.sqrt(torch.sum((b - b_mean)**2, dim=0))
    return (cov[0]/(std_a[0]*std_b[0]+1e-8)).item(), (cov[1]/(std_a[1]*std_b[1]+1e-8)).item()

# Тренування однієї епохи
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    for mel, target in loader:
        mel, target = mel.to(DEVICE), target.to(DEVICE)

        pred = model(mel)
        loss = criterion(pred, target)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM)
        optimizer.step()

        total_loss += loss.item() * mel.size(0)
    return total_loss / len(loader.dataset)

# Валідація з MSE, MAE та PCC
def validate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    y_true_all, y_pred_all = [], []

    with torch.no_grad():
        for mel, target in loader:
            mel, target = mel.to(DEVICE), target.to(DEVICE)
            pred = model(mel)

            loss = criterion(pred, target)
            total_loss += loss.item() * mel.size(0)
            total_mae += mae(pred, target) * mel.size(0)

            y_true_all.append(target.cpu())
            y_pred_all.append(pred.cpu())

    y_true_all = torch.cat(y_true_all, dim=0)
    y_pred_all = torch.cat(y_pred_all, dim=0)

    pcc_val, pcc_aro = pcc(y_true_all, y_pred_all)

    return total_loss / len(loader.dataset), total_mae / len(loader.dataset), pcc_val, pcc_aro, y_true_all, y_pred_all

# Графіки метрик
def plot_metrics(train_losses, val_losses, val_maes, pcc_vals, pcc_aros):
    plt.figure(figsize=(15,5))

    # Loss
    plt.subplot(1,3,1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Train vs Val Loss')
    plt.legend()

    # MAE
    plt.subplot(1,3,2)
    plt.plot(val_maes, label='Val MAE', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.title('Validation MAE')
    plt.legend()

    # PCC
    plt.subplot(1,3,3)
    plt.plot(pcc_vals, label='PCC Valence')
    plt.plot(pcc_aros, label='PCC Arousal')
    plt.xlabel('Epoch')
    plt.ylabel('PCC')
    plt.title('Validation PCC')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Головна функція
def main():
    os.makedirs('checkpoints', exist_ok=True)

    train_ds = EmotionVADDataset(CSV_TRAIN)
    val_ds = EmotionVADDataset(CSV_VAL)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    print("Using device:", DEVICE)

    model = EmotionCRNN().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_val = math.inf
    train_losses, val_losses, val_maes, pcc_vals, pcc_aros = [], [], [], [], []

    for epoch in range(1, EPOCHS+1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_mae, pcc_val, pcc_aro, _, _ = validate(model, val_loader, criterion)
        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_maes.append(val_mae)
        pcc_vals.append(pcc_val)
        pcc_aros.append(pcc_aro)

        elapsed = time.time() - t0
        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val MAE: {val_mae:.4f} | PCC Val: {pcc_val:.4f} | PCC Aro: {pcc_aro:.4f} | Time: {elapsed:.1f}s | LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Save best model
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), os.path.join('checkpoints', SAVE_PATH))

    print('Training finished. Best val loss:', best_val)
    plot_metrics(train_losses, val_losses, val_maes, pcc_vals, pcc_aros)

if __name__ == '__main__':
    main()
