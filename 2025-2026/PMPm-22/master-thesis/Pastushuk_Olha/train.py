import json
import os
import math
import sys
import time
import torch
import torch.nn as nn
import pandas as pd 
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from datetime import datetime 
from tqdm import tqdm  # Додано імпорт tqdm

from dataset import EmotionVADDataset
from backend.model import EmotionCRNN

# Config
CSV_TRAIN = "data/labels_train.csv"
CSV_VAL = "data/labels_val.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 256
EPOCHS = 40
LR = 2e-4
WEIGHT_DECAY = 1e-4  # L2 regularization
CLIP_GRAD_NORM = 5.0
CHECKPOINT_DIR = os.path.join('backend', 'checkpoints')
SAVE_PATH = "emotion_crnn_best.pth"
W_MSE = 0.2
W_CCC = 0.8

# Метрики
def rmse(a, b):
    return torch.sqrt(torch.mean((a-b)**2)).item()

def mae(a, b):
    return torch.mean(torch.abs(a-b)).item()

def pcc(a, b):
    a_mean = torch.mean(a, dim=0)
    b_mean = torch.mean(b, dim=0)
    cov = torch.sum((a - a_mean) * (b - b_mean), dim=0)
    std_a = torch.sqrt(torch.sum((a - a_mean)**2, dim=0)) + 1e-8
    std_b = torch.sqrt(torch.sum((b - b_mean)**2, dim=0))
    return (cov[0]/(std_a[0]*std_b[0]+1e-8)).item(), (cov[1]/(std_a[1]*std_b[1]+1e-8)).item(), (cov[2]/(std_a[2]*std_b[2]+1e-8)).item()

def ccc_loss(pred, target):
    # Розрахунок Concordance Correlation Coefficient для батчу
    eps = 1e-8
    p_mean = torch.mean(pred, dim=0)
    t_mean = torch.mean(target, dim=0)
    
    p_var = torch.var(pred, dim=0, unbiased=False)
    t_var = torch.var(target, dim=0, unbiased=False)
    
    # Коваріація
    cov = torch.mean((pred - p_mean) * (target - t_mean), dim=0)
    
    ccc = (2 * cov) / (p_var + t_var + torch.pow(p_mean - t_mean, 2) + eps)
    return 1 - torch.mean(ccc)

# Тренування однієї епохи
# Тренування однієї епохи
def train_one_epoch(model, loader, criterion, optimizer, epoch):
    model.train()
    total_loss = 0.0
    total_samples = 0 # Лічильник оброблених аудіозаписів

    pbar = tqdm(loader, desc=f"Epoch {epoch:03d} [Train]", leave=False)
    
    for mel, target in pbar:
        mel, target = mel.to(DEVICE), target.to(DEVICE)
        pred = model(mel)

        mse_loss = criterion(pred, target)
        ccc_val = ccc_loss(pred, target) # ccc_loss вже повертає (1 - CCC)
        
        # Комбінований Loss
        loss = W_MSE * mse_loss + W_CCC * ccc_val   

        optimizer.zero_grad()
        loss.backward()
        
        # Градієнтний кліппінг (важливо при змішуванні лоссів)
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM)
        
        optimizer.step()

        total_loss += loss.item() * mel.size(0)
        total_samples += mel.size(0)
        
    return total_loss / total_samples

# Валідація
def validate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    
    all_preds, all_targets = [], []

    pbar = tqdm(loader, desc="Validating", leave=False)
    
    with torch.no_grad():
        for mel, target in pbar:
            mel, target = mel.to(DEVICE), target.to(DEVICE)
            batch_size = mel.size(0)
            
            pred = model(mel)

            all_preds.append(pred.cpu())
            all_targets.append(target.cpu())

            mse_loss = criterion(pred, target)
            ccc_val = ccc_loss(pred, target)
            loss = W_MSE * mse_loss + W_CCC * ccc_val   
        
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    # Об'єднуємо всі результати для розрахунку метрик
    y_true_all = torch.cat(all_targets, dim=0)
    y_pred_all = torch.cat(all_preds, dim=0)

    # MAE
    mae_each = torch.mean(torch.abs(y_true_all - y_pred_all), dim=0)
    mae_v, mae_a, mae_d = mae_each[0].item(), mae_each[1].item(), mae_each[2].item()

    # PCC
    pcc_val, pcc_aro, pcc_dom = pcc(y_true_all, y_pred_all)

    print(f"Detailed MAE -> V: {mae_v:.4f}, A: {mae_a:.4f}, D: {mae_d:.4f}")
    
    # Повертаємо середній loss, розділений на точну кількість зразків
    final_loss = total_loss / total_samples
    return (final_loss, mae_each.mean().item(), 
            pcc_val, pcc_aro, pcc_dom, y_true_all, y_pred_all)
# Графіки метрик
def plot_metrics(history, save_path=None):
    plt.figure(figsize=(15,5))

    # Loss
    plt.subplot(1,3,1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Train vs Val Loss')
    plt.legend()

    # MAE
    plt.subplot(1,3,2)
    plt.plot(history['val_mae'], label='Val MAE', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.title('Validation MAE')
    plt.legend()

    # PCC
    plt.subplot(1,3,3)
    plt.plot(history['pcc_val'], label='PCC Valence')
    plt.plot(history['pcc_aro'], label='PCC Arousal')
    plt.plot(history['pcc_dom'], label='PCC Dominance')
    plt.xlabel('Epoch')
    plt.ylabel('PCC')
    plt.title('Validation PCC')
    plt.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
    plt.close()

def get_stats(loader):
    print("Calculating dataset stats...")

    total_sum = 0.0
    total_sq_sum = 0.0
    total_count = 0

    for mel, _ in tqdm(loader):
        total_sum += mel.sum().item()
        total_sq_sum += (mel ** 2).sum().item()
        total_count += mel.numel()

    mean = total_sum / total_count
    std = ((total_sq_sum / total_count) - mean**2) ** 0.5

    return mean, std

# Головна функція
def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs('logs', exist_ok=True) 
    os.makedirs('plots', exist_ok=True) 

    stats_path = os.path.join(CHECKPOINT_DIR, 'dataset_stats.json')
    
    # ПЕРЕВІРКА НАЯВНОСТІ ЗБЕРЕЖЕНИХ СТАТИСТИК
    if os.path.exists(stats_path):
        print(f"Loading existing stats from {stats_path}...")
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        mean = stats['mean']
        std = stats['std']
    else:
        # ОБЧИСЛЕННЯ, ЯКЩО ФАЙЛУ НЕМАЄ
        print("Stats file not found. Calculating dataset stats...")
        stats_ds = EmotionVADDataset(CSV_TRAIN, augment=False)
        stats_loader = DataLoader(stats_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        mean, std = get_stats(stats_loader)
        
        # ЗБЕРЕЖЕННЯ ДЛЯ НАСТУПНИХ ЗАПУСКІВ
        with open(stats_path, 'w') as f:
            json.dump({'mean': mean, 'std': std}, f)
        print(f"Stats saved to {stats_path}")

    log_file_path = os.path.join('logs', f'train_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    csv_logs = []

    # 2. ПІДГОТОВКА РОБОЧИХ ДАТАСЕТІВ
    train_ds = EmotionVADDataset(CSV_TRAIN, augment=True) # Тут аугментація дозволена
    val_ds = EmotionVADDataset(CSV_VAL, augment=False)

    train_ds.set_stats(mean, std)
    val_ds.set_stats(mean, std)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    print("Using device:", DEVICE)

    # Ініціалізація моделі та навчання
    model = EmotionCRNN().to(DEVICE)
    
    criterion = nn.MSELoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_val = math.inf
    history = {"train_loss": [], "val_loss": [], "val_mae": [], "pcc_val": [], "pcc_aro": [], "pcc_dom": []}

    for epoch in range(1, EPOCHS+1):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\nEpoch {epoch:03d} | Start time: {timestamp}") # Додано виведення часу початку епохи
        t0 = time.time()
        
        # Передаємо номер епохи для відображення в tqdm
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, epoch)
        val_loss, val_mae, pcc_val, pcc_aro, pcc_dom, _, _ = validate(model, val_loader, criterion)
        
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mae"].append(val_mae)
        history["pcc_val"].append(pcc_val)
        history["pcc_aro"].append(pcc_aro)
        history["pcc_dom"].append(pcc_dom)

        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]['lr']
        
        log_entry = {
            "epoch": epoch,
            "timestamp": timestamp,
            "train_loss": round(train_loss, 5),
            "val_loss": round(val_loss, 5),
            "val_mae": round(val_mae, 5),
            "pcc_valence": round(pcc_val, 5),
            "pcc_arousal": round(pcc_aro, 5),
            "pcc_dominance": round(pcc_dom, 5),
            "lr": f"{current_lr:.8f}",
            "time_sec": round(elapsed, 2)
        }
        csv_logs.append(log_entry)
        pd.DataFrame(csv_logs).to_csv(log_file_path, index=False) 

        # Основний принт залишається для фіксації результату епохи в консолі
        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val MAE: {val_mae:.4f} | PCC Val: {pcc_val:.4f} | PCC Aro: {pcc_aro:.4f} | PCC Dom: {pcc_dom:.4f} | Time: {elapsed:.1f}s | LR: {current_lr:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, SAVE_PATH))
            print(f"  --> Saved best model")

        if epoch % 5 == 0:
            plot_path = os.path.join('plots', f'metrics_epoch_{epoch}.png')
            plot_metrics(history, save_path=plot_path)
            print(f"  --> Interval plot saved to {plot_path}")

    plot_metrics(history, save_path='plots/final_metrics.png')
    print('Training finished. Best val loss:', best_val)
    print(f'Logs saved to {log_file_path}') 

if __name__ == '__main__':
    main()