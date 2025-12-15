import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_loss(train_losses, val_losses):
    """Графік зміни функції втрат під час тренування"""
    plt.figure(figsize=(8,5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_va_scatter(y_true, y_pred):
    """Розсіювання передбачених та істинних значень VA"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    plt.figure(figsize=(8,6))
    plt.scatter(y_true[:,0], y_true[:,1], c='blue', alpha=0.6, label='True VA')
    plt.scatter(y_pred[:,0], y_pred[:,1], c='red', alpha=0.6, label='Predicted VA')
    plt.xlabel("Valence")
    plt.ylabel("Arousal")
    plt.title("True vs Predicted VA")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_va_trend(y_true, y_pred):
    """Графік по часу для прикладу кількох зразків"""
    num_samples = min(5, len(y_true))
    plt.figure(figsize=(10,6))
    for i in range(num_samples):
        plt.plot(y_true[i,:,0], label=f"True Valence {i}", linestyle='-')
        plt.plot(y_pred[i,:,0], label=f"Pred Valence {i}", linestyle='--')
        plt.plot(y_true[i,:,1], label=f"True Arousal {i}", linestyle=':')
        plt.plot(y_pred[i,:,1], label=f"Pred Arousal {i}", linestyle='-.')
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.title("Valence/Arousal Trend over Time")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_pcc(pcc_vals, pcc_aros):
    """Графік зміни коефіцієнта кореляції Пірсона по епохах"""
    plt.figure(figsize=(8,5))
    plt.plot(pcc_vals, label="PCC Valence")
    plt.plot(pcc_aros, label="PCC Arousal")
    plt.xlabel("Epoch")
    plt.ylabel("PCC")
    plt.title("Pearson Correlation Coefficient over Epochs")
    plt.ylim(-1, 1)
    plt.legend()
    plt.grid(True)
    plt.show()
