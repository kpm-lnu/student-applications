import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D # Потрібно для 3D графіка

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

def plot_vad_scatter(y_true, y_pred):
    """3D Розсіювання передбачених та істинних значень VAD"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Малюємо істинні значення синім, передбачені - червоним
    ax.scatter(y_true[:,0], y_true[:,1], y_true[:,2], c='blue', alpha=0.5, label='True VAD')
    ax.scatter(y_pred[:,0], y_pred[:,1], y_pred[:,2], c='red', alpha=0.5, label='Predicted VAD')

    ax.set_xlabel("Valence")
    ax.set_ylabel("Arousal")
    ax.set_zlabel("Dominance")
    ax.set_title("True vs Predicted VAD Space")
    ax.legend()
    plt.show()

def plot_vad_trend(y_true, y_pred):
    """Графік по часу для порівняння Valence, Arousal та Dominance"""
    num_samples = min(3, len(y_true)) # Візьмемо 3 зразки для наочності
    plt.figure(figsize=(12, 8))
    
    for i in range(num_samples):
        plt.subplot(num_samples, 1, i+1)
        # Valence
        plt.plot(y_true[i, 0], label=f"True Val {i}", color='blue', linestyle='-')
        plt.plot(y_pred[i, 0], label=f"Pred Val {i}", color='blue', linestyle='--')
        # Arousal
        plt.plot(y_true[i, 1], label=f"True Aro {i}", color='red', linestyle='-')
        plt.plot(y_pred[i, 1], label=f"Pred Aro {i}", color='red', linestyle='--')
        # Dominance
        plt.plot(y_true[i, 2], label=f"True Dom {i}", color='green', linestyle='-')
        plt.plot(y_pred[i, 2], label=f"Pred Dom {i}", color='green', linestyle='--')
        
        plt.ylabel("Value")
        plt.legend(loc='upper right', fontsize='small')
        if i == 0:
            plt.title("VAD Trend Comparison")

    plt.xlabel("Sample Index")
    plt.tight_layout()
    plt.show()

def plot_pcc(pcc_vals, pcc_aros, pcc_doms):
    """Графік зміни коефіцієнта Пірсона для всіх 3 вимірів"""
    plt.figure(figsize=(8,5))
    plt.plot(pcc_vals, label="PCC Valence", color='blue')
    plt.plot(pcc_aros, label="PCC Arousal", color='red')
    plt.plot(pcc_doms, label="PCC Dominance", color='green') # Додано третю лінію
    
    plt.xlabel("Epoch")
    plt.ylabel("PCC")
    plt.title("Pearson Correlation Coefficient over Epochs")
    plt.ylim(-1, 1)
    plt.legend()
    plt.grid(True)
    plt.show()