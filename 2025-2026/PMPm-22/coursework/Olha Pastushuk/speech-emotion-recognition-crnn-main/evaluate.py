import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from dataset import EmotionVADDataset
from model import EmotionCRNN

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CSV_VAL = "data/labels_val.csv"
CHECKPOINT_PATH = "checkpoints/emotion_crnn_best.pth"
BATCH_SIZE = 16

# PCC
def pcc(a, b):
    a_mean = torch.mean(a, dim=0)
    b_mean = torch.mean(b, dim=0)
    cov = torch.sum((a - a_mean) * (b - b_mean), dim=0)
    std_a = torch.sqrt(torch.sum((a - a_mean)**2, dim=0))
    std_b = torch.sqrt(torch.sum((b - b_mean)**2, dim=0))
    return (cov[0]/(std_a[0]*std_b[0]+1e-8)).item(), (cov[1]/(std_a[1]*std_b[1]+1e-8)).item()

def evaluate():
    val_ds = EmotionVADDataset(CSV_VAL, augment=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    model = EmotionCRNN().to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()

    preds, targets = [], []
    with torch.no_grad():
        for mel, va in val_loader:
            mel, va = mel.to(DEVICE), va.to(DEVICE)
            out = model(mel)
            preds.append(out.cpu())
            targets.append(va.cpu())

    preds = torch.cat(preds)
    targets = torch.cat(targets)

    # PCC
    pcc_val, pcc_aro = pcc(targets, preds)
    print(f"PCC Valence: {pcc_val:.4f}, PCC Arousal: {pcc_aro:.4f}")

    # Scatter plot Valence
    plt.figure(figsize=(6,6))
    plt.scatter(targets[:,0], preds[:,0], alpha=0.5)
    plt.xlabel("True Valence")
    plt.ylabel("Predicted Valence")
    plt.title("Valence: True vs Predicted")
    plt.show()

    # Scatter plot Arousal
    plt.figure(figsize=(6,6))
    plt.scatter(targets[:,1], preds[:,1], alpha=0.5)
    plt.xlabel("True Arousal")
    plt.ylabel("Predicted Arousal")
    plt.title("Arousal: True vs Predicted")
    plt.show()

if __name__ == "__main__":
    evaluate()
