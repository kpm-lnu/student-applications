# 03_evaluate.py
# ============================================================
# ЕТАП 5: Оцінка моделей та формування звіту
# ============================================================
# Запуск: python 03_evaluate.py
#
# Що робить цей скрипт:
#   1. Завантажує кожну збережену модель
#   2. Запускає inference на валідаційній вибірці
#   3. Будує Confusion Matrix для кожної моделі
#   4. Виводить Classification Report (Precision/Recall/F1)
#   5. Будує порівняльний графік усіх трьох моделей
#   6. Визначає та зберігає найкращу модель
# ============================================================

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ─────────────────────────────────────────────────────────
# КОНФІГУРАЦІЯ (має збігатися з 02_train.py!)
# ─────────────────────────────────────────────────────────
CONFIG = {
    "max_len":     128,
    "batch_size":  32,   # При оцінці можна збільшити — немає градієнтів
    "val_split":   0.20,
    "random_seed": 42,
}

MODELS_TO_EVALUATE = {
    "bert-base":  "models/bert-base",
    "distilbert": "models/distilbert",
    "roberta":    "models/roberta",
}

DATA_DIR    = "data"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Пристрій: {device}\n")


# ─────────────────────────────────────────────────────────
# КЛАС ДАТАСЕТУ (копія з 02_train.py)
# ─────────────────────────────────────────────────────────
from torch.utils.data import Dataset

class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        return {
            'input_ids':      encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label':          torch.tensor(int(self.labels[idx]), dtype=torch.long),
        }


# ─────────────────────────────────────────────────────────
# ФУНКЦІЯ ПОВНОГО INFERENCE
# ─────────────────────────────────────────────────────────
def get_predictions(model, loader, device) -> tuple:
    """
    Запускає повний inference та повертає передбачення та ймовірності.

    Повертає:
        true_labels  : справжні мітки (numpy array)
        pred_labels  : передбачені мітки (numpy array)
        pred_probs   : ймовірність класу 1 після softmax (для ROC-AUC)
    """
    model.eval()
    all_labels = []
    all_preds  = []
    all_probs  = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="  Inference", leave=False):
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits  = outputs.logits

            # softmax перетворює logits у ймовірності (сума = 1.0)
            # dim=1 означає softmax по класах (вісь класів)
            probs = torch.softmax(logits, dim=1)

            preds = torch.argmax(logits, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            # [:, 1] — ймовірність класу "справжня новина"
            all_probs.extend(probs[:, 1].cpu().numpy())

    return (
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs),
    )


# ─────────────────────────────────────────────────────────
# ФУНКЦІЯ ОЦІНКИ ОДНІЄЇ МОДЕЛІ
# ─────────────────────────────────────────────────────────
def evaluate_model(model_name: str, model_dir: str, val_texts, val_labels) -> dict:
    """
    Завантажує модель і обчислює всі метрики якості.

    Метрики:
    - Accuracy  : (TP+TN) / (TP+TN+FP+FN) — загальна точність
    - Precision : TP / (TP+FP) — з усіх передбачених "справжніх",
                  скільки справді справжніх? (важливо: уникнення хибних спрацювань)
    - Recall    : TP / (TP+FN) — з усіх справжніх новин,
                  скільки ми знайшли? (важливо: не пропустити фейки)
    - F1-Score  : 2 * (P * R) / (P + R) — гармонічне середнє P та R
    - ROC-AUC   : площа під кривою ROC — незалежна від порогу міра якості

    Args:
        model_name : назва моделі
        model_dir  : шлях до збереженої моделі
        val_texts  : тексти валідаційної вибірки
        val_labels : мітки валідаційної вибірки

    Returns:
        Словник з усіма метриками
    """
    print(f"\n── Оцінка: {model_name} ──")

    if not os.path.exists(model_dir):
        print(f"  ⚠ Модель не знайдено: {model_dir}")
        print(f"  Запустіть спочатку: python 02_train.py")
        return None

    # Завантаження збереженої моделі та токенізатора
    print(f"  Завантаження моделі з {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model     = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model     = model.to(device)

    # DataLoader
    dataset = NewsDataset(val_texts, val_labels, tokenizer, CONFIG['max_len'])
    loader  = DataLoader(dataset, batch_size=CONFIG['batch_size'],
                         shuffle=False, num_workers=2)

    # Inference
    true_labels, pred_labels, pred_probs = get_predictions(model, loader, device)

    # --- Обчислення метрик ---
    acc       = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, average='weighted')
    recall    = recall_score(true_labels, pred_labels, average='weighted')
    f1        = f1_score(true_labels, pred_labels, average='weighted')
    roc_auc   = roc_auc_score(true_labels, pred_probs)

    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  ROC-AUC   : {roc_auc:.4f}")

    # --- Детальний Classification Report ---
    print(f"\n  Classification Report:")
    report = classification_report(
        true_labels, pred_labels,
        target_names=['Фейк (0)', 'Справжня (1)'],
        digits=4,
    )
    print(report)

    # Зберігаємо звіт у файл
    report_path = f"{RESULTS_DIR}/03_report_{model_name}.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"=== Classification Report: {model_name} ===\n\n")
        f.write(report)
    print(f"  ✓ Звіт збережено: {report_path}")

    # --- Confusion Matrix ---
    cm = confusion_matrix(true_labels, pred_labels)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Confusion Matrix та ROC-крива: {model_name}", fontweight='bold')

    # Subplot 1: Confusion Matrix
    # annot=True — виводить числа у клітинках
    # fmt='d'    — формат: цілі числа
    # cmap       — кольорова схема
    sns.heatmap(
        cm, ax=axes[0],
        annot=True, fmt='d', cmap='Blues',
        xticklabels=['Фейк (прогноз)', 'Справжня (прогноз)'],
        yticklabels=['Фейк (реальна)', 'Справжня (реальна)'],
        linewidths=0.5,
    )
    axes[0].set_title("Confusion Matrix")
    axes[0].set_ylabel("Реальний клас")
    axes[0].set_xlabel("Передбачений клас")

    # Розшифровка:
    # TN (0,0): Фейк → Фейк (правильно)
    # FP (0,1): Фейк → Справжня (хибнопозитивний — небезпечно!)
    # FN (1,0): Справжня → Фейк (хибнонегативний)
    # TP (1,1): Справжня → Справжня (правильно)
    tn, fp, fn, tp = cm.ravel()
    axes[0].set_xlabel(
        f"TN={tn:,}  FP={fp:,}  FN={fn:,}  TP={tp:,}",
        fontsize=9, color='gray'
    )

    # Subplot 2: ROC-крива
    fpr, tpr, thresholds = roc_curve(true_labels, pred_probs)
    axes[1].plot(fpr, tpr, 'b-', linewidth=2,
                 label=f'ROC (AUC = {roc_auc:.4f})')
    axes[1].plot([0, 1], [0, 1], 'r--', linewidth=1, label='Випадкова модель')
    axes[1].fill_between(fpr, tpr, alpha=0.1)
    axes[1].set_title("ROC-крива")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].legend(fontsize=10)
    axes[1].grid(alpha=0.3)
    axes[1].set_xlim([0, 1])
    axes[1].set_ylim([0, 1.02])

    plt.tight_layout()
    plot_path = f"{RESULTS_DIR}/03_confusion_{model_name}.png"
    plt.savefig(plot_path, dpi=120, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Confusion Matrix збережено: {plot_path}")

    return {
        'model_name': model_name,
        'accuracy':   acc,
        'precision':  precision,
        'recall':     recall,
        'f1_score':   f1,
        'roc_auc':    roc_auc,
        'confusion_matrix': cm.tolist(),
        'model_dir':  model_dir,
    }


# ─────────────────────────────────────────────────────────
# ПОРІВНЯЛЬНИЙ ГРАФІК УСІХ МОДЕЛЕЙ
# ─────────────────────────────────────────────────────────
def plot_model_comparison(all_metrics: list):
    """Будує порівняльний бар-чарт для всіх метрик усіх моделей."""
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    metric_labels   = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']

    model_names = [r['model_name'] for r in all_metrics]
    x = np.arange(len(metrics_to_plot))
    width = 0.25  # Ширина одного стовпця
    colors = ['#2196F3', '#4CAF50', '#FF9800']

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (result, color) in enumerate(zip(all_metrics, colors)):
        values = [result[m] for m in metrics_to_plot]
        bars = ax.bar(x + i * width, values, width,
                      label=result['model_name'], color=color, alpha=0.85,
                      edgecolor='white')
        # Підписи над стовпцями
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003,
                f"{val:.3f}",
                ha='center', va='bottom', fontsize=8
            )

    ax.set_title("Порівняння моделей за метриками якості",
                 fontweight='bold', fontsize=13)
    ax.set_ylabel("Значення метрики")
    ax.set_xticks(x + width)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0.8, 1.02)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=1.0, color='gray', linestyle=':', linewidth=0.8)

    plt.tight_layout()
    path = f"{RESULTS_DIR}/03_model_comparison.png"
    plt.savefig(path, dpi=120, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n✓ Порівняльний графік збережено: {path}")


# ─────────────────────────────────────────────────────────
# ГОЛОВНИЙ БЛОК
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":

    print("=" * 60)
    print("ЕТАП 5: ОЦІНКА МОДЕЛЕЙ")
    print("=" * 60)

    # Завантаження даних
    prepared_path = f"{DATA_DIR}/prepared_dataset.csv"
    if not os.path.exists(prepared_path):
        print("Файл не знайдено. Запустіть: python 01_eda.py")
        sys.exit(1)

    df = pd.read_csv(prepared_path).dropna(subset=['text_combined', 'label'])
    texts  = df['text_combined'].tolist()
    labels = df['label'].astype(int).tolist()

    # Той самий розподіл, що і при навчанні (random_state=42 гарантує це)
    _, val_texts, _, val_labels = train_test_split(
        texts, labels,
        test_size=CONFIG['val_split'],
        random_state=CONFIG['random_seed'],
        stratify=labels,
    )

    print(f"Валідаційна вибірка: {len(val_texts):,} статей\n")

    # --- Оцінка всіх моделей ---
    all_metrics = []
    for model_name, model_dir in MODELS_TO_EVALUATE.items():
        result = evaluate_model(model_name, model_dir, val_texts, val_labels)
        if result:
            all_metrics.append(result)

    if not all_metrics:
        print("\n⚠ Жодної моделі не знайдено. Запустіть: python 02_train.py")
        sys.exit(1)

    # --- Порівняльний графік ---
    plot_model_comparison(all_metrics)

    # --- Зведена таблиця ---
    print("\n" + "=" * 60)
    print("ЗВЕДЕНА ТАБЛИЦЯ РЕЗУЛЬТАТІВ")
    print("=" * 60)
    df_results = pd.DataFrame(all_metrics)[
        ['model_name', 'accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    ]
    df_results.columns = ['Модель', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    print(df_results.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Визначення найкращої моделі за F1-Score
    best = max(all_metrics, key=lambda x: x['f1_score'])
    print(f"\n🏆 Найкраща модель: {best['model_name']}")
    print(f"   F1-Score : {best['f1_score']:.4f}")
    print(f"   ROC-AUC  : {best['roc_auc']:.4f}")

    # Збереження для використання в inference
    best_info = {'best_model': best['model_name'], 'model_dir': best['model_dir']}
    with open(f"{RESULTS_DIR}/best_model.json", 'w') as f:
        json.dump(best_info, f, indent=2)

    # Зведена таблиця у CSV
    df_results.to_csv(f"{RESULTS_DIR}/03_metrics_summary.csv", index=False)

    print(f"\n✓ Усі результати збережено у папці: {RESULTS_DIR}/")
    print("\n" + "=" * 60)
    print("Оцінка завершена! Переходьте до: python 04_inference.py")
    print("=" * 60)
