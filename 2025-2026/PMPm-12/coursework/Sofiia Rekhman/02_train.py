# 02_train.py
# ============================================================
# ЕТАП 3–4: Навчання трансформерних моделей (Fine-tuning)
# ============================================================
# Запуск: python 02_train.py
#
# Що робить цей скрипт:
#   1. Завантажує підготовлений датасет
#   2. Ділить на Train/Validation (80/20)
#   3. Послідовно навчає BERT, DistilBERT та RoBERTa
#   4. Для кожної моделі: токенізація → DataLoader → цикл навчання
#   5. Зберігає ваги найкращої епохи кожної моделі
#   6. Будує криві навчання (loss + accuracy по епохах)
# ============================================================

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ─────────────────────────────────────────────────────────
# КОНФІГУРАЦІЯ ГІПЕРПАРАМЕТРІВ
# ─────────────────────────────────────────────────────────
CONFIG = {
    # Максимальна довжина токенів.
    # BERT має абсолютний ліміт у 512 токенів.
    # 128 — компроміс: охоплює ~70% статей повністю,
    # значно зменшує пам'ять та час навчання.
    "max_len": 128,

    # Розмір батчу: скільки прикладів обробляється одночасно.
    # Більший батч = стабільніший градієнт, але більше VRAM.
    # 16 — безпечне значення для GPU 6–8 GB.
    # Якщо є CUDA OOM помилка → варто зменшити до 8.
    "batch_size": 16,

    # Кількість проходів по всьому датасету.
    # 3–5 епох — стандарт для fine-tuning BERT.
    # Більше → ризик overfitting.
    "num_epochs": 3,

    # Швидкість навчання: 2e-5 є найбільш рекомендованою
    # для fine-tuning трансформерів (з оригінальної статті BERT).
    # Діапазон для експериментів: 1e-5 до 5e-5.
    "learning_rate": 2e-5,

    # Вага L2-регуляризації для AdamW.
    # Допомагає уникати overfitting у великих моделях.
    "weight_decay": 0.01,

    # Частка навчальних кроків для "розігріву" learning rate.
    # LR поступово зростає з 0 до learning_rate,
    # потім лінійно спадає — це стабілізує навчання.
    "warmup_ratio": 0.1,

    # Відтворюваність результатів
    "random_seed": 42,

    # Частка даних для валідації
    "val_split": 0.20,
}

# Три моделі для порівняння
# Ключ: назва для збереження, значення: HuggingFace model ID
MODELS = {
    "bert-base": {
        "model_id": "bert-base-uncased",
        "description": "BERT Base — базова лінія (110M параметрів)",
    },
    "distilbert": {
        "model_id": "distilbert-base-uncased",
        "description": "DistilBERT — 40% менший, 60% швидший за BERT",
    },
    "roberta": {
        "model_id": "roberta-base",
        "description": "RoBERTa — оптимізований BERT (125M параметрів)",
    },
}

# Шляхи
DATA_DIR    = "data"
MODELS_DIR  = "models"
RESULTS_DIR = "results"

for d in [MODELS_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ─────────────────────────────────────────────────────────
# ВИЗНАЧЕННЯ ПРИСТРОЮ (GPU або CPU)
# ─────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Пристрій: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("⚠ GPU не знайдено. Навчання на CPU буде дуже повільним!")
    print("  Рекомендовано: Google Colab (безкоштовний GPU)")

# Фіксуємо seed для відтворюваності
torch.manual_seed(CONFIG["random_seed"])
np.random.seed(CONFIG["random_seed"])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(CONFIG["random_seed"])

print(f"\nКонфігурація: {json.dumps(CONFIG, indent=2, ensure_ascii=False)}")


# ─────────────────────────────────────────────────────────
# КЛАС ДАТАСЕТУ PyTorch
# ─────────────────────────────────────────────────────────
class NewsDataset(Dataset):
    """
    Кастомний датасет для PyTorch DataLoader.

    Успадковується від torch.utils.data.Dataset та реалізує
    два обов'язкові методи: __len__ та __getitem__.

    Токенізація відбувається "ліниво" (on-the-fly) у __getitem__,
    а не вся одразу, щоб заощадити оперативну пам'ять.
    """

    def __init__(self, texts: list, labels: list, tokenizer, max_len: int):
        """
        Args:
            texts    : список рядків (очищені тексти статей)
            labels   : список цілих чисел (0 або 1)
            tokenizer: завантажений токенізатор HuggingFace
            max_len  : максимальна довжина послідовності токенів
        """
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        """Повертає кількість прикладів у датасеті."""
        return len(self.texts)

    def __getitem__(self, idx: int):
        """
        Повертає один токенізований приклад.

        tokenizer() робить:
        1. Розбиває текст на субслова (subword tokenization)
        2. Додає спеціальні токени: [CLS] на початку, [SEP] в кінці
        3. Якщо текст довший max_len — обрізає (truncation=True)
        4. Якщо коротший — доповнює нулями (padding='max_length')
        5. Повертає:
           - input_ids      : числові ID токенів
           - attention_mask : 1 для реальних токенів, 0 для padding

        Returns:
            dict з тензорами для моделі + мітка класу
        """
        text  = str(self.texts[idx])
        label = int(self.labels[idx])

        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',   # Доповнює коротші послідовності
            truncation=True,         # Обрізає довші послідовності
            return_tensors='pt',     # Повертає PyTorch тензори
        )

        return {
            # squeeze(0) видаляє зайвий вимір батчу від tokenizer
            'input_ids':      encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label':          torch.tensor(label, dtype=torch.long),
        }


# ─────────────────────────────────────────────────────────
# ФУНКЦІЯ НАВЧАННЯ ОДНОЇ ЕПОХИ
# ─────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, scheduler, device) -> tuple:
    """
    Виконує одну епоху навчання.

    Алгоритм:
    1. model.train() — вмикає dropout, batch normalization у режим навчання
    2. Для кожного батчу:
       a) Forward pass: модель обчислює logits та loss
       b) Backward pass: обчислення градієнтів (loss.backward())
       c) Clip gradients: обрізання великих градієнтів (стабільність)
       d) Optimizer step: оновлення ваг у напрямку від градієнту
       e) Scheduler step: коригування learning rate за розкладом
       f) Zero gradients: обнулення для наступного батчу

    Returns:
        (avg_loss, accuracy) за епоху
    """
    model.train()
    total_loss    = 0.0
    correct_preds = 0
    total_preds   = 0

    # tqdm додає прогрес-бар у консоль
    progress = tqdm(loader, desc="  Навчання", leave=False)

    for batch in progress:
        # Переміщення тензорів на GPU/CPU
        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels         = batch['label'].to(device)

        # Скидання градієнтів від попереднього батчу
        optimizer.zero_grad()

        # Forward pass
        # outputs містить: loss, logits (і hidden states якщо запитати)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,         # Якщо передати labels — модель сама рахує CrossEntropyLoss
        )

        loss   = outputs.loss
        logits = outputs.logits   # "сирі" оцінки класів до softmax

        # Backward pass — обчислення градієнтів
        loss.backward()

        # Gradient clipping: обмежує норму градієнтів до 1.0.
        # Без цього великі градієнти можуть "взірвати" навчання.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Оновлення ваг
        optimizer.step()

        # Оновлення learning rate за розкладом
        scheduler.step()

        # Підрахунок точності
        # argmax(dim=1) → клас з найвищим logit для кожного прикладу
        preds          = torch.argmax(logits, dim=1)
        correct_preds += (preds == labels).sum().item()
        total_preds   += labels.size(0)
        total_loss    += loss.item()

        # Оновлення прогрес-бару
        progress.set_postfix({
            'loss': f"{loss.item():.4f}",
            'acc':  f"{correct_preds/total_preds:.3f}"
        })

    avg_loss = total_loss / len(loader)
    accuracy = correct_preds / total_preds
    return avg_loss, accuracy


# ─────────────────────────────────────────────────────────
# ФУНКЦІЯ ВАЛІДАЦІЇ
# ─────────────────────────────────────────────────────────
def evaluate(model, loader, device) -> tuple:
    """
    Оцінює модель на валідаційній вибірці.

    ВАЖЛИВО: torch.no_grad() вимикає підрахунок градієнтів —
    це зменшує споживання пам'яті та прискорює обчислення.
    model.eval() вимикає dropout (усі нейрони активні).

    Returns:
        (avg_loss, accuracy)
    """
    model.eval()
    total_loss    = 0.0
    correct_preds = 0
    total_preds   = 0

    with torch.no_grad():
        progress = tqdm(loader, desc="  Валідація", leave=False)
        for batch in progress:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['label'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            preds          = torch.argmax(outputs.logits, dim=1)
            correct_preds += (preds == labels).sum().item()
            total_preds   += labels.size(0)
            total_loss    += outputs.loss.item()

    avg_loss = total_loss / len(loader)
    accuracy = correct_preds / total_preds
    return avg_loss, accuracy


# ─────────────────────────────────────────────────────────
# ФУНКЦІЯ НАВЧАННЯ ОДНІЄЇ МОДЕЛІ (повний цикл)
# ─────────────────────────────────────────────────────────
def train_model(
    model_name: str,
    model_id:   str,
    train_texts: list,
    train_labels: list,
    val_texts:   list,
    val_labels:  list,
    config:      dict,
) -> dict:
    """
    Повний цикл навчання однієї моделі.

    Args:
        model_name  : назва для збереження (напр. "bert-base")
        model_id    : HuggingFace model ID (напр. "bert-base-uncased")
        train_texts : тексти для навчання
        train_labels: мітки для навчання
        val_texts   : тексти для валідації
        val_labels  : мітки для валідації
        config      : словник гіперпараметрів

    Returns:
        Словник з результатами та шляхом до збереженої моделі
    """
    print(f"\n{'='*60}")
    print(f"Навчання: {model_name}")
    print(f"  HuggingFace ID: {model_id}")
    print(f"{'='*60}")

    start_time = time.time()

    # --- Крок 1: Завантаження токенізатора ---
    # Токенізатор специфічний для кожної моделі:
    # BERT та DistilBERT використовують WordPiece,
    # RoBERTa — BPE (Byte-Pair Encoding).
    print("  Завантаження токенізатора...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # --- Крок 2: Створення датасетів ---
    train_dataset = NewsDataset(train_texts, train_labels, tokenizer, config['max_len'])
    val_dataset   = NewsDataset(val_texts,   val_labels,   tokenizer, config['max_len'])

    # --- Крок 3: DataLoader ---
    # DataLoader автоматично формує батчі, перемішує дані (train),
    # завантажує дані у паралельних потоках (num_workers).
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,          # Перемішування для кращого навчання
        num_workers=2,         # Паралельне завантаження даних
        pin_memory=True,       # Пришвидшення передачі на GPU
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'] * 2,  # Валідація: подвійний батч (немає градієнтів)
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # --- Крок 4: Завантаження моделі ---
    # AutoModelForSequenceClassification додає поверх базової
    # трансформерної моделі лінійний класифікатор (Linear + Dropout).
    # num_labels=2: бінарна класифікація (справжня/фейк).
    print("  Завантаження моделі...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=2,           # 0 = фейк, 1 = справжня
        ignore_mismatched_sizes=True,  # На випадок різних версій
    )
    model = model.to(device)

    # Кількість параметрів
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Параметрів: {n_params:,}")

    # --- Крок 5: Оптимізатор AdamW ---
    # AdamW = Adam з weight decay (L2-регуляризація).
    # Важливо: НЕ застосовуються weight decay до bias та LayerNorm —
    # це стандартна практика для трансформерів.
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters()
                      if not any(nd in n for nd in no_decay)],
            'weight_decay': config['weight_decay'],
        },
        {
            'params': [p for n, p in model.named_parameters()
                      if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config['learning_rate'])

    # --- Крок 6: Learning Rate Scheduler ---
    # Лінійний warmup: LR зростає з 0 до learning_rate за перші N кроків,
    # потім лінійно спадає до 0 — це покращує стабільність навчання.
    total_steps  = len(train_loader) * config['num_epochs']
    warmup_steps = int(total_steps * config['warmup_ratio'])

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    print(f"  Навчальних кроків: {total_steps:,} | Warmup: {warmup_steps:,}")

    # --- Крок 7: Цикл навчання ---
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss':   [], 'val_acc':   [],
    }

    best_val_acc   = 0.0
    best_epoch     = 0
    best_model_dir = os.path.join(MODELS_DIR, model_name)

    for epoch in range(1, config['num_epochs'] + 1):
        print(f"\n  Епоха {epoch}/{config['num_epochs']}")

        # Навчання
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scheduler, device)

        # Валідація
        val_loss, val_acc = evaluate(model, val_loader, device)

        # Логування
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"  Train  → Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        print(f"  Val    → Loss: {val_loss:.4f}   | Acc: {val_acc:.4f}")

        # Виявлення overfitting
        if epoch > 1:
            prev_val_loss = history['val_loss'][-2]
            if val_loss > prev_val_loss * 1.05:  # Зростання > 5%
                print("  ⚠ Увага: val_loss зростає — можливий початок overfitting!")

        # Збереження найкращої моделі (за val_acc)
        if val_acc > best_val_acc:
            best_val_acc   = val_acc
            best_epoch     = epoch
            # Зберігаємо модель та токенізатор разом
            model.save_pretrained(best_model_dir)
            tokenizer.save_pretrained(best_model_dir)
            print(f"  ✓ Нова найкраща модель збережена (epoch={epoch}, val_acc={val_acc:.4f})")

    # Підрахунок часу навчання
    elapsed = time.time() - start_time
    elapsed_str = f"{int(elapsed // 60)}хв {int(elapsed % 60)}с"

    print(f"\n  Навчання завершено за {elapsed_str}")
    print(f"  Найкраща епоха: {best_epoch} (val_acc={best_val_acc:.4f})")

    # --- Крок 8: Графік кривих навчання ---
    _plot_training_curves(model_name, history, config['num_epochs'])

    return {
        'model_name':    model_name,
        'best_val_acc':  best_val_acc,
        'best_epoch':    best_epoch,
        'elapsed_sec':   elapsed,
        'elapsed_str':   elapsed_str,
        'history':       history,
        'model_dir':     best_model_dir,
    }


def _plot_training_curves(model_name: str, history: dict, num_epochs: int):
    """Будує та зберігає криві навчання (loss + accuracy)."""
    epochs = range(1, num_epochs + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Криві навчання: {model_name}", fontweight='bold')

    # Loss
    ax1.plot(epochs, history['train_loss'], 'b-o', label='Train Loss', markersize=5)
    ax1.plot(epochs, history['val_loss'],   'r-o', label='Val Loss',   markersize=5)
    ax1.set_title("Функція втрат")
    ax1.set_xlabel("Епоха")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Accuracy
    ax2.plot(epochs, history['train_acc'], 'b-o', label='Train Acc', markersize=5)
    ax2.plot(epochs, history['val_acc'],   'r-o', label='Val Acc',   markersize=5)
    ax2.set_title("Точність")
    ax2.set_xlabel("Епоха")
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0.5, 1.0)
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    path = f"{RESULTS_DIR}/02_training_{model_name}.png"
    plt.savefig(path, dpi=120, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Криві навчання збережено: {path}")


# ─────────────────────────────────────────────────────────
# ГОЛОВНИЙ БЛОК
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":

    # --- Завантаження підготовленого датасету ---
    prepared_path = f"{DATA_DIR}/prepared_dataset.csv"
    if not os.path.exists(prepared_path):
        print("Файл prepared_dataset.csv не знайдено.")
        print("Спочатку запустіть: python 01_eda.py")
        sys.exit(1)

    print("Завантаження підготовленого датасету...")
    df = pd.read_csv(prepared_path)
    print(f"  Завантажено {len(df):,} статей")

    # Видалення рядків з NaN
    df = df.dropna(subset=['text_combined', 'label'])
    texts  = df['text_combined'].tolist()
    labels = df['label'].astype(int).tolist()

    # --- Розподіл на Train/Validation (80/20) ---
    # stratify=labels: зберігає пропорцію класів в обох вибірках.
    # Без stratify може виникнути випадковий дисбаланс.
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels,
        test_size=CONFIG['val_split'],
        random_state=CONFIG['random_seed'],
        stratify=labels,
    )

    print(f"\nРозподіл даних:")
    print(f"  Train: {len(train_texts):,} статей")
    print(f"  Val:   {len(val_texts):,} статей")

    # --- Навчання всіх трьох моделей ---
    all_results = []

    for model_name, model_info in MODELS.items():
        result = train_model(
            model_name=model_name,
            model_id=model_info['model_id'],
            train_texts=train_texts,
            train_labels=train_labels,
            val_texts=val_texts,
            val_labels=val_labels,
            config=CONFIG,
        )
        all_results.append(result)

    # --- Зведена таблиця результатів ---
    print("\n" + "=" * 60)
    print("ЗВЕДЕНА ТАБЛИЦЯ РЕЗУЛЬТАТІВ НАВЧАННЯ")
    print("=" * 60)
    print(f"{'Модель':<15} {'Val Acc':>10} {'Найкраща епоха':>16} {'Час навчання':>14}")
    print("-" * 60)
    for r in all_results:
        print(f"{r['model_name']:<15} {r['best_val_acc']:>9.4f}  {r['best_epoch']:>14}  {r['elapsed_str']:>12}")

    # Збереження результатів навчання
    results_path = f"{RESULTS_DIR}/training_summary.json"
    save_results = [{k: v for k, v in r.items() if k != 'history'} for r in all_results]
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(save_results, f, ensure_ascii=False, indent=2)
    print(f"\n✓ Результати збережено: {results_path}")

    print("\n" + "=" * 60)
    print("Навчання завершено! Переходьте до: python 03_evaluate.py")
    print("=" * 60)
