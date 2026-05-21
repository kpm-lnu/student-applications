# colab_notebook.py
# ============================================================
# ВЕРСІЯ ДЛЯ GOOGLE COLAB
# ============================================================
# Можна скопіювати цей код у нову Colab-нотатку.
# Runtime → Change runtime type → GPU (T4)
#
# Colab дає безкоштовний GPU (NVIDIA T4 ~16GB VRAM).
# Навчання однієї моделі займе ~20–40 хвилин.
# ============================================================

# ── Крок 1: Встановлення бібліотек ────────────────────────
# (виконанати у першій клітинці Colab)
"""
!pip install transformers==4.37.2 datasets==2.16.1 torch==2.1.2 \
             scikit-learn==1.3.2 seaborn==0.13.1 wordcloud==1.9.3 \
             accelerate==0.26.1 tqdm -q
"""

# ── Крок 2: Завантаження датасету ─────────────────────────
"""
# Варіант A: через Kaggle API
!pip install kaggle -q
from google.colab import files
files.upload()  # Завантажте kaggle.json
!mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d rahulogoel/isot-fake-news-dataset
!unzip isot-fake-news-dataset.zip -d data/

# Варіант B: вручну
# Завантажте на свій ПК і завантажте через:
from google.colab import files
uploaded = files.upload()  # Виберіть True.csv та Fake.csv
import os; os.makedirs('data', exist_ok=True)
import shutil
for fname in uploaded:
    shutil.move(fname, f'data/{fname}')
"""

# ── Крок 3: Клонування структури проєкту ─────────────────
"""
# Завантажити окремі .py файли через files.upload()
"""

# ── Крок 4: Запуск скриптів ───────────────────────────────
"""
%run 01_eda.py
%run 00_classical_models.py
%run 02_train.py
%run 03_evaluate.py
"""

# ── Крок 5: Збереження результатів на Google Drive ────────
"""
from google.colab import drive
drive.mount('/content/drive')

import shutil
shutil.copytree('models_classical',  '/content/drive/MyDrive/fake_news_models_classical',  dirs_exist_ok=True)
shutil.copytree('results_classical', '/content/drive/MyDrive/fake_news_results_classical', dirs_exist_ok=True)
shutil.copytree('models',  '/content/drive/MyDrive/fake_news_models',  dirs_exist_ok=True)
shutil.copytree('results', '/content/drive/MyDrive/fake_news_results', dirs_exist_ok=True)
print("✓ Збережено на Google Drive!")
"""

# ── Інлайн-версія для швидкого тестування ─────────────────
# Якщо хочете запустити все в одній нотатці без файлів:

import os
import re
import time
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tqdm.notebook import tqdm  # Версія для Jupyter/Colab
import matplotlib.pyplot as plt
import seaborn as sns

print("✓ Всі бібліотеки імпортовано")
print(f"  PyTorch: {torch.__version__}")
print(f"  GPU: {'✓ ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else '✗ (CPU режим)'}")

# --- Конфігурація ---
CONFIG = {
    "max_len":       128,
    "batch_size":    16,
    "num_epochs":    3,
    "learning_rate": 2e-5,
    "weight_decay":  0.01,
    "warmup_ratio":  0.1,
    "random_seed":   42,
    "val_split":     0.20,
    # Для Colab: навчаємо лише DistilBERT для економії часу демонстрації
    "demo_model_id": "distilbert-base-uncased",
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Допоміжні функції ---
def remove_reuters_tag(text):
    text = re.sub(r'^.*?\(Reuters\)\s*-\s*', '', str(text), flags=re.IGNORECASE)
    text = re.sub(r'^[A-Z\s,]+\s*-\s*', '', text)
    return text.strip()

def prepare_text(title, text):
    title = remove_reuters_tag(str(title))
    body  = remove_reuters_tag(str(text))
    return f"{title} [SEP] {body}"

class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts, self.labels = texts, labels
        self.tokenizer, self.max_len = tokenizer, max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        enc = self.tokenizer(str(self.texts[idx]), max_length=self.max_len,
                             padding='max_length', truncation=True, return_tensors='pt')
        return {'input_ids': enc['input_ids'].squeeze(0),
                'attention_mask': enc['attention_mask'].squeeze(0),
                'label': torch.tensor(int(self.labels[idx]), dtype=torch.long)}

# --- Завантаження даних ---
print("\nЗавантаження датасету...")
df_true = pd.read_csv("data/True.csv"); df_true['label'] = 1
df_fake = pd.read_csv("data/Fake.csv"); df_fake['label'] = 0
df = pd.concat([df_true, df_fake]).sample(frac=1, random_state=42).reset_index(drop=True)
df['text_combined'] = df.apply(lambda r: prepare_text(r['title'], r['text']), axis=1)
df = df.dropna(subset=['text_combined'])
print(f"  Завантажено: {len(df):,} статей | Справжніх: {df['label'].sum():,} | Фейків: {(df['label']==0).sum():,}")

# --- Розподіл ---
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['text_combined'].tolist(), df['label'].tolist(),
    test_size=CONFIG['val_split'], random_state=CONFIG['random_seed'], stratify=df['label']
)

# --- Навчання (DistilBERT для демонстрації) ---
print(f"\nНавчання: {CONFIG['demo_model_id']}")
tokenizer = AutoTokenizer.from_pretrained(CONFIG['demo_model_id'])
model = AutoModelForSequenceClassification.from_pretrained(CONFIG['demo_model_id'], num_labels=2).to(device)

train_loader = DataLoader(NewsDataset(train_texts, train_labels, tokenizer, CONFIG['max_len']),
                          batch_size=CONFIG['batch_size'], shuffle=True)
val_loader   = DataLoader(NewsDataset(val_texts, val_labels, tokenizer, CONFIG['max_len']),
                          batch_size=CONFIG['batch_size'] * 2)

optimizer = AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
total_steps = len(train_loader) * CONFIG['num_epochs']
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=int(total_steps * CONFIG['warmup_ratio']),
                                            num_training_steps=total_steps)

history = {'train_loss':[], 'val_loss':[], 'train_acc':[], 'val_acc':[]}

for epoch in range(1, CONFIG['num_epochs'] + 1):
    # Train
    model.train()
    tl, tc, tt = 0, 0, 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch} Train"):
        optimizer.zero_grad()
        out = model(input_ids=batch['input_ids'].to(device),
                    attention_mask=batch['attention_mask'].to(device),
                    labels=batch['label'].to(device))
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); scheduler.step()
        tl += out.loss.item(); tc += (out.logits.argmax(1) == batch['label'].to(device)).sum().item(); tt += len(batch['label'])
    # Val
    model.eval()
    vl, vc, vt = 0, 0, 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch} Val  "):
            out = model(input_ids=batch['input_ids'].to(device),
                        attention_mask=batch['attention_mask'].to(device),
                        labels=batch['label'].to(device))
            vl += out.loss.item(); vc += (out.logits.argmax(1) == batch['label'].to(device)).sum().item(); vt += len(batch['label'])
    history['train_loss'].append(tl/len(train_loader)); history['train_acc'].append(tc/tt)
    history['val_loss'].append(vl/len(val_loader));     history['val_acc'].append(vc/vt)
    print(f"  E{epoch}: train_loss={tl/len(train_loader):.4f} val_acc={vc/vt:.4f}")

# --- Збереження ---
os.makedirs("models/distilbert-colab", exist_ok=True)
model.save_pretrained("models/distilbert-colab")
tokenizer.save_pretrained("models/distilbert-colab")
print(f"\n✓ Модель збережена: models/distilbert-colab")

# --- Фінальний графік ---
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Криві навчання: DistilBERT")
a1.plot(history['train_loss'], 'b-o', label='Train'); a1.plot(history['val_loss'], 'r-o', label='Val')
a1.set_title("Loss"); a1.legend(); a1.grid(alpha=0.3)
a2.plot(history['train_acc'], 'b-o', label='Train'); a2.plot(history['val_acc'], 'r-o', label='Val')
a2.set_title("Accuracy"); a2.legend(); a2.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("training_curves.png", dpi=120, bbox_inches='tight')
plt.show()
print("✓ Графік збережено: training_curves.png")
