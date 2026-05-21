# 01_eda.py
# ============================================================
# ЕТАП 2: Експлоративний аналіз даних (EDA)
# ============================================================
# Запуск: python 01_eda.py
#
# Що робить цей скрипт:
#   1. Завантажує та очищує датасет ISOT
#   2. Виводить базову статистику по класах
#   3. Будує розподіл довжини текстів
#   4. Генерує WordCloud для кожного класу
#   5. Показує приклади до/після очищення
#   6. Зберігає всі графіки у папку results/
# ============================================================

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Бекенд без GUI — для збереження у файл
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from wordcloud import WordCloud

# Додаємо кореневу папку до шляху для імпорту utils
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.text_cleaner import load_and_prepare_dataset, remove_reuters_tag

# ── Налаштування ──────────────────────────────────────────
RESULTS_DIR = "results"
DATA_DIR    = "data"

# Параметр токенізатора
MAX_LEN_TARGET = 128

os.makedirs(RESULTS_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────
# 1. ЗАВАНТАЖЕННЯ ДАНИХ
# ─────────────────────────────────────────────────────────
print("=" * 60)
print("ЕТАП 2: АНАЛІЗ ДАНИХ (EDA)")
print("=" * 60)

df = load_and_prepare_dataset(DATA_DIR)

# ─────────────────────────────────────────────────────────
# 2. БАЗОВА СТАТИСТИКА
# ─────────────────────────────────────────────────────────
print("\n── Розподіл класів ──")
class_counts = df['label'].value_counts()
print(f"  Справжніх (label=1): {class_counts.get(1, 0):,}")
print(f"  Фейкових  (label=0): {class_counts.get(0, 0):,}")
balance_ratio = class_counts.min() / class_counts.max() * 100
print(f"  Баланс класів: {balance_ratio:.1f}% (100% = ідеальний баланс)")

if balance_ratio < 60:
    print("  ⚠ Значний дисбаланс! Розгляньте клас-ваги у loss функції.")
else:
    print("  ✓ Баланс прийнятний, зважування класів не обов'язкове.")

# Статистика категорій
if 'subject' in df.columns:
    print("\n── Категорії новин ──")
    subj_stat = df.groupby(['subject', 'label']).size().unstack(fill_value=0)
    print(subj_stat.to_string())

# ─────────────────────────────────────────────────────────
# 3. АНАЛІЗ ДОВЖИНИ ТЕКСТІВ
# ─────────────────────────────────────────────────────────
print("\n── Аналіз довжини текстів ──")

# Підрахунок кількості слів у кожній статті
df['word_count'] = df['text_combined'].apply(lambda x: len(x.split()))
# Підрахунок приблизної кількості токенів
df['approx_tokens'] = (df['word_count'] * 1.3).astype(int)

for label_val, label_name in [(1, "Справжні"), (0, "Фейки")]:
    subset = df[df['label'] == label_val]['word_count']
    print(f"\n  {label_name}:")
    print(f"    Мінімум:    {subset.min():,} слів")
    print(f"    Медіана:    {subset.median():.0f} слів")
    print(f"    Середнє:   {subset.mean():.0f} слів")
    print(f"    Максимум:  {subset.max():,} слів")
    # Скільки статей поміщається в MAX_LEN_TARGET токенів
    fits_in_max = (df[df['label'] == label_val]['approx_tokens'] <= MAX_LEN_TARGET).mean() * 100
    print(f"    Поміщається у {MAX_LEN_TARGET} токенів: {fits_in_max:.1f}%")

print(f"\n  📌 Обґрунтування max_len={MAX_LEN_TARGET}:")
total_fits = (df['approx_tokens'] <= MAX_LEN_TARGET).mean() * 100
print(f"     {total_fits:.1f}% усіх статей повністю поміщаються.")
print(f"     Решта буде обрізана. Це стандартний компроміс між")
print(f"     якістю та обчислювальними ресурсами.")

# ─────────────────────────────────────────────────────────
# 4. ПРИКЛАДИ ДО/ПІСЛЯ ОЧИЩЕННЯ (демонстрація data leakage)
# ─────────────────────────────────────────────────────────
print("\n── Приклади до/після видалення маркера Reuters ──")
true_examples = df[df['label'] == 1].head(3)
for _, row in true_examples.iterrows():
    original = str(row.get('original_text', ''))[:120]
    cleaned  = str(row['text_combined'])[:120]
    print(f"\n  ДО:   {original}...")
    print(f"  ПІСЛЯ: {cleaned}...")

# ─────────────────────────────────────────────────────────
# 5. ПОБУДОВА ГРАФІКІВ
# ─────────────────────────────────────────────────────────
print("\n── Генерація графіків ──")

fig = plt.figure(figsize=(16, 12))
fig.suptitle("EDA: Датасет виявлення фейкових новин (ISOT)",
             fontsize=15, fontweight='bold', y=0.98)

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# --- Графік 1: Розподіл класів ---
ax1 = fig.add_subplot(gs[0, 0])
colors = ['#2196F3', '#F44336']
bars = ax1.bar(['Справжні (1)', 'Фейки (0)'],
               [class_counts.get(1, 0), class_counts.get(0, 0)],
               color=colors, edgecolor='white', linewidth=1.5)
for bar, count in zip(bars, [class_counts.get(1, 0), class_counts.get(0, 0)]):
    ax1.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 200, f"{count:,}",
             ha='center', va='bottom', fontsize=10)
ax1.set_title("Розподіл класів", fontweight='bold')
ax1.set_ylabel("Кількість статей")
ax1.set_ylim(0, max(class_counts.values) * 1.15)
ax1.grid(axis='y', alpha=0.3)

# --- Графік 2: Гістограма довжини текстів ---
ax2 = fig.add_subplot(gs[0, 1])
for label_val, color, name in [(1, '#2196F3', 'Справжні'), (0, '#F44336', 'Фейки')]:
    data = df[df['label'] == label_val]['word_count']
    ax2.hist(data, bins=50, alpha=0.6, color=color, label=name, density=True)
# Лінія MAX_LEN_TARGET (переводимо з токенів у приблизні слова)
ax2.axvline(x=MAX_LEN_TARGET / 1.3, color='green', linestyle='--',
            linewidth=1.5, label=f'≈max_len={MAX_LEN_TARGET} токенів')
ax2.set_title("Розподіл довжини текстів", fontweight='bold')
ax2.set_xlabel("Кількість слів")
ax2.set_ylabel("Щільність")
ax2.legend(fontsize=9)
ax2.set_xlim(0, df['word_count'].quantile(0.99))
ax2.grid(alpha=0.3)

# --- Графік 3: Box-plot порівняння довжин ---
ax3 = fig.add_subplot(gs[0, 2])
plot_data = [
    df[df['label'] == 1]['word_count'].values,
    df[df['label'] == 0]['word_count'].values
]
bp = ax3.boxplot(plot_data, labels=['Справжні', 'Фейки'],
                 patch_artist=True, notch=False)
for patch, color in zip(bp['boxes'], ['#2196F3', '#F44336']):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
ax3.axhline(y=MAX_LEN_TARGET / 1.3, color='green', linestyle='--',
            linewidth=1.5, label=f'≈max_len={MAX_LEN_TARGET}')
ax3.set_title("Box-plot: довжина текстів", fontweight='bold')
ax3.set_ylabel("Кількість слів")
ax3.legend(fontsize=9)
ax3.grid(axis='y', alpha=0.3)

# --- Графік 4: WordCloud для справжніх новин ---
ax4 = fig.add_subplot(gs[1, 0])
true_text = " ".join(df[df['label'] == 1]['text_combined'].sample(
    min(500, len(df[df['label'] == 1])), random_state=42
).tolist())
wc_true = WordCloud(
    width=500, height=300,
    background_color='white',
    colormap='Blues',
    max_words=100,
    collocations=False  # Уникаємо повторення фраз
).generate(true_text)
ax4.imshow(wc_true, interpolation='bilinear')
ax4.set_title("WordCloud: Справжні новини", fontweight='bold', color='#1565C0')
ax4.axis('off')

# --- Графік 5: WordCloud для фейкових новин ---
ax5 = fig.add_subplot(gs[1, 1])
fake_text = " ".join(df[df['label'] == 0]['text_combined'].sample(
    min(500, len(df[df['label'] == 0])), random_state=42
).tolist())
wc_fake = WordCloud(
    width=500, height=300,
    background_color='white',
    colormap='Reds',
    max_words=100,
    collocations=False
).generate(fake_text)
ax5.imshow(wc_fake, interpolation='bilinear')
ax5.set_title("WordCloud: Фейкові новини", fontweight='bold', color='#B71C1C')
ax5.axis('off')

# --- Графік 6: Розподіл за категоріями ---
ax6 = fig.add_subplot(gs[1, 2])
if 'subject' in df.columns:
    subj_counts = df.groupby(['subject', 'label']).size().unstack(fill_value=0)
    subj_counts.plot(kind='barh', ax=ax6, color=['#F44336', '#2196F3'],
                     edgecolor='white', linewidth=0.5)
    ax6.set_title("Статті за категоріями", fontweight='bold')
    ax6.set_xlabel("Кількість")
    ax6.legend(['Фейки (0)', 'Справжні (1)'], fontsize=9)
    ax6.grid(axis='x', alpha=0.3)

plt.savefig(f"{RESULTS_DIR}/01_eda_analysis.png",
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  ✓ Збережено: {RESULTS_DIR}/01_eda_analysis.png")

# ─────────────────────────────────────────────────────────
# 6. ЗБЕРЕЖЕННЯ ПІДГОТОВЛЕНОГО ДАТАСЕТУ
# ─────────────────────────────────────────────────────────
# Зберігаємо очищений датасет, щоб не повторювати обробку при навчанні
save_path = f"{DATA_DIR}/prepared_dataset.csv"
df[['text_combined', 'label', 'subject', 'title']].to_csv(save_path, index=False)
print(f"  ✓ Збережено: {save_path}")

print("\n" + "=" * 60)
print("EDA завершено! Переходьте до: python 02_train.py")
print("=" * 60)
