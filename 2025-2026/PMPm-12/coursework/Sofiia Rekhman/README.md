# Виявлення фейкових новин за допомогою трансформерів

---

## Структура проєкту

```
fake_news_detector/
│
├── requirements.txt          # Залежності
├── README.md                 # Цей файл
│
├── data/                     # Папка для датасету ISOT
│   ├── True.csv              # ← завантажити вручну (див. нижче)
│   └── Fake.csv              # ← завантажити вручну (див. нижче)
│
├── utils/
│   └── text_cleaner.py       # Утиліти очищення тексту
│
├── 00_classical_models.py    # Етап 1: Навчання та аналіз класичних алгоритмів
├── 01_eda.py                 # Етап 2: Аналіз та підготовка даних
├── 02_train.py               # Етап 4: Навчання моделей
├── 03_evaluate.py            # Етап 5: Оцінка та збереження результатів
├── 04_inference.py           # Етап 6: Інференс — перевірка нових новин
│
├── models_classical/         # Збережені ваги моделей класичних методів (генерується автоматично)
├── results_classical/        # Графіки та звіти класичних методів (генерується автоматично)
├── models/                   # Збережені ваги моделей трансформерних методів (генерується автоматично)
└── results/                  # Графіки та звіти трансформерних методів (генерується автоматично)
```

---

## Крок 1: Завантаження датасету ISOT

1. Перейдіть на Kaggle: https://www.kaggle.com/datasets/rahulogoel/isot-fake-news-dataset/
2. Завантажте `archive.zip`
3. Розпакуйте файли `True.csv` та `Fake.csv` у папку `data/`

---

## Крок 2: Налаштування середовища

### Варіант A — Conda (рекомендовано)
```bash
# Створити нове середовище з Python 3.10
conda create -n fake_news python=3.10 -y

# Активувати середовище
conda activate fake_news

# Встановити залежності
pip install -r requirements.txt
```

### Варіант B — venv
```bash
# Створити віртуальне середовище
python -m venv venv

# Активувати (Linux/Mac)
source venv/bin/activate

# Активувати (Windows)
venv\Scripts\activate

# Встановити залежності
pip install -r requirements.txt
```

---

## Крок 3: Запуск по порядку

```bash
# 1. Аналіз та підготовка даних (EDA)
python 01_eda.py

# 2. Навчання та порівняння результатів класичних моделей
python 00_classical_models.py

# 3. Навчання всіх трьох моделей
python 02_train.py

# 4. Оцінка та порівняння результатів
python 03_evaluate.py

# 5. Інтерактивна перевірка нових новин
python 04_inference.py
```

---

## GPU vs CPU

- **З GPU (NVIDIA CUDA)**: навчання займає ~15–30 хв на модель
- **Без GPU (тільки CPU)**: ~2–5 годин на модель (не рекомендовано)

Перевірити наявність GPU:
```python
import torch
print(torch.cuda.is_available())  # True = є GPU
```

---

## Цитування датасету

> Ahmed H, Traore I, Saad S. "Detecting opinion spams and fake news using text classification",
> Journal of Security and Privacy, Volume 1, Issue 1, Wiley, January/February 2018.
