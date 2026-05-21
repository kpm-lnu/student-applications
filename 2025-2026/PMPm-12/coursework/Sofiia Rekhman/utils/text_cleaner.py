# utils/text_cleaner.py
# ============================================================
# Утиліти для очищення та попередньої обробки тексту новин.
# Цей модуль імпортується в усіх скриптах проєкту.
# ============================================================

import re
import pandas as pd


def remove_reuters_tag(text: str) -> str:
    """
    Видаляє маркер джерела Reuters із початку тексту.

    ЧОМУ ЦЕ КРИТИЧНО:
    Майже всі справжні новини в датасеті ISOT починаються з
    "(НАЗВА МІСТА, країна) - " або просто "(Reuters) - ".
    Якщо цей маркер залишити, модель вивчить простий патерн:
    "є такий рядок → справжня новина" — замість аналізу змісту.
    Це називається data leakage (витік даних).

    Приклад:
        Вхід:  "WASHINGTON (Reuters) - President signed..."
        Вихід: "President signed..."
    """
    # Видаляємо шаблон виду: "МІСТО (Reuters) - " або просто "(Reuters) - "
    # re.sub замінює знайдений шаблон на порожній рядок
    text = re.sub(r'^.*?\(Reuters\)\s*-\s*', '', text, flags=re.IGNORECASE)
    # Також видаляємо геолокаційні маркери без слова Reuters
    # Приклад: "WASHINGTON - Some text" → "Some text"
    text = re.sub(r'^[A-Z\s,]+\s*-\s*', '', text)
    return text.strip()


def clean_text(text: str) -> str:
    """
    Базове очищення тексту для NLP-задачі.

    Виконує:
    1. Видалення маркера Reuters
    2. Видалення зайвих пробілів та переносів рядків
    3. Зберігає пунктуацію — вона несе семантичне навантаження!

    НЕ робимо (на відміну від класичного ML):
    - Не видаляємо стоп-слова (трансформери самі навчаться їх ігнорувати)
    - Не стемуємо (трансформери працюють з повними словами через subword tokenization)
    - Не переводимо у нижній регістр (BERT та ін. враховують регістр)

    Args:
        text: Сирий текст статті

    Returns:
        Очищений текст
    """
    if not isinstance(text, str):
        return ""

    # Крок 1: Видалення маркера Reuters
    text = remove_reuters_tag(text)

    # Крок 2: Заміна множинних пробілів/переносів на один пробіл
    text = re.sub(r'\s+', ' ', text)

    # Крок 3: Видалення HTML-тегів (якщо є)
    text = re.sub(r'<[^>]+>', '', text)

    return text.strip()


def combine_title_and_text(title: str, text: str, separator: str = " [SEP] ") -> str:
    """
    Об'єднує заголовок та текст новини в один рядок.

    ЧОМУ ОБ'ЄДНУЄМО:
    Заголовок часто містить найбільш виразні ознаки фейку
    (сенсаційність, емоційне забарвлення). Тому ми подаємо
    обидва поля моделі разом.

    Токен [SEP] — спеціальний розділювач у архітектурі BERT,
    який дозволяє моделі розуміти межу між двома частинами.

    Args:
        title: Заголовок статті
        text: Тіло статті
        separator: Розділювач між частинами

    Returns:
        Об'єднаний рядок
    """
    clean_title = clean_text(str(title))
    clean_body = clean_text(str(text))
    return clean_title + separator + clean_body


def load_and_prepare_dataset(data_dir: str = "data") -> pd.DataFrame:
    """
    Завантажує файли True.csv та Fake.csv і готує єдиний датафрейм.

    Структура результуючого датафрейму:
        - text_combined : заголовок + [SEP] + текст (очищений)
        - label         : 1 = справжня, 0 = фейк
        - subject       : категорія новини
        - title         : оригінальний заголовок
        - original_text : оригінальний текст (для відлагодження)

    Args:
        data_dir: Шлях до папки з CSV-файлами

    Returns:
        Підготовлений DataFrame
    """
    import os

    true_path = os.path.join(data_dir, "True.csv")
    fake_path = os.path.join(data_dir, "Fake.csv")

    # Перевірка наявності файлів
    if not os.path.exists(true_path) or not os.path.exists(fake_path):
        raise FileNotFoundError(
            f"Файли датасету не знайдено у папці '{data_dir}'.\n"
            "Завантажте True.csv та Fake.csv з Kaggle"
        )

    print("Завантаження датасету ISOT...")

    # Завантаження файлів
    df_true = pd.read_csv(true_path)
    df_fake = pd.read_csv(fake_path)

    print(f"  Справжніх статей: {len(df_true):,}")
    print(f"  Фейкових статей:  {len(df_fake):,}")

    # Присвоєння міток: 1 = справжня, 0 = фейк
    df_true['label'] = 1
    df_fake['label'] = 0

    # Об'єднання в один датафрейм
    df = pd.concat([df_true, df_fake], ignore_index=True)

    # Перемішування (важливо: щоб під час навчання класи чергувалися)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Збереження оригінальних полів для відлагодження
    df['original_text'] = df['text'].copy()

    # Очищення та об'єднання полів
    print("Очищення тексту та видалення маркерів Reuters...")
    df['text_combined'] = df.apply(
        lambda row: combine_title_and_text(row['title'], row['text']),
        axis=1
    )

    # Видалення рядків з порожнім текстом (на всяк випадок)
    before = len(df)
    df = df[df['text_combined'].str.len() > 10].reset_index(drop=True)
    after = len(df)
    if before != after:
        print(f"  Видалено {before - after} рядків з порожнім текстом")

    print(f"Датасет готовий: {len(df):,} статей\n")

    return df
