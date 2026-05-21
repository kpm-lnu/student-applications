# 04_inference.py
# ============================================================
# ЕТАП 6: Інференс — інтерактивна перевірка новин
# ============================================================
# Запуск: python 04_inference.py
#
# Що робить цей скрипт:
#   1. Завантажує найкращу збережену модель
#   2. Надає інтерактивний інтерфейс у терміналі
#   3. Приймає текст/заголовок новини від користувача
#   4. Видає вердикт із відсотком впевненості
#   5. Опціонально: демонстрація на прикладах
# ============================================================

import os
import sys
import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.text_cleaner import combine_title_and_text

# ─────────────────────────────────────────────────────────
# КЛАС ДЕТЕКТОРА ФЕЙКОВИХ НОВИН
# ─────────────────────────────────────────────────────────
class FakeNewsDetector:
    """
    Обгортка над навченою трансформерною моделлю.

    Надає простий API для перевірки новин:
        detector = FakeNewsDetector("models/roberta")
        result = detector.predict("Заголовок", "Текст новини")
    """

    # Назви класів
    CLASS_NAMES = {0: "ФЕЙК", 1: "СПРАВЖНЯ"}

    # Емодзі для відображення результату
    CLASS_EMOJI = {0: "🔴", 1: "🟢"}

    def __init__(self, model_dir: str, max_len: int = 128):
        """
        Завантажує модель та токенізатор із збереженої директорії.

        Args:
            model_dir : шлях до папки зі збереженою моделлю
            max_len   : максимальна довжина послідовності (має збігатися з навчанням)
        """
        self.max_len   = max_len
        self.model_dir = model_dir

        # Визначення пристрою
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Завантаження моделі: {model_dir}")
        print(f"Пристрій: {self.device}")

        # Завантаження токенізатора
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

        # Завантаження моделі
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model = self.model.to(self.device)

        # Переводимо модель у режим inference
        # (вимикає dropout, встановлює eval mode)
        self.model.eval()

        print("✓ Модель готова до роботи!\n")

    def predict(self, title: str, text: str = "") -> dict:
        """
        Класифікує одну новину.

        Алгоритм:
        1. Очищення та об'єднання заголовку та тексту
        2. Токенізація рядка
        3. Forward pass через модель (без градієнтів)
        4. Softmax → ймовірності класів
        5. Вибір класу з найвищою ймовірністю

        Args:
            title : заголовок новини (обов'язково)
            text  : текст новини (опціонально)

        Returns:
            Словник з результатом:
            {
                'label':       int (0 або 1),
                'class_name':  str ("ФЕЙК" або "СПРАВЖНЯ"),
                'confidence':  float (0.0 – 1.0),
                'prob_fake':   float (ймовірність фейку),
                'prob_real':   float (ймовірність справжньої),
            }
        """
        # Об'єднання та очищення вхідних даних
        if text:
            input_text = combine_title_and_text(title, text)
        else:
            input_text = title

        # Токенізація
        encoding = self.tokenizer(
            input_text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',  # Повертає PyTorch тензори
        )

        # Переміщення на GPU/CPU
        input_ids      = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        # Inference (без підрахунку градієнтів)
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        # Логіти → ймовірності через softmax
        # logits: "сирий" вихід мережі (можуть бути будь-якими числами)
        # softmax: перетворює у ймовірності (0–1, сума = 1)
        logits = outputs.logits
        probs  = torch.softmax(logits, dim=1).squeeze(0)

        prob_fake = probs[0].item()  # Ймовірність класу 0 (ФЕЙК)
        prob_real = probs[1].item()  # Ймовірність класу 1 (СПРАВЖНЯ)

        # Визначення класу
        pred_label = 1 if prob_real > prob_fake else 0
        confidence = max(prob_real, prob_fake)

        return {
            'label':      pred_label,
            'class_name': self.CLASS_NAMES[pred_label],
            'emoji':      self.CLASS_EMOJI[pred_label],
            'confidence': confidence,
            'prob_fake':  prob_fake,
            'prob_real':  prob_real,
        }

    def predict_batch(self, articles: list) -> list:
        """
        Класифікує список новин.

        Args:
            articles : список словників {'title': ..., 'text': ...}

        Returns:
            Список результатів predict()
        """
        results = []
        for article in articles:
            title = article.get('title', '')
            text  = article.get('text', '')
            result = self.predict(title, text)
            result['title'] = title[:80]
            results.append(result)
        return results

    def print_result(self, result: dict, title: str = "", verbose: bool = True):
        """
        Красиво виводить результат у консоль.

        Args:
            result  : словник із predict()
            title   : заголовок для відображення
            verbose : чи показувати детальні ймовірності
        """
        print("\n" + "─" * 55)
        if title:
            print(f"Новина: {title[:70]}{'...' if len(title) > 70 else ''}")

        # Головний вердикт
        verdict_color = ""  # ANSI кольори для терміналу
        print(f"\n{result['emoji']} ВЕРДИКТ: {result['class_name']}")
        print(f"   Впевненість: {result['confidence'] * 100:.1f}%")

        if verbose:
            # Бар-чарт у терміналі
            bar_len = 30
            real_bar = int(result['prob_real'] * bar_len)
            fake_bar = int(result['prob_fake'] * bar_len)
            print(f"\n   Справжня:  [{'█' * real_bar}{'░' * (bar_len - real_bar)}] {result['prob_real']*100:.1f}%")
            print(f"   Фейк:      [{'█' * fake_bar}{'░' * (bar_len - fake_bar)}] {result['prob_fake']*100:.1f}%")

        print("─" * 55)


# ─────────────────────────────────────────────────────────
# ДЕМОНСТРАЦІЯ НА ПРИКЛАДАХ
# ─────────────────────────────────────────────────────────
DEMO_ARTICLES = [
    {
        "title": "U.S. Senate passes $1.1 trillion spending bill",
        "text": (
            "The Senate passed a massive $1.1 trillion spending bill on "
            "Saturday that funds the government through September 2015, "
            "averting a shutdown with a bipartisan vote of 56 to 40. "
            "The House approved the measure on Thursday."
        ),
        "expected": "Справжня (Reuters-стиль)"
    },
    {
        "title": "BREAKING: Hillary Clinton Admits She 'Made Up' Russia Hacking Story",
        "text": (
            "In a stunning turn of events, Hillary Clinton has reportedly "
            "admitted to her inner circle that the Russia hacking narrative "
            "was completely fabricated to explain her devastating loss to "
            "Donald Trump. Liberal media is covering this up!!! SHARE before "
            "they delete this!!!"
        ),
        "expected": "Фейк (емоційна мова, сенсаційність)"
    },
    {
        "title": "Federal Reserve raises interest rates by quarter point",
        "text": (
            "The Federal Reserve raised its benchmark interest rate by a "
            "quarter of a percentage point on Wednesday, the third increase "
            "this year, and signaled it was on track to hike borrowing costs "
            "one more time in 2017 and three times next year."
        ),
        "expected": "Справжня (фактична, нейтральна)"
    },
    {
        "title": "EXPOSED: George Soros Paying Protestors $35/Hour to Destabilize America",
        "text": (
            "Documents leaked to our exclusive sources reveal that the "
            "globalist billionaire George Soros has been funneling millions "
            "to paid protestors across the United States. The mainstream "
            "media won't tell you about this conspiracy! Wake up sheeple!"
        ),
        "expected": "Фейк (теорія змови, вигадані 'джерела')"
    },
]


# ─────────────────────────────────────────────────────────
# ГОЛОВНИЙ БЛОК
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":

    print("=" * 60)
    print("ІНФЕРЕНС — ДЕТЕКТОР ФЕЙКОВИХ НОВИН")
    print("=" * 60)

    # --- Визначення найкращої моделі ---
    best_model_json = "results/best_model.json"
    default_model   = "models/roberta"  # Fallback

    if os.path.exists(best_model_json):
        with open(best_model_json, 'r') as f:
            best_info = json.load(f)
        model_dir = best_info.get('model_dir', default_model)
        print(f"Найкраща модель (з оцінки): {best_info.get('best_model', '?')}")
    else:
        model_dir = default_model
        print(f"⚠ best_model.json не знайдено, використовується: {model_dir}")
        print("  Запустіть 03_evaluate.py для визначення найкращої моделі.")

    # Можна вказати конкретну модель аргументом командного рядка:
    # python 04_inference.py models/bert-base
    if len(sys.argv) > 1:
        model_dir = sys.argv[1]
        print(f"Використовується модель з аргументу: {model_dir}")

    # Перевірка наявності моделі
    if not os.path.exists(model_dir):
        print(f"\n❌ Модель не знайдена: {model_dir}")
        print("Спочатку запустіть навчання: python 02_train.py")
        sys.exit(1)

    # --- Завантаження детектора ---
    detector = FakeNewsDetector(model_dir)

    # ─────────────────────────────────────────────────────
    # РЕЖИМ 1: Демонстрація на прикладах
    # ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("ДЕМОНСТРАЦІЯ НА ТЕСТОВИХ ПРИКЛАДАХ")
    print("=" * 60)

    correct = 0
    for i, article in enumerate(DEMO_ARTICLES, 1):
        print(f"\n[Приклад {i}/{ len(DEMO_ARTICLES)}]")
        print(f"Очікування: {article['expected']}")

        result = detector.predict(article['title'], article['text'])
        detector.print_result(result, title=article['title'])

    # ─────────────────────────────────────────────────────
    # РЕЖИМ 2: Інтерактивний ввід від користувача
    # ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("ІНТЕРАКТИВНИЙ РЕЖИМ")
    print("Введіть новину для перевірки. 'q' — вийти.")
    print("=" * 60)

    while True:
        print()
        title = input("📰 Заголовок новини (або 'q' для виходу): ").strip()

        if title.lower() in ('q', 'quit', 'exit', 'вийти'):
            print("\nДо побачення!")
            break

        if not title:
            print("⚠ Введіть заголовок.")
            continue

        text = input("📄 Текст новини (Enter — пропустити): ").strip()

        # Запуск inference
        print("\n⏳ Аналізую...")
        result = detector.predict(title, text)
        detector.print_result(result, title=title, verbose=True)

        # Пояснення для користувача
        if result['confidence'] < 0.70:
            print(
                "  ℹ Увага: впевненість нижче 70% — результат може бути "
                "ненадійним. Перевірте новину в інших джерелах."
            )
        elif result['confidence'] > 0.95:
            print(
                f"  ✓ Висока впевненість ({result['confidence']*100:.1f}%) — "
                f"модель впевнена у своєму рішенні."
            )
