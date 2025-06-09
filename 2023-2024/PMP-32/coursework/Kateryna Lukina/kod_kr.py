# Словник з асоційованими сентиментами
sentiment_dict = {
    "добре": 1,
    "нейтрально": 0,
    "погано": -1
}

# Функція для обчислення сентименту тексту за словником
def calculate_sentiment(text):
    words = text.lower().split()  # Розбиваємо текст на слова та переводимо в нижній регістр
    sentiment_score = 0
    for word in words:
        # Видаляємо розділові знаки з кінця слова
        word = word.strip('.,!?')
        if word in sentiment_dict:
            sentiment_score += sentiment_dict[word]  # Додаємо сентимент слова до загального сентименту
    return sentiment_score

# Функція для оцінки загального сентименту тексту
def evaluate_sentiment(score):
    if score > 0:
        return "добре"
    elif score == 0:
        return "нейтрально"
    else:
        return "погано"

# Приклад використання
text = "Цей продукт працює добре і мені дуже сподобався. Але погано, що ціна трохи висока."
sentiment_score = calculate_sentiment(text)
sentiment_evaluation = evaluate_sentiment(sentiment_score)

print("Сентимент тексту:", sentiment_score)
print("Оцінка сентименту:", sentiment_evaluation)
