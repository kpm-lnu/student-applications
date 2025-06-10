import pytz

# Патч для astimezone
def patched_astimezone(tz):
    if tz is None:
        from tzlocal import get_localzone
        tz = get_localzone()
    if hasattr(tz, 'zone'):
        return tz
    try:
        return pytz.timezone(str(tz))
    except Exception:
        raise TypeError('Only timezones from the pytz library are supported')

import apscheduler.util
apscheduler.util.astimezone = patched_astimezone

# Тепер імпортуємо Telegram і все інше
import logging
import joblib
import pandas as pd
from scipy.sparse import hstack
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
import asyncio

#Завантаження моделі
model = joblib.load('fake_news_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

#Слова-клікбейти
clickbait_words = [
    "shocking", "scandal", "unbelievable", "you won’t believe", "amazing", "incredible",
    "top secret", "revealed", "finally exposed", "exposed", "the truth about", "what happened next",
    "goes viral", "breaks the internet", "jaw dropping", "insane", "crazy", "epic", "this will change everything",
    "life-changing", "must see", "watch now", "nobody saw this coming", "hidden truth", "banned", "illegal",
    "this one trick", "doctors hate", "before it's deleted", "this is why", "number one reason", "you need to know",
    "don’t ignore", "can’t believe", "warning", "alert", "urgent", "explosive", "shocking discovery",
    "outrage", "meltdown", "insider info", "hack", "leak", "conspiracy", "controversial", "exposed secret",
    "government cover-up", "game changer"
]

#Ручні ознаки
def hand_crafted_features(text_series):
    df = pd.DataFrame({'text': text_series})
    df['caps'] = df['text'].apply(lambda x: int(x.isupper()) if isinstance(x, str) else 0)
    df['excl'] = df['text'].apply(lambda x: x.count('!') if isinstance(x, str) else 0)
    df['clickbait'] = df['text'].apply(
        lambda x: sum(word in x.lower() for word in clickbait_words) if isinstance(x, str) else 0
    )
    df['link'] = df['text'].apply(lambda x: int("http" in x or "www" in x) if isinstance(x, str) else 0)
    return df[['caps', 'excl', 'clickbait', 'link']]

#Аналіз тексту
def predict_fake_news(text):
    series = pd.Series([text])
    tfidf = vectorizer.transform(series)
    features = hand_crafted_features(series)
    full = hstack([tfidf, features.values])
    prediction = model.predict(full)[0]
    prob = model.predict_proba(full)[0][prediction]
    label = "🟩 * Правдива новина *" if prediction == 0 else "🟥 * Фейкова новина *"
    return f"{label}\n\nЙмовірність: `{prob:.2f}`"

#Команди
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("👋 Надішли мені текст новини, а я скажу чи вона фейкова.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    result = predict_fake_news(text)
    await update.message.reply_markdown(result)
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Привіт! Надішли мені текст новини, а я скажу чи вона фейкова."
    )

if __name__ == '__main__':
    TOKEN = "7691365421:AAHvtKf7ndH9y4KmE32HG7fRzVNbwWo7xE8"
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    print("✅ Бот SuperficialInspector запущено")
    asyncio.run(app.run_polling())
