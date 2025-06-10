from flask import Flask, request, render_template
import nltk
import emoji
from transformers import pipeline
from nltk.sentiment.vader import SentimentIntensityAnalyzer

app = Flask(__name__)

# Завантаження ресурсів
nltk.download('vader_lexicon')
vader_analyzer = SentimentIntensityAnalyzer()
bert_classifier = pipeline("sentiment-analysis")

# === 1. Кастомний словник сленгу/лайки/сарказму ===
custom_words = {
    # 🟢 Позитивний сленг
    'slay': 2.5, 'based': 2.2, 'rizz': 1.8, 'goated': 2.5, 'lit': 2.0,
    'fire': 1.7, 'fr': 1.0, 'no cap': 1.5, 'deadass': 0.8, 'vibe': 1.2,
    'ate': 2.0, 'werk': 1.3, 'iconic': 2.0, 'valid': 1.5, 'queen': 2.0,
    'mother': 1.7, 'serve': 1.6, 'periodt': 1.8, 'snatched': 1.5, 'lmao': 1.3, 'hella': 0.8, 'savage': 0.5,

    # 🟡 Нейтрально-іронічні або двозначні
    '💅': -0.5, '💀': -1.0, '🫠': -0.8, 'mood': 0.2, 'okayyy': -0.2,
    'lolz': -0.5, 'k': -0.5, 'yikes': -1.5, 'sksksk': 0.2,
    '🙃': -0.5, 'bruh': -0.3, 'sheesh': 0.5, 'ai generated': -1.0,

    # 🔴 Негативний сленг, сарказм або образи
    'mid': -1.2, 'delulu': -1.5, 'cringe': -2.0, 'sus': -1.7,
    'nah fam': -1.5, 'ratio': -1.0, 'corny': -1.3, 'basic': -1.0,
    'clown': -1.8, 'flop': -1.5, 'l': -1.0, 'npc': -1.5, 'bombastic': -0.5,

    # 🤬 Нецензурна лексика
    'fuck': -3.5, 'fucked': -3.5, 'shit': -2.8, 'shitty': -2.9,
    'bitch': -2.5, 'bitches': -2.0, 'damn': -1.8, 'wtf': -2.2,
    'tf': -2.3, 'idgaf': -2.5, 'idc': -1.0, 'asshole': -3.2,
    'dumbass': -2.8, 'bullshit': -3.0, 'smh': -1.2
}
vader_analyzer.lexicon.update(custom_words)

# === 2. Емоджі словник ===
emoji_sentiment = {
    '😊': 1, '😂': 1, '😍': 1, '😃': 1, '😆': 1,
    '😢': -1, '😭': -1, '😡': -1, '😠': -1, '🤬': -1,
    '😐': 0, '😶': 0, '😑': 0, '😏': -0.5, '😒': -0.5,
    '👍': 0.5, '👎': -0.5, '💀': -1, '🤡': -1, '🔥': 0.5,
    '😇': 1, '🥰': 1, '🙃': -0.5, '🫠': -0.5, '😎': 0.5,
    '🥺': -1, '😤': -0.7, '🤔': 0, '😬': -0.5, '😳': -0.5,
    '💩': -1, '😴': -0.2, '🤢': -1, '🤑': 0.3, '💔': -1,
    '❤️': 1, '✨': 0.5, '🥳': 1, '🙄': -0.7, '😪': -0.4,
    '🥲': -0.2, '🥹': -0.2, '☹️': -1, '🤷🏻‍♀️': 0, '🤦🏽‍♀️': -1,
    '😵‍💫': -0.5, '🤣': 0.5, '😝': 1, '😌': 1, '🙈': 1, '🤓': 0,
    '🤐': 0, '😘': 0, '😙': 1, '😚': 1, '🤪': 0.5, '😜': 0.5,
    '🥸': 0, '🙂': 0.5, '‍↕️': 0, '‍↔️': 0, '🥱': -0.2,
    '🤕': -1, '☠️': -1, '😼': 0.2, '😺': 0.8, '🫶🏻': 1, '🤝': 0.5,
    '🙌🏻': 1, '👊🏻': 0.5, '👋': 1, '👅': 1, '👄': 1, '💋': 1,
    '🧡': 1, '💛': 1, '💚': 1, '💙': 1, '💜': 1, '🖤': 1,
    '🤍': 1, '🤎': 1, '❤️‍🔥': 1, '❤️‍🩹': -0.5, '❣️': 1,
    '💕': 1, '💞': 1, '💓': 1, '💗': 1, '💖': 1, '💘': 1,
    '💝': 1, '✅': 0, '❌': -1
}

# === 3. Допоміжні функції ===
def extract_emojis(text):
    return [ch for ch in text if ch in emoji.EMOJI_DATA]

def analyze_emoji(emojis):
    scores = [emoji_sentiment.get(e, 0) for e in emojis]
    return sum(scores) / len(scores) if scores else 0

def analyze_text(text):
    vader_score = vader_analyzer.polarity_scores(text)['compound']
    bert_result = bert_classifier(text)[0]
    bert_score = bert_result['score'] if bert_result['label'] == 'POSITIVE' else -bert_result['score']
    emoji_score = analyze_emoji(extract_emojis(text))

    final_score = (vader_score + bert_score + emoji_score) / 3

    if final_score > 0.3:
        mood = "позитивний"
    elif final_score < -0.3:
        mood = "негативний"
    else:
        mood = "нейтральний"

    return {
        "vader": round(vader_score, 3),
        "bert": round(bert_score, 3),
        "emoji": round(emoji_score, 3),
        "final": round(final_score, 3),
        "mood": mood
    }

# === 4. Основний маршрут Flask ===
@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        user_text = request.form['text']
        result = analyze_text(user_text)
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
