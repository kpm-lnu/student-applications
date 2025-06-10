import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack

KEYWORDS = [
    "shocking", "scandal", "unbelievable", "you won’t believe", "amazing", "incredible",
    "top secret", "revealed", "finally exposed", "exposed", "the truth about", "what happened next",
    "goes viral", "breaks the internet", "jaw dropping", "insane", "crazy", "epic", "this will change everything",
    "life-changing", "must see", "watch now", "nobody saw this coming", "hidden truth", "banned", "illegal",
    "this one trick", "doctors hate", "before it's deleted", "this is why", "number one reason", "you need to know",
    "don’t ignore", "can’t believe", "warning", "alert", "urgent", "explosive", "shocking discovery",
    "outrage", "meltdown", "insider info", "hack", "leak", "conspiracy", "controversial", "exposed secret",
    "government cover-up", "game changer"
]

def extract_custom_features(series):
    df = pd.DataFrame({'text': series})
    df['upper'] = df['text'].apply(lambda x: int(x.isupper()) if isinstance(x, str) else 0)
    df['exclamations'] = df['text'].apply(lambda x: x.count('!') if isinstance(x, str) else 0)
    df['bait_score'] = df['text'].apply(
        lambda x: sum(word in x.lower() for word in KEYWORDS) if isinstance(x, str) else 0
    )
    df['contains_link'] = df['text'].apply(lambda x: int("http" in x or "www" in x) if isinstance(x, str) else 0)
    return df[['upper', 'exclamations', 'bait_score', 'contains_link']]

# --- Load dataset
fakes = pd.read_csv('Fake.csv')
reals = pd.read_csv('True.csv')
fakes['label'] = 1
reals['label'] = 0
data = pd.concat([fakes, reals], ignore_index=True).dropna(subset=['text'])

X = data['text']
y = data['label']

X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Feature engineering
tfidf = TfidfVectorizer(max_features=1000)
X_tfidf = tfidf.fit_transform(X_train)
X_custom = extract_custom_features(X_train)
X_combined = hstack([X_tfidf, X_custom.values])

# --- Train model
classifier = LogisticRegression(max_iter=1000)
classifier.fit(X_combined, y_train)

# --- Save model and vectorizer
joblib.dump(classifier, 'fake_news_model.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
print("[INFO] Training completed and models saved.")