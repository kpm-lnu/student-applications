import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight


users = pd.read_csv("users.csv", sep=';')
articles = pd.read_csv("smoking_articles.csv", sep=';', on_bad_lines='skip')

print("Користувачі:")
print(users.head())
print("\nСтатті:")
print(articles.head())

label_encoders = {}
for col in ['gender', 'highest_qualification', 'marital_status', 'gross_income', 'smoke']:
    le = LabelEncoder()
    users[col] = le.fit_transform(users[col])
    label_encoders[col] = le

scaler = StandardScaler()
users[['age']] = scaler.fit_transform(users[['age']])

vectorizer = TfidfVectorizer(max_features=100)
article_vectors = vectorizer.fit_transform(articles['content']).toarray()

pca = PCA(n_components=6)
article_vectors_reduced = pca.fit_transform(article_vectors)

user_vectors = users[['gender', 'age', 'marital_status', 'highest_qualification', 'gross_income', 'smoke']].values

recommendations = {}

for i, user in users.iterrows():
    user_vector = user_vectors[i]

    similarities = cosine_similarity([user_vector], article_vectors_reduced).flatten()

    top_articles_idx = np.argsort(similarities)[-3:][::-1]

    recommendations[user['id']] = articles.iloc[top_articles_idx]['id'].tolist()

interaction_data = {
    'user_id': [],
    'article_id': [],
    'interaction': []  
}

for user_id, recs in recommendations.items():
    for article_id in recs:
        interaction_data['user_id'].append(user_id)
        interaction_data['article_id'].append(article_id)
        interaction_data['interaction'].append(1)

    non_interacted_articles = articles[~articles['id'].isin(recs)].sample(n=5)['id'].tolist()
    for article_id in non_interacted_articles:
        interaction_data['user_id'].append(user_id)
        interaction_data['article_id'].append(article_id)
        interaction_data['interaction'].append(0) 

interactions = pd.DataFrame(interaction_data)

print("\nВзаємодії:")
print(interactions.head())

train_data, test_data = train_test_split(interactions, test_size=0.2, random_state=42)

num_users = users['id'].max() + 1
num_articles = articles['id'].max() + 1

user_input = tf.keras.Input(shape=(1,), name='user')
article_input = tf.keras.Input(shape=(1,), name='article')

embedding_dim = 16

user_embedding = tf.keras.layers.Embedding(input_dim=num_users, output_dim=embedding_dim)(user_input)
user_embedding = tf.keras.layers.Flatten()(user_embedding)

article_embedding = tf.keras.layers.Embedding(input_dim=num_articles, output_dim=embedding_dim)(article_input)
article_embedding = tf.keras.layers.Flatten()(article_embedding)

merged = tf.keras.layers.Concatenate()([user_embedding, article_embedding])

x = tf.keras.layers.Dense(256, activation='swish', kernel_regularizer=tf.keras.regularizers.l2(0.001))(merged)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(128, activation='swish', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(64, activation='swish', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.2)(x)



output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(inputs=[user_input, article_input], outputs=output)

optimizer = tf.keras.optimizers.SGD(learning_rate=0.0005, momentum=0.9)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])


# optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, weight_decay=0.0001)
# model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',  
    patience=3,         
    restore_best_weights=True 
)

user_input_data = train_data['user_id'].values
article_input_data = train_data['article_id'].values
interaction_labels = train_data['interaction'].values

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(interaction_labels),
    y=interaction_labels
)
class_weights = dict(enumerate(class_weights))

history = model.fit(
    [user_input_data, article_input_data],
    interaction_labels,
    epochs=50,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stopping],
    class_weight=class_weights
)

test_user_input_data = test_data['user_id'].values
test_article_input_data = test_data['article_id'].values
test_interaction_labels = test_data['interaction'].values

model.evaluate([test_user_input_data, test_article_input_data], test_interaction_labels)
loss, accuracy = model.evaluate([test_user_input_data, test_article_input_data], test_interaction_labels)
print(f"\nТочність моделі на тестових даних: {accuracy * 100:.2f}%")
y_pred = model.predict([test_user_input_data, test_article_input_data])

y_pred_binary = (y_pred > 0.5).astype(int)

f1 = f1_score(test_interaction_labels, y_pred_binary)
roc_auc = roc_auc_score(test_interaction_labels, y_pred)
precision = precision_score(test_interaction_labels, y_pred_binary)
recall = recall_score(test_interaction_labels, y_pred_binary)

print(f"F1-середнє: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

plt.figure(figsize=(14, 5))

# Точність
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Тренувальна точність')
plt.plot(history.history['val_accuracy'], label='Валідаційна точність')
plt.xlabel('Епоха')
plt.ylabel('Точність')
plt.title('Точність моделі під час тренування')
plt.legend()
plt.grid(True)

# Втрати (Loss)
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Тренувальні втрати')
plt.plot(history.history['val_loss'], label='Валідаційні втрати')
plt.xlabel('Епоха')
plt.ylabel('Втрати')
plt.title('Втрати моделі під час тренування')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()