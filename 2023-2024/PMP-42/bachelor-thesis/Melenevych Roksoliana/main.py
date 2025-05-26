import pandas as pd
import numpy as np
import arxiv
import requests
import os
import re
import fitz
from collections import defaultdict
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

from get_authors_with_publications import get_authors_with_publications
from download_author_papers import download_author_papers
from convert_pdf_to_text import convert_pdf_to_text
from preprocess_text import preprocess_text


def main():
    author_names = get_authors_with_publications()
    data = pd.DataFrame(columns=['text', 'author'])
    download_dir = 'D:\\University\\Dyplomna_robota\\python_code\\arxiv_papers'

    for author_name in author_names:
        pdf_paths = download_author_papers(author_name, download_dir)

        for pdf_path in pdf_paths:
            txt = convert_pdf_to_text(pdf_path)
            txt_file_path = pdf_path.replace('.pdf', '.txt')
            with open(txt_file_path, 'w', encoding='utf-8') as txt_file:
                txt_file.write(txt)
            print(f"Downloaded and converted {len(pdf_paths)} papers for author '{author_name}'.")

            data = data.append({'text': txt, 'author': author_name}, ignore_index=True)

    data['processed_text'] = data['text'].apply(preprocess_text)

    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(data['processed_text']).toarray()
    y = data['author']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "SVC": SVC(),
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier()
    }

    param_grids = {
        "SVC": {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'gamma': ['scale', 'auto'],
            'degree': [2, 3, 4]
        },
        "Naive Bayes": {
            'alpha': [0.1, 0.5, 1, 2, 5],
            'fit_prior': [True, False]
        },
        "Logistic Regression": {
            'C': [0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2', 'elasticnet', 'none'],
            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
        },
        "Random Forest": {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]
        }
    }

    for model_name, model in models.items():
        param_grid = param_grids[model_name]
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1)
        
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_

        model.fit(X_train, y_train)
        
        y_pred = best_model.predict(X_test)
        
        print(f"Classification Report for {model_name}:\n")
        print(classification_report(y_test, y_pred))
        print("\n\n")


if __name__ == "__main__":
    main()