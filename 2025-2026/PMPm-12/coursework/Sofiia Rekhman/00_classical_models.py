# 00_classical_models.py
# ============================================================
# КЛАСИЧНІ ML-АЛГОРИТМИ для виявлення фейкових новин
# ============================================================
# Запуск: python 00_classical_models.py
#
# Що робить цей скрипт:
#   1. Завантажує підготовлений датасет (з 01_eda.py)
#   2. Будує TF-IDF векторизатор (замість токенізатора BERT)
#   3. Навчає 6 класичних алгоритмів:
#        - Logistic Regression
#        - Naive Bayes (MultinomialNB)
#        - Support Vector Machine (LinearSVC)
#        - Random Forest
#        - Gradient Boosting (XGBoost)
#        - Gradient Boosting (LightGBM)
#   4. Для кожної моделі: Classification Report + Confusion Matrix
#   5. Зберігає всі результати у папку results_classical/
#   6. Зберігає pickle-файли навчених моделей
#
# Встановлення додаткових залежностей (якщо ще не встановлено):
#   pip install xgboost lightgbm
# ============================================================

import os
import sys
import time
import json
import pickle
import warnings
warnings.filterwarnings('ignore')  # Прибираємо зайві попередження sklearn

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Класичні ML алгоритми
from sklearn.linear_model    import LogisticRegression
from sklearn.naive_bayes     import MultinomialNB
from sklearn.svm             import LinearSVC
from sklearn.ensemble        import RandomForestClassifier
from sklearn.calibration     import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics         import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# TF-IDF — основний інструмент векторизації для класичного ML
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

# Gradient Boosting (встановіть окремо якщо відсутні)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠ XGBoost не встановлено. Запустіть: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠ LightGBM не встановлено. Запустіть: pip install lightgbm")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ─────────────────────────────────────────────────────────
# КОНФІГУРАЦІЯ
# ─────────────────────────────────────────────────────────
CONFIG = {
    "val_split":   0.20,
    "random_seed": 42,
    # TF-IDF параметри
    "tfidf_max_features": 100_000,   # Словник: топ-100к найчастіших n-грам
    "tfidf_ngram_range":  (1, 2),    # Уніграми + біграми (напр. "fake news")
    "tfidf_min_df":       2,         # Ігнорувати терми, що зустрічаються < 2 рази
    "tfidf_max_df":       0.95,      # Ігнорувати терми у > 95% документів (стоп-слова)
    "tfidf_sublinear_tf": True,      # log(1+tf) замість tf — зменшує вагу частих слів
}

DATA_DIR    = "data"
RESULTS_DIR = "results_classical"   # Окрема папка від трансформерів
MODELS_DIR  = "models_classical"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR,  exist_ok=True)


# ─────────────────────────────────────────────────────────
# ВИЗНАЧЕННЯ МОДЕЛЕЙ
# ─────────────────────────────────────────────────────────
# Кожна модель — словник з: об'єктом sklearn, назвою, описом.
# Усі моделі використовують однаковий TF-IDF на вході.
#
# ЧОМУ ТАКІ ГІПЕРПАРАМЕТРИ:
# ─────────────────────────
# LogisticRegression:
#   C=1.0    — стандартна регуляризація (менше C = сильніша)
#   max_iter=1000 — збільшено для великих датасетів (за замовч. 100)
#   solver='lbfgs' — ефективний для великих датасетів
#
# MultinomialNB:
#   alpha=0.1 — Лапласове згладжування (менше = ближче до MLE)
#   Важливо: НЕ працює з від'ємними значеннями TF-IDF,
#   тому використовуємо sublinear_tf (завжди > 0)
#
# LinearSVC:
#   C=1.0 — penalтy параметр
#   max_iter=2000 — для великих датасетів
#   Обернено до LogReg: SVM максимізує margin, а не ймовірність.
#   Обертаємо у CalibratedClassifierCV для отримання predict_proba
#   (потрібно для ROC-AUC)
#
# RandomForest:
#   n_estimators=200 — кількість дерев (більше = краще, але повільніше)
#   max_depth=None    — дерева ростуть до повного розкриття
#   min_samples_leaf=2 — мінімальний розмір листа (регуляризація)
#   n_jobs=-1         — паралельне навчання на всіх ядрах CPU
#   Увага: RF + TF-IDF = повільно через 100k фіч. Можна знизити n_estimators.
#
# XGBoost:
#   n_estimators=300      — кількість дерев
#   max_depth=6           — глибина (стандарт для XGBoost)
#   learning_rate=0.1     — крок навчання
#   subsample=0.8         — частка рядків на дерево (регуляризація)
#   colsample_bytree=0.8  — частка фіч на дерево
#   use_label_encoder=False — вимикаємо застаріле кодування
#   eval_metric='logloss' — метрика для early stopping
#
# LightGBM:
#   n_estimators=300      — кількість дерев
#   num_leaves=63         — ключовий параметр (2^max_depth - 1 = 63 для depth=6)
#   learning_rate=0.1     — крок навчання
#   feature_fraction=0.8  — частка фіч на ітерацію (аналог colsample у XGBoost)
#   bagging_fraction=0.8  — частка рядків (аналог subsample)
#   bagging_freq=5        — кожні N ітерацій виконувати bagging
#   verbosity=-1          — вимкнути логи LightGBM

def build_models() -> dict:
    """Повертає словник {назва: sklearn-сумісний об'єкт}."""
    models = {
        "logistic_regression": {
            "model": LogisticRegression(
                C=1.0,
                max_iter=1000,
                solver='lbfgs',
                random_state=CONFIG['random_seed'],
                n_jobs=-1,
            ),
            "label": "Logistic Regression",
            "color": "#2196F3",
            "needs_dense": False,  # Чи потребує щільна матриця (dense)
        },
        "naive_bayes": {
            "model": MultinomialNB(alpha=0.1),
            "label": "Naive Bayes",
            "color": "#9C27B0",
            "needs_dense": False,
        },
        "svm": {
            # LinearSVC не має predict_proba → обертаємо у CalibratedClassifierCV
            # method='sigmoid' — Platt scaling для калібрування ймовірностей
            "model": CalibratedClassifierCV(
                LinearSVC(C=1.0, max_iter=2000, random_state=CONFIG['random_seed']),
                method='sigmoid',
                cv=3,
            ),
            "label": "Support Vector Machine",
            "color": "#FF5722",
            "needs_dense": False,
        },
        "random_forest": {
            "model": RandomForestClassifier(
                n_estimators=200,
                max_depth=None,
                min_samples_leaf=2,
                random_state=CONFIG['random_seed'],
                n_jobs=-1,
            ),
            "label": "Random Forest",
            "color": "#4CAF50",
            "needs_dense": True,   # RandomForest потребує щільну матрицю
        },
    }

    # XGBoost (якщо встановлено)
    if XGBOOST_AVAILABLE:
        models["xgboost"] = {
            "model": xgb.XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=CONFIG['random_seed'],
                eval_metric='logloss',
                verbosity=0,
                n_jobs=-1,
            ),
            "label": "XGBoost",
            "color": "#FF9800",
            "needs_dense": True,
        }

    # LightGBM (якщо встановлено)
    if LIGHTGBM_AVAILABLE:
        models["lightgbm"] = {
            "model": lgb.LGBMClassifier(
                n_estimators=300,
                num_leaves=63,
                learning_rate=0.1,
                feature_fraction=0.8,
                bagging_fraction=0.8,
                bagging_freq=5,
                random_state=CONFIG['random_seed'],
                verbosity=-1,
                n_jobs=-1,
            ),
            "label": "LightGBM",
            "color": "#009688",
            "needs_dense": True,
        }

    return models


# ─────────────────────────────────────────────────────────
# КЛАС TF-IDF ВЕКТОРИЗАТОРА
# ─────────────────────────────────────────────────────────
def build_tfidf_vectorizer() -> TfidfVectorizer:
    """
    Будує та повертає TF-IDF векторизатор.

    TF-IDF (Term Frequency–Inverse Document Frequency):
    ────────────────────────────────────────────────────
    TF(t, d)  = кількість входжень терму t у документі d
    IDF(t)    = log(N / df(t)), де N = кількість документів,
                df(t) = кількість документів, що містять t

    TF-IDF(t, d) = TF(t, d) × IDF(t)

    Інтуїція: слово "the" зустрічається скрізь (низький IDF),
    тому отримує малу вагу. Слово "fabricated" рідкісне
    (високий IDF), тому отримує велику вагу.

    sublinear_tf=True: замість tf використовує 1+log(tf),
    що зменшує вплив дуже частих слів в одному документі.

    ngram_range=(1,2): включає і окремі слова, і пари слів.
    Приклад: "fake news" як біграма несе більше інформації,
    ніж просто "fake" + "news" окремо.
    """
    return TfidfVectorizer(
        max_features   = CONFIG['tfidf_max_features'],
        ngram_range    = CONFIG['tfidf_ngram_range'],
        min_df         = CONFIG['tfidf_min_df'],
        max_df         = CONFIG['tfidf_max_df'],
        sublinear_tf   = CONFIG['tfidf_sublinear_tf'],
        strip_accents  = 'unicode',   # Нормалізація акцентів
        analyzer       = 'word',      # Токенізація по словах
        token_pattern  = r'\b[a-zA-Z][a-zA-Z]+\b',  # Лише слова довжиною >= 2
        lowercase      = True,        # На відміну від BERT — нижній регістр (для класичного ML)
    )


# ─────────────────────────────────────────────────────────
# НАВЧАННЯ ТА ОЦІНКА ОДНІЄЇ МОДЕЛІ
# ─────────────────────────────────────────────────────────
def train_and_evaluate(
    name:        str,
    model_info:  dict,
    X_train,
    X_val,
    y_train:     np.ndarray,
    y_val:       np.ndarray,
) -> dict:
    """
    Навчає та оцінює одну класичну модель.

    Відмінність від трансформерів:
    - Немає епох — навчання одноразове (fit)
    - Немає батчів — модель бачить усі дані одразу (або за частинами у деяких)
    - Набагато швидше, але не враховує контекст слів

    Args:
        name       : ключ моделі (напр. "logistic_regression")
        model_info : словник з об'єктом моделі та метаданими
        X_train    : TF-IDF матриця (sparse або dense)
        X_val      : TF-IDF матриця для валідації
        y_train    : мітки навчальної вибірки
        y_val      : мітки валідаційної вибірки

    Returns:
        Словник з метриками (сумісний з форматом 03_evaluate.py)
    """
    label = model_info['label']
    clf   = model_info['model']
    print(f"\n── {label} ──")

    # Деякі моделі потребують щільну (dense) матрицю
    # (scipy sparse — це ефективний формат для TF-IDF, але не всі моделі його підтримують)
    if model_info.get('needs_dense'):
        print(f"  Перетворення у dense матрицю...", end=' ')
        X_tr = X_train.toarray() if hasattr(X_train, 'toarray') else X_train
        X_vl = X_val.toarray()   if hasattr(X_val,   'toarray') else X_val
        print("готово")
    else:
        X_tr, X_vl = X_train, X_val

    # ── Навчання ──
    print(f"  Навчання...", end=' ', flush=True)
    t0 = time.time()
    clf.fit(X_tr, y_train)
    elapsed = time.time() - t0
    elapsed_str = f"{elapsed:.1f}с" if elapsed < 60 else f"{int(elapsed//60)}хв {int(elapsed%60)}с"
    print(f"готово за {elapsed_str}")

    # ── Передбачення ──
    y_pred = clf.predict(X_vl)

    # ── Ймовірності для ROC-AUC ──
    # predict_proba повертає [P(клас 0), P(клас 1)] для кожного прикладу
    try:
        y_prob = clf.predict_proba(X_vl)[:, 1]  # ймовірність класу 1 (справжня)
        roc_auc = roc_auc_score(y_val, y_prob)
    except AttributeError:
        # Деякі моделі не мають predict_proba (лінійний SVM без калібрування)
        y_prob  = None
        roc_auc = float('nan')
        print("  ⚠ predict_proba недоступна, ROC-AUC не розраховується")

    # ── Метрики ──
    acc       = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, average='weighted', zero_division=0)
    recall    = recall_score(y_val, y_pred, average='weighted',    zero_division=0)
    f1        = f1_score(y_val, y_pred, average='weighted',        zero_division=0)

    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    if not np.isnan(roc_auc):
        print(f"  ROC-AUC   : {roc_auc:.4f}")

    # ── Детальний звіт ──
    report = classification_report(
        y_val, y_pred,
        target_names=['Фейк (0)', 'Справжня (1)'],
        digits=4,
    )
    print(f"\n  Classification Report:\n{report}")

    report_path = f"{RESULTS_DIR}/report_{name}.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"=== Classification Report: {label} ===\n\n")
        f.write(report)
        f.write(f"\nЧас навчання: {elapsed_str}\n")
    print(f"  ✓ Звіт збережено: {report_path}")

    # ── Confusion Matrix + ROC-крива ──
    _plot_confusion_and_roc(name, label, y_val, y_pred, y_prob, roc_auc)

    # ── Збереження натренованої моделі (pickle) ──
    # pickle серіалізує Python-об'єкт у бінарний файл.
    # Для sklearn це стандартний спосіб збереження.
    model_path = f"{MODELS_DIR}/{name}.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(clf, f)
    print(f"  ✓ Модель збережена: {model_path}")

    # ── Важливість фіч (для підтримуючих моделей) ──
    _plot_feature_importance(name, label, clf, vectorizer=None)  # vectorizer передамо пізніше

    return {
        'model_name':  name,
        'label':       label,
        'color':       model_info['color'],
        'accuracy':    acc,
        'precision':   precision,
        'recall':      recall,
        'f1_score':    f1,
        'roc_auc':     roc_auc if not np.isnan(roc_auc) else None,
        'elapsed_sec': elapsed,
        'elapsed_str': elapsed_str,
        'y_pred':      y_pred,
        'y_prob':      y_prob,
    }


def _plot_confusion_and_roc(
    name: str, label: str,
    y_true, y_pred, y_prob, roc_auc: float
):
    """Confusion Matrix та ROC-крива для однієї моделі."""
    cm = confusion_matrix(y_true, y_pred)

    n_subplots = 2 if y_prob is not None else 1
    fig, axes = plt.subplots(1, n_subplots, figsize=(6 * n_subplots, 4))
    if n_subplots == 1:
        axes = [axes]
    fig.suptitle(f"{label}", fontweight='bold')

    # Confusion Matrix
    sns.heatmap(
        cm, ax=axes[0],
        annot=True, fmt='d', cmap='Blues',
        xticklabels=['Фейк', 'Справжня'],
        yticklabels=['Фейк', 'Справжня'],
        linewidths=0.5,
    )
    axes[0].set_title("Confusion Matrix")
    axes[0].set_ylabel("Реальний клас")
    axes[0].set_xlabel("Передбачений клас")
    tn, fp, fn, tp = cm.ravel()
    axes[0].set_xlabel(f"TN={tn:,}  FP={fp:,}  FN={fn:,}  TP={tp:,}", fontsize=9, color='gray')

    # ROC-крива
    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        axes[1].plot(fpr, tpr, linewidth=2, label=f'AUC = {roc_auc:.4f}')
        axes[1].plot([0, 1], [0, 1], 'r--', linewidth=1, label='Випадкова')
        axes[1].fill_between(fpr, tpr, alpha=0.1)
        axes[1].set_title("ROC-крива")
        axes[1].set_xlabel("False Positive Rate")
        axes[1].set_ylabel("True Positive Rate")
        axes[1].legend(fontsize=10)
        axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = f"{RESULTS_DIR}/confusion_{name}.png"
    plt.savefig(path, dpi=120, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Графік збережено: {path}")


def _plot_feature_importance(name: str, label: str, clf, vectorizer):
    """
    Будує графік топ-20 найважливіших фіч.

    Підтримується для:
    - LogisticRegression  → coef_ (ваги логістичної регресії)
    - LinearSVC (через CalibratedCV) → не доступні напряму
    - RandomForest        → feature_importances_ (середнє зменшення Gini)
    - XGBoost / LightGBM  → feature_importances_ (gain-based)
    """
    if vectorizer is None:
        return  # Векторизатор ще не доступний на цьому кроці

    feature_names = np.array(vectorizer.get_feature_names_out())

    coef = None
    importance_type = ""

    # LogisticRegression: coef_ має форму (1, n_features) для бінарної задачі
    if hasattr(clf, 'coef_'):
        coef = clf.coef_[0] if clf.coef_.ndim > 1 else clf.coef_
        importance_type = "Ваги (LogReg/SVM)"

    # RandomForest, XGBoost, LightGBM: feature_importances_
    elif hasattr(clf, 'feature_importances_'):
        coef = clf.feature_importances_
        importance_type = "Feature Importance"

    # CalibratedClassifierCV: дістаємо з внутрішнього estimator
    elif hasattr(clf, 'calibrated_classifiers_'):
        try:
            base = clf.calibrated_classifiers_[0].estimator
            if hasattr(base, 'coef_'):
                coef = base.coef_[0]
                importance_type = "Ваги (SVM/калібрований)"
        except Exception:
            pass

    if coef is None or len(coef) != len(feature_names):
        return

    # Топ-20 фіч: 10 найбільш "справжніх" + 10 найбільш "фейкових"
    n = 20
    if importance_type.startswith("Ваги"):
        # Позитивні коефіцієнти → клас "справжня"
        # Негативні коефіцієнти → клас "фейк"
        top_pos_idx = np.argsort(coef)[-n//2:][::-1]   # Топ "справжня"
        top_neg_idx = np.argsort(coef)[:n//2]           # Топ "фейк"
        top_idx     = np.concatenate([top_neg_idx, top_pos_idx])
        colors_bar  = ['#F44336'] * (n//2) + ['#2196F3'] * (n//2)
    else:
        top_idx    = np.argsort(coef)[-n:][::-1]
        colors_bar = ['#FF9800'] * n

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(
        range(len(top_idx)),
        coef[top_idx],
        color=colors_bar,
        edgecolor='white', linewidth=0.5,
    )
    ax.set_yticks(range(len(top_idx)))
    ax.set_yticklabels(feature_names[top_idx], fontsize=9)
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.set_title(f"Топ-{n} важливих слів: {label}\n({importance_type})", fontweight='bold')
    ax.set_xlabel(importance_type)
    ax.grid(axis='x', alpha=0.3)

    if importance_type.startswith("Ваги"):
        ax.text(0.02, 0.02, "← Фейк", transform=ax.transAxes,
                color='#F44336', fontsize=9)
        ax.text(0.75, 0.02, "Справжня →", transform=ax.transAxes,
                color='#2196F3', fontsize=9)

    plt.tight_layout()
    path = f"{RESULTS_DIR}/features_{name}.png"
    plt.savefig(path, dpi=120, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Важливість фіч збережено: {path}")


# ─────────────────────────────────────────────────────────
# ПОРІВНЯЛЬНІ ГРАФІКИ КЛАСИЧНИХ МОДЕЛЕЙ
# ─────────────────────────────────────────────────────────
def plot_classical_comparison(all_results: list):
    """Порівняльні графіки для всіх класичних моделей."""

    labels  = [r['label'] for r in all_results]
    colors  = [r['color'] for r in all_results]
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Класичні ML-моделі: порівняння метрик", fontweight='bold', fontsize=13)

    # ── Subplot 1: Grouped bar chart ──
    x = np.arange(len(metrics))
    bar_w = 0.8 / len(all_results)  # Ширина підпристосована до кількості моделей

    for i, (result, color) in enumerate(zip(all_results, colors)):
        values = [result[m] for m in metrics]
        offset = (i - len(all_results) / 2 + 0.5) * bar_w
        bars = axes[0].bar(x + offset, values, bar_w,
                           label=result['label'], color=color, alpha=0.85,
                           edgecolor='white')
        for bar, val in zip(bars, values):
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002,
                f"{val:.3f}", ha='center', va='bottom', fontsize=7, rotation=45,
            )

    axes[0].set_title("Метрики якості")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metric_labels)
    axes[0].set_ylim(0.80, 1.03)
    axes[0].legend(fontsize=8, loc='lower right')
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].axhline(y=1.0, color='gray', linestyle=':', linewidth=0.8)

    # ── Subplot 2: Час навчання (горизонтальний) ──
    times   = [r['elapsed_sec'] for r in all_results]
    y_pos   = range(len(all_results))
    h_bars  = axes[1].barh(y_pos, times, color=colors, alpha=0.85, edgecolor='white')
    for bar, t in zip(h_bars, times):
        axes[1].text(
            bar.get_width() + max(times) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{t:.1f}с", va='center', fontsize=9,
        )
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(labels)
    axes[1].set_title("Час навчання (секунди)")
    axes[1].set_xlabel("Секунди")
    axes[1].grid(axis='x', alpha=0.3)
    axes[1].set_xlim(0, max(times) * 1.2)

    plt.tight_layout()
    path = f"{RESULTS_DIR}/classical_comparison.png"
    plt.savefig(path, dpi=130, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n✓ Порівняльний графік збережено: {path}")


# ─────────────────────────────────────────────────────────
# ГОЛОВНИЙ БЛОК
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":

    print("=" * 60)
    print("КЛАСИЧНІ ML-АЛГОРИТМИ: ВИЯВЛЕННЯ ФЕЙКОВИХ НОВИН")
    print("=" * 60)

    # ── 1. Завантаження підготовленого датасету ──
    prepared_path = f"{DATA_DIR}/prepared_dataset.csv"
    if not os.path.exists(prepared_path):
        print("Файл не знайдено. Спочатку запустіть: python 01_eda.py")
        sys.exit(1)

    print("Завантаження датасету...")
    df = pd.read_csv(prepared_path).dropna(subset=['text_combined', 'label'])
    texts  = df['text_combined'].tolist()
    labels = df['label'].astype(int).values

    print(f"  Всього: {len(df):,} | Справжніх: {labels.sum():,} | Фейків: {(labels==0).sum():,}")

    # ── 2. Розподіл Train/Val (той самий seed = ті самі приклади) ──
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        texts, labels,
        test_size=CONFIG['val_split'],
        random_state=CONFIG['random_seed'],
        stratify=labels,
    )
    print(f"\nРозподіл: Train={len(X_train_raw):,} | Val={len(X_val_raw):,}")

    # ── 3. TF-IDF векторизація ──
    # fit() вивчає словник на ТІЛЬКИ навчальних даних.
    # transform() застосовує вивчений словник до val даних.
    # ВАЖЛИВО: ніколи не робіть fit_transform() на всьому датасеті —
    # це призведе до data leakage (val входить у словник).
    print("\nTF-IDF векторизація...")
    t0 = time.time()
    vectorizer = build_tfidf_vectorizer()

    X_train = vectorizer.fit_transform(X_train_raw)    # Вивчити + перетворити
    X_val   = vectorizer.transform(X_val_raw)          # Тільки перетворити!

    print(f"  Форма матриці: {X_train.shape}")
    print(f"  Словник: {len(vectorizer.vocabulary_):,} термів")
    print(f"  Щільність: {X_train.nnz / (X_train.shape[0] * X_train.shape[1]):.4%}")
    print(f"  Час векторизації: {time.time()-t0:.1f}с")

    # Збереження векторизатора
    with open(f"{MODELS_DIR}/tfidf_vectorizer.pkl", 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f"  ✓ Векторизатор збережено: {MODELS_DIR}/tfidf_vectorizer.pkl")

    # ── 4. Навчання та оцінка всіх моделей ──
    models     = build_models()
    all_results = []

    for name, model_info in models.items():
        result = train_and_evaluate(name, model_info, X_train, X_val, y_train, y_val)

        # Будуємо важливість фіч після того, як vectorizer вже готовий
        _plot_feature_importance(name, result['label'], model_info['model'], vectorizer)

        all_results.append(result)

    # ── 5. Порівняльні графіки ──
    plot_classical_comparison(all_results)

    # ── 6. Зведена таблиця ──
    print("\n" + "=" * 70)
    print("ЗВЕДЕНА ТАБЛИЦЯ: КЛАСИЧНІ МОДЕЛІ")
    print("=" * 70)
    print(f"{'Модель':<28} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'AUC':>7} {'Час':>8}")
    print("-" * 70)

    # Сортування за F1-Score
    all_results_sorted = sorted(all_results, key=lambda x: x['f1_score'], reverse=True)

    for r in all_results_sorted:
        auc_str = f"{r['roc_auc']:.4f}" if r['roc_auc'] is not None else "  N/A "
        print(
            f"{r['label']:<28} "
            f"{r['accuracy']:>7.4f} "
            f"{r['precision']:>7.4f} "
            f"{r['recall']:>7.4f} "
            f"{r['f1_score']:>7.4f} "
            f"{auc_str:>7} "
            f"{r['elapsed_str']:>8}"
        )

    best = all_results_sorted[0]
    print(f"\n🏆 Найкраща класична модель: {best['label']} (F1={best['f1_score']:.4f})")

    # ── 7. Збереження результатів у JSON ──
    save_data = []
    for r in all_results:
        save_data.append({k: v for k, v in r.items()
                          if k not in ('y_pred', 'y_prob')})  # Масиви не серіалізуємо

    results_path = f"{RESULTS_DIR}/classical_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n✓ Результати збережено: {results_path}")

    # CSV-версія для зручного перегляду
    pd.DataFrame(all_results_sorted)[
        ['label', 'accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'elapsed_str']
    ].to_csv(f"{RESULTS_DIR}/classical_metrics.csv", index=False)

    print("\n" + "=" * 60)
    print("Класичні моделі навчено! Переходьте до: python 06_compare.py")
    print("=" * 60)
