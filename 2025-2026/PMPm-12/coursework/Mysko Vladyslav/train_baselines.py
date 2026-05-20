from pathlib import Path
import time
import json

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as plt


# =========================
# CONFIG
# =========================

INPUT_CSV = Path("dataset/features_balanced.csv")

OUTPUT_DIR = Path("results_baselines")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TEST_SIZE = 0.2
RANDOM_SEED = 42

DROP_COLUMNS = {
    "file",
    "label",
    "original_file",
}

CLASS_LABELS_UA = {
    "compressed": "архіви / стиснуті",
    "executable": "виконувані файли",
    "image": "зображення",
    "pdf": "PDF",
    "text": "текст",
}

MODEL_NAMES_UA = {
    "random_forest": "Випадковий ліс",
    "logistic_regression": "Логістична регресія",
    "svm_rbf": "SVM з RBF-ядром",
    "svm_linear": "Лінійний SVM",
    "knn": "k-найближчих сусідів",
}

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False


# =========================
# DATA LOADING
# =========================

def load_dataset():
    df = pd.read_csv(INPUT_CSV)

    y = df["label"].values

    feature_cols = [
        col for col in df.columns
        if col not in DROP_COLUMNS
    ]

    X = df[feature_cols].values.astype(np.float32)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    return X, y_encoded, label_encoder, feature_cols, df


# =========================
# HELPERS
# =========================

def translate_class_names(class_names):
    return [
        CLASS_LABELS_UA.get(class_name, class_name)
        for class_name in class_names
    ]


def get_model_name_ua(model_name):
    return MODEL_NAMES_UA.get(model_name, model_name)


# =========================
# PLOTTING
# =========================

def plot_confusion_matrix(cm, class_names, title, output_path):
    class_names_ua = translate_class_names(class_names)

    plt.figure(figsize=(8, 6))

    image = plt.imshow(cm)
    plt.title(title)

    cbar = plt.colorbar(image)
    cbar.set_label("Кількість зразків")

    ticks = np.arange(len(class_names_ua))
    plt.xticks(ticks, class_names_ua, rotation=45, ha="right")
    plt.yticks(ticks, class_names_ua)

    plt.xlabel("Передбачений клас")
    plt.ylabel("Справжній клас")

    for i in range(len(class_names_ua)):
        for j in range(len(class_names_ua)):
            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def save_feature_importance(model, feature_cols, model_name):
    if not hasattr(model, "feature_importances_"):
        print(f"{model_name}: feature_importances_ not available")
        return

    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    csv_path = OUTPUT_DIR / f"feature_importance_{model_name}.csv"
    importance_df.to_csv(csv_path, index=False)

    # Take the top 21 and drop the first entry (index 0)
    top20 = importance_df.head(21).iloc[1:]
    model_name_ua = get_model_name_ua(model_name)

    plt.figure(figsize=(10, 6))

    # Назви ознак залишаються без перекладу
    plt.barh(
        top20["feature"][::-1],
        top20["importance"][::-1],
    )

    plt.xlabel("Важливість ознаки")
    plt.ylabel("Ознака")
    plt.title(f"Топ-20 найважливіших ознак — {model_name_ua}")

    plt.tight_layout()

    png_path = OUTPUT_DIR / f"feature_importance_{model_name}.png"
    plt.savefig(png_path, dpi=300)
    plt.close()

    print(f"Saved feature importance: {csv_path}")
    print(f"Saved plot: {png_path}")

# =========================
# TRAIN / EVAL
# =========================

def evaluate_model(name, model, X_train, X_test, y_train, y_test, class_names):
    print(f"\n==============================")
    print(f"MODEL: {name}")
    print(f"==============================")

    start_train = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_train

    start_pred = time.time()
    y_pred = model.predict(X_test)
    pred_time = time.time() - start_pred

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")
    print(f"Train time: {train_time:.3f} sec")
    print(f"Prediction time: {pred_time:.3f} sec")

    class_names_ua = translate_class_names(class_names)

    report = classification_report(
        y_test,
        y_pred,
        target_names=class_names_ua,
        output_dict=True,
        zero_division=0,
    )

    print("\nClassification report:")
    print(classification_report(
        y_test,
        y_pred,
        target_names=class_names_ua,
        zero_division=0,
    ))

    cm = confusion_matrix(y_test, y_pred)

    model_name_ua = get_model_name_ua(name)

    cm_path = OUTPUT_DIR / f"confusion_matrix_{name}.png"
    plot_confusion_matrix(
        cm,
        class_names,
        title=f"Матриця помилок — {model_name_ua}",
        output_path=cm_path,
    )

    result = {
        "model": name,
        "model_ua": model_name_ua,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "train_time_sec": train_time,
        "prediction_time_sec": pred_time,
        "confusion_matrix_path": str(cm_path),
    }

    report_path = OUTPUT_DIR / f"classification_report_{name}.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)

    return result


# =========================
# MAIN
# =========================

def main():
    X, y, label_encoder, feature_cols, df = load_dataset()

    class_names = list(label_encoder.classes_)
    class_names_ua = translate_class_names(class_names)

    print("Dataset shape:", X.shape)
    print("Classes:", class_names)
    print("Classes UA:", class_names_ua)

    print("\nClass distribution:")
    print(df["label"].value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            class_weight="balanced",
        ),

        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=3000,
                C=1.0,
                solver="lbfgs",
                multi_class="auto",
                class_weight="balanced",
                random_state=RANDOM_SEED,
            )),
        ]),

        "svm_rbf": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(
                kernel="rbf",
                C=10,
                gamma="scale",
                class_weight="balanced",
                random_state=RANDOM_SEED,
            )),
        ]),

        "svm_linear": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(
                kernel="linear",
                C=1,
                class_weight="balanced",
                random_state=RANDOM_SEED,
            )),
        ]),

        "knn": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(
                n_neighbors=5,
                weights="distance",
            )),
        ]),
    }

    results = []
    trained_models = {}

    for name, model in models.items():
        result = evaluate_model(
            name=name,
            model=model,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            class_names=class_names,
        )

        results.append(result)
        trained_models[name] = model

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("macro_f1", ascending=False)

    results_path = OUTPUT_DIR / "baseline_results.csv"
    results_df.to_csv(results_path, index=False)

    save_feature_importance(
        trained_models["random_forest"],
        feature_cols,
        "random_forest",
    )

    print("\n==============================")
    print("FINAL RESULTS")
    print("==============================")
    print(results_df)

    print(f"\nSaved results to: {results_path}")


if __name__ == "__main__":
    main()