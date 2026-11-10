from pathlib import Path
import json
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "dataset"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

def read_dataset():
    fake_path = DATA_DIR / "Fake.csv"
    real_path = DATA_DIR / "True.csv"

    if not fake_path.exists() or not real_path.exists():
        raise FileNotFoundError(
            "Dataset files are missing. Put Fake.csv and True.csv inside the dataset folder."
        )

    fake = pd.read_csv(fake_path)
    real = pd.read_csv(real_path)

    for name, frame in [("Fake.csv", fake), ("True.csv", real)]:
        if "text" not in frame.columns:
            raise ValueError(f"{name} must contain a column named 'text'.")

    fake["title"] = fake["title"].fillna("") if "title" in fake else ""
    real["title"] = real["title"].fillna("") if "title" in real else ""
    fake["text"] = fake["text"].fillna("")
    real["text"] = real["text"].fillna("")

    fake["content"] = (fake["title"].astype(str) + " " + fake["text"].astype(str)).str.strip()
    real["content"] = (real["title"].astype(str) + " " + real["text"].astype(str)).str.strip()
    fake["label"] = "FAKE"
    real["label"] = "REAL"

    data = pd.concat(
        [fake[["content", "label"]], real[["content", "label"]]],
        ignore_index=True
    )
    data["content"] = data["content"].str.replace(r"\s+", " ", regex=True).str.strip()
    data = data[data["content"].str.len() >= 40]
    data = data.drop_duplicates(subset="content").sample(frac=1, random_state=42)
    if data["label"].nunique() != 2:
        raise ValueError("Both FAKE and REAL classes must be present in the dataset.")
    return data

def main():
    print("Loading dataset...")
    data = read_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        data["content"], data["label"],
        test_size=0.20, random_state=42, stratify=data["label"]
    )

    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Multinomial Naive Bayes": MultinomialNB()
    }

    results = []
    best_pipeline = None
    best_name = None
    best_f1 = -1

    for name, classifier in candidates.items():
        print(f"Training {name}...")
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                stop_words="english",
                max_features=100000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95
            )),
            ("classifier", classifier)
        ])
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        scores = {
            "model": name,
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision": float(precision_score(y_test, predictions, pos_label="FAKE", zero_division=0)),
            "recall": float(recall_score(y_test, predictions, pos_label="FAKE", zero_division=0)),
            "f1": float(f1_score(y_test, predictions, pos_label="FAKE", zero_division=0))
        }
        results.append(scores)
        print(
            f"{name}: accuracy={scores['accuracy']:.4f}, "
            f"precision={scores['precision']:.4f}, "
            f"recall={scores['recall']:.4f}, f1={scores['f1']:.4f}"
        )

        if scores["f1"] > best_f1:
            best_f1 = scores["f1"]
            best_pipeline = pipeline
            best_name = name
            best_predictions = predictions

    joblib.dump(best_pipeline, MODEL_DIR / "fake_news_model.joblib")

    metrics = {
        "model_name": best_name,
        "dataset_size": int(len(data)),
        "accuracy": float(accuracy_score(y_test, best_predictions)),
        "precision_fake": float(precision_score(y_test, best_predictions, pos_label="FAKE", zero_division=0)),
        "recall_fake": float(recall_score(y_test, best_predictions, pos_label="FAKE", zero_division=0)),
        "f1_fake": float(f1_score(y_test, best_predictions, pos_label="FAKE", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, best_predictions, labels=["FAKE", "REAL"]).tolist(),
        "comparison": results
    }
    with open(MODEL_DIR / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print("\nTraining complete!")
    print(f"Best model: {best_name}")
    print(f"Saved model to: {MODEL_DIR / 'fake_news_model.joblib'}")
    print(f"Saved metrics to: {MODEL_DIR / 'metrics.json'}")

if __name__ == "__main__":
    main()
