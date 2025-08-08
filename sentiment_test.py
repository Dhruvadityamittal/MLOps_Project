import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier  # ✅ New model

# === Config ===
BASE_DIR = os.getcwd()  # or "/opt/airflow" if running in Docker
DATA_PATH = os.path.join(BASE_DIR, "data", "IMDB_Dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "sentiment_model.pkl")


def load_dataset():
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    if not os.path.exists(DATA_PATH):
        df = pd.DataFrame({
            'review': [
                'This movie was great!',
                'Awful acting and story.',
                'I loved the cinematography.',
                'Terrible. I walked out halfway.',
                'Best film ever made!',
                'Worst experience of my life.'
            ],
            'sentiment': ['positive', 'negative', 'positive', 'negative', 'positive', 'negative']
        })
        df.to_csv(DATA_PATH, index=False)
        print("✅ Dataset created and saved.")
    else:
        print("✅ Dataset already exists.")


def preprocess_and_train():
    df = pd.read_csv(DATA_PATH)
    df.dropna(inplace=True)

    X = df['review']
    y = LabelEncoder().fit_transform(df['sentiment'])

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
        ('clf', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
    ])
    pipeline.fit(X_train, y_train)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    print("✅ Model trained using XGBoost and saved.")
    return "✅ Training completed with XGBoost and model saved."


def evaluate_model(message_from_train):
    print(f"Message from training step: {message_from_train}")

    df = pd.read_csv(DATA_PATH)
    df.dropna(inplace=True)

    X = df['review']
    y = LabelEncoder().fit_transform(df['sentiment'])
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = joblib.load(MODEL_PATH)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"✅ Model accuracy: {acc:.4f}")


# === Run all steps ===
if __name__ == "__main__":
    load_dataset()
    train_status = preprocess_and_train()
    evaluate_model(train_status)
