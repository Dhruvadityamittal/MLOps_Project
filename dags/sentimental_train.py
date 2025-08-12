from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import mlflow

# === Config ===
BASE_DIR = "/opt/airflow/data"  # if using Docker-based Airflow
DATA_PATH = os.path.join(BASE_DIR, "IMDB_Dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "sentiment_model.pkl")

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
}



with DAG(
    dag_id='xgboost_sentiment_training_dag',
    default_args=default_args,
    description='Train and evaluate sentiment model using XGBoost',
    # schedule_interval=None,
    catchup=False
) as dag:

    def load_dataset(ti):
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        # if not os.path.exists(DATA_PATH):
        #     df = pd.DataFrame({
        #         'review': [
        #             'This movie was great!',
        #             'Awful acting and story.',
        #             'I loved the cinematography.',
        #             'Terrible. I walked out halfway.',
        #             'Best film ever made!',
        #             'Worst experience of my life.'
        #         ],
        #         'sentiment': ['positive', 'negative', 'positive', 'negative', 'positive', 'negative']
        #     })
        # df.to_csv(DATA_PATH, index=False)
        # print("✅ Dataset created and saved.")
        # else:s
        print("✅ Dataset already exists.")
        ti.xcom_push(key='test_size', value=0.2)
        ti.xcom_push(key='random_state', value=42)

    def preprocess_and_train(ti):
        df = pd.read_csv(DATA_PATH)
        df.dropna(inplace=True)

        X = df['review']
        y = LabelEncoder().fit_transform(df['sentiment'])

        X_train, _, y_train, _ = train_test_split(X, y, test_size=ti.xcom_pull(task_ids='load_dataset', key='test_size'), random_state= ti.xcom_pull(task_ids='load_dataset', key='random_state'))

        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000)),
            ('clf', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
        ])
        pipeline.fit(X_train, y_train)

        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(pipeline, MODEL_PATH)

        print("✅ Model trained using XGBoost and saved.")
        ti.xcom_push(key='train_status', value='✅ Training completed with XGBoost and model saved.')

    def evaluate_model(ti):
        # mlflow.set_tracking_uri("http://host.docker.internal:5000")

        message_from_train = ti.xcom_pull(task_ids='preprocess_and_train', key='train_status')
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

        # === MLflow Logging ===
        # mlflow.set_tracking_uri("http://localhost:5000")  # Or your MLflow server URI
        mlflow.set_tracking_uri("http://mlflow:5000")
        mlflow.set_experiment("SentimentAnalysis-XGBoost")

        with mlflow.start_run(run_name="XGB_Evaluation"):
            mlflow.log_metric("accuracy", acc)
            mlflow.log_param("model_type", "XGBoost")
            # mlflow.sklearn.log_model(model, "xgb_model")  # Save model for reproducibility
            print("✅ Metrics logged to MLflow.")

    # Airflow Tasks
    task_load = PythonOperator(
        task_id='load_dataset',
        python_callable=load_dataset
    )

    task_train = PythonOperator(
        task_id='preprocess_and_train',
        python_callable=preprocess_and_train,
        
    )

    task_eval = PythonOperator(
        task_id='evaluate_model',
        python_callable=evaluate_model,
        
    )

    # Define Task Dependencies
    task_load >> task_train >> task_eval
