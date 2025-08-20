import argparse
import json
import os
import time
import subprocess
from pathlib import Path
from typing import Optional, Tuple

import mlflow
import mlflow.sklearn
import dagshub
import numpy as np
import pandas as pd
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.linear_model import LogisticRegression, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    RocCurveDisplay
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")  # for headless environments
import matplotlib.pyplot as plt
from src.utils.logging_utils import setup_logger

logger = setup_logger(level=os.getenv("LOG_LEVEL","INFO"),
                   log_dir=os.getenv("LOG_DIR","/tmp/train-logs"),
                   filename=f"train_{os.getenv('RUN_ID','local')}.log")

def configure_mlflow():
    dagshub.init(repo_owner= os.getenv("DAGSHUB_REPO_OWNER"), repo_name=os.getenv("DAGSHUB_REPO"), mlflow=True)
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        logger.warning("MLFLOW_TRACKING_URI is not set. Logging locally (mlruns/) instead.")
    mlflow.set_tracking_uri(tracking_uri or "file:./mlruns")

def run(cmd, cwd=None):
    logger.info("RUN: %s", " ".join(cmd))
    try:
        subprocess.check_call(cmd, cwd=cwd)
    except subprocess.CalledProcessError as e:
        logger.error(f"Unexpected error: {str(e)}")

def bootstrap_pointers_and_data():
    """
    Clones only the DVC pointers under src/data/ at the exact ref,
    then pulls the actual snapshot from the DVC remote.
    """
    repo_url = os.environ["DVC_REPO_URL"]
    ref = os.environ["DVC_GIT_REF"]
    subdir = os.getenv("DVC_SUBDIR", "src/data") 
    workdir = Path("/workspace/repo")

    workdir.parent.mkdir(parents=True, exist_ok=True)

    # 1) Minimal clone + sparse checkout of just `subdir`
    #    --filter=blob:none avoids fetching file contents; --sparse limits to paths we set
    run(["git", "clone", "--filter=blob:none", "--sparse", repo_url, str(workdir)])
    run(["git", "fetch", "--depth", "1", "origin", ref], cwd=str(workdir))
    run(["git", "checkout", "FETCH_HEAD"], cwd=str(workdir))
    run(["git", "sparse-checkout", "set", subdir], cwd=str(workdir))

    # 2) Pull the DVC snapshot (uses IAM role / AWS env to access the DVC remote on S3)
    dvc_dir = workdir / subdir
    if not (dvc_dir / ".dvc").exists():
        logger.warning("No .dvc directory found under %s; is this a DVC subrepo?", dvc_dir)
    run(["dvc", "pull", "-j", str(os.cpu_count() or 4)], cwd=str(dvc_dir))


def load_data(data_dir, test_size, val_size, seed):
    X = pd.read_csv(Path(data_dir)/"features.csv")
    y = pd.read_csv(Path(data_dir)/"target.csv")

    # primary train/test split
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    # split train/val
    val_rel = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_rel, random_state=seed, stratify=y_trainval
    )
    return (X_train, X_val, X_test, y_train, y_val, y_test)


def compute_metrics(y_true, y_pred, y_prob):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }

def log_confusion_matrix(y_true, y_pred, title="Confusion matrix"):
    fig, ax = plt.subplots(figsize=(4, 4))
    cm = confusion_matrix(y_true, y_pred)
    im = ax.imshow(cm, cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center")
    mlflow.log_figure(fig, "plots/confusion_matrix.png")
    plt.close(fig)


def log_roc_curve(y_true, y_prob, title="ROC Curve"):
    fig, ax = plt.subplots(figsize=(4, 4))
    RocCurveDisplay.from_predictions(y_true, y_prob, ax=ax)
    ax.set_title(title)
    mlflow.log_figure(fig, "plots/roc_curve.png")
    plt.close(fig)


def train_one(model_obj, X_train, y_train, X_val, y_val):
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", model_obj)])
    t0 = time.time()
    pipe.fit(X_train, y_train)
    train_time = time.time() - t0

    y_val_pred = pipe.predict(X_val)
    # For ROC AUC we need prob of positive class 1
    if hasattr(pipe, "predict_proba"):
        y_val_prob = pipe.predict_proba(X_val)[:, 1]
    else:
        # fallback for models without predict_proba
        y_val_prob = pipe.decision_function(X_val)
        # scale to [0,1] via logistic if needed
        y_val_prob = 1 / (1 + np.exp(-y_val_prob))

    metrics = compute_metrics(y_val, y_val_pred, y_val_prob)
    return pipe, metrics, train_time

# ---------- Training ----------
def train(
    data_dir: Optional[str],
    test_size: float,
    random_state: int,
    model_params
):
    dataset_source = str(data_dir)

    # Metadata / lineage
    git_sha = os.environ["DVC_GIT_REF"]
    # Try to read DVC md5 if you’re using the subrepo under src/data
    dvc_md5 = os.environ["dvc_md5"]
    penalty = model_params['logreg']['penalty']
    C = model_params['logreg']['C']
    max_iter = model_params['logreg']['max_iter']
    rf_n_estimators = model_params["random_forest"]['n_estimators']
    rf_max_depth = model_params["random_forest"]['max_depth']

    # ----- Log schema snapshot as an artifact (dtypes, basic stats)
    schema = {
        "n_features": X_train.shape[1],
        # "feature_names": feature_names,
        "dtypes": {c: str(t) for c, t in X_train.dtypes.items()},
    }

    # Models to try
    candidates = {
        "logreg": LogisticRegression(
            penalty=penalty, C=C, max_iter=max_iter, n_jobs=None
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=rf_n_estimators,
            max_depth=rf_max_depth,
            n_jobs=-1,
            random_state= random_state,
        ),
    }

    best = {"name": None, "pipe": None, "metrics": {"roc_auc": -1.0}, "train_time": None}
    for name, model in candidates.items():
        with mlflow.start_run(run_name=name) as run:
            pipe, m, train_time = train_one(model, X_train, y_train, X_val, y_val)

            # Log params & metrics
            mlflow.log_params({
                "model": name,
                "seed": random_state,
                "test_size": test_size,
                "val_size": test_size,
                **({"penalty": penalty, "C":C} if name == "logreg" else {}),
                **({"rf_n_estimators": rf_n_estimators, "rf_max_depth": rf_max_depth} if name == "random_forest" else {}),
            })
            mlflow.set_tags(
                {
                    "code_sha": git_sha,
                    "data_source": dataset_source,
                    "dvc_md5": dvc_md5,
                }
            )
            mlflow.log_metrics(m)
            mlflow.log_metric("train_time_sec", train_time)

            # Log schema snapshot
            mlflow.log_text(json.dumps(schema, indent=2), "artifacts/schema.json")

            # Plots
            # Compute probabilities for ROC/confusion on val
            if hasattr(pipe, "predict_proba"):
                y_val_prob = pipe.predict_proba(X_val)[:, 1]
            else:
                p = pipe.decision_function(X_val)
                y_val_prob = 1 / (1 + np.exp(-p))
            y_val_pred = pipe.predict(X_val)
            log_confusion_matrix(y_val, y_val_pred, title=f"{name} - Confusion (val)")
            log_roc_curve(y_val, y_val_prob, title=f"{name} - ROC (val)")

            # Save splits (once is fine; re-log OK)
            out = Path("artifacts"); out.mkdir(parents=True, exist_ok=True)
            X_test.assign(target=y_test).to_csv(out / "test_split.csv", index=False)
            X_val.assign(target=y_val).to_csv(out / "val_split.csv", index=False)
            X_train.assign(target=y_train).to_csv(out / "train_split.csv", index=False)
            mlflow.log_artifacts(str(out), artifact_path="data_splits")

            # Track the best on val ROC AUC
            if m["roc_auc"] > best["metrics"]["roc_auc"]:
                best = {"name": name, "pipe": pipe, "metrics": m, "train_time": train_time, "run_id": run.info.run_id}

            run = mlflow.active_run()
            logger.info(f"Run logged: run_id={run.info.run_id}")
            logger.info(f"Metrics: {json.dumps(m, indent=2)}")

def register_best_model(best, data_dir, registered_model_name):
    best_name = best["name"]
    best_pipe = best["pipe"]
    print(f"Best model: {best_name} (val ROC_AUC={best['metrics']['roc_auc']:.4f})")

    set_alias = "staging"
    dataset_source = str(data_dir)
    git_sha = os.environ["DVC_GIT_REF"]
    # Try to read DVC md5 if you’re using the subrepo under src/data
    dvc_md5 = os.environ["dvc_md5"]

    with mlflow.start_run(run_name=f"best_{best_name}") as run:
        # Test metrics
        if hasattr(best_pipe, "predict_proba"):
            y_test_prob = best_pipe.predict_proba(X_test)[:, 1]
        else:
            p = best_pipe.decision_function(X_test)
            y_test_prob = 1 / (1 + np.exp(-p))
        y_test_pred = best_pipe.predict(X_test)
        test_metrics = compute_metrics(y_test, y_test_pred, y_test_prob)
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

        # Model signature & example
        signature = infer_signature(X_test, y_test_pred)
        input_example = X_test.head(1)

        # Log the pipeline as an MLflow model and register it
        mlflow.sklearn.log_model(
            sk_model=best_pipe,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
            registered_model_name=registered_model_name,
            pip_requirements=[
                "scikit-learn==1.5.1",
                "pandas",
                "numpy",
                "mlflow",
            ],
            metadata={
                "data_source": dataset_source,
                "dvc_md5": dvc_md5,
                "code_sha": git_sha,
            },
        )

        # Optional: set an alias (e.g., "staging") on the newest version
        try:
            client = MlflowClient()
            latest = client.get_latest_versions(registered_model_name, stages=["None"])
            if latest:
                mv = latest[0]
                client.set_registered_model_alias(registered_model_name, set_alias, mv.version)
        except Exception as e:
            print(f"[WARN] Could not set alias '{set_alias}': {e}")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, default="/workspace/repo/src/data/dataset",
                   help="Folder or file with CSV/Parquet data. If absent, fallback to sklearn dataset.")
    p.add_argument("--test-size", type=float, default=0.15)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--penalty", default="l2")     # for logistic
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--max-iter", type=int, default=1000)
    p.add_argument("--experiment", type=str, default="breast_cancer_sklearn")
    p.add_argument("--rf_n_estimators", type=int, default=300)
    p.add_argument("--rf_max_depth", type=int, default=None)
    p.add_argument("--registered_model_name", default="breast_cancer_model")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model_parameters = {"logreg": {
                            "penalty": args.penalty, 
                            "C": args.C,
                            "max_iter":args.max_iter
                        },
                        "random_forest": {
                            "n_estimators":args.rf_n_estimators,
                            "max_depth":args.rf_max_depth
                        }
    }

    bootstrap_pointers_and_data()
    if args.data_dir:
        try:
            X_train, X_val, X_test, y_train, y_val, y_test = load_data(Path(args.data_dir), args.test_size, args.test_size, args.random_state)
        except Exception as e:
            logger.warning(f"Failed to load from {args.data_dir}: {e}.")
    configure_mlflow()
    mlflow.set_experiment(args.experiment)
    best_model = train(
        data_dir=args.data_dir,
        test_size=args.test_size,
        random_state=args.random_state,
        model_params = model_parameters
    )

    register_best_model(best_model, args.data_dir, args.registered_model_name)
