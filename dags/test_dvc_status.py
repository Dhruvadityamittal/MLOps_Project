import subprocess
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

BASE_DIR = "/opt/airflow/data"  # where your repo & data are
DATA_FILE = "IMDB_Dataset.csv.dvc"  # DVC metadata file, not raw CSV

default_args = {
    "owner": "airflow",
    "start_date": datetime(2024, 1, 1),
    "retries": 1,
}

def check_dvc_status(ti):
    """Check if DVC-tracked dataset has changed."""
    try:
        # Run DVC status inside repo
        result = subprocess.run(
            ["dvc", "status"],
            cwd=BASE_DIR,
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"DVC status failed: {result.stderr}")

        if result.stdout.strip():
            print("📂 Data has changed, will run pipeline.")
            ti.xcom_push(key='Data Status', value= 'Data Changed')
            return True
        else:
            print("✅ No changes in DVC-tracked data. Skipping training.")
            ti.xcom_push(key='Data Status', value= 'Data Not Changed')
            # Stop Airflow DAG execution here
            raise AirflowSkipException("No data changes detected.")
    except FileNotFoundError:
        raise RuntimeError("DVC not installed inside Airflow container.")

with DAG(
    dag_id="dvc_xgboost_pipeline",
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
) as dag:

    check_data_task = PythonOperator(
        task_id="check_dvc_status",
        python_callable=check_dvc_status
    )

    # Then your existing train & evaluate tasks
    # check_data_task >> task_train >> task_eval
