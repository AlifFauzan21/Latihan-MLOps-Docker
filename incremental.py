from sklearn.datasets import load_iris
import pandas as pd
import mlflow
from sklearn.utils import shuffle
from joblib import dump
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# 1. Siapkan Data
iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
data['target'] = iris.target

data_baru = data.iloc[50:]
data_baru = shuffle(data_baru, random_state=42)

batch_size = 30
batches = [data_baru.iloc[i:i + batch_size] for i in range(0, len(data_baru), batch_size)]

# 2. Ambil Model Lama
print("Sedang mengambil model lama dari MLflow...")

try:
    experiment = mlflow.get_experiment_by_name("Online Training Iris")
    if experiment is None:
        raise Exception("Experiment tidak ditemukan")

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"]
    )

    if runs.empty:
        raise Exception("Belum ada runs")

    latest_run_id = runs.iloc[0]["run_id"]
    artifact_uri = f"runs:/{latest_run_id}/model_artifacts/online_model"
    model = mlflow.sklearn.load_model(artifact_uri)
    model_is_fresh = False
    print("Berhasil memuat model lama!")

except Exception as e:
    print(f"Model lama tidak ditemukan. Membuat model baru... ({e})")
    from sklearn.linear_model import SGDClassifier
    model = SGDClassifier(loss='log_loss', random_state=42)
    model_is_fresh = True

# 3. Online Learning dalam Parent Run
ALL_CLASSES = np.array([0, 1, 2])

mlflow.set_experiment("Online Training Iris")

with mlflow.start_run(run_name="Online_Training_Parent") as parent_run:

    print("\nMensimulasikan masuknya data baru...")
    for i, batch in enumerate(batches):
        with mlflow.start_run(run_name=f"Batch_{i+1}", nested=True):
            X_batch = batch.drop(columns=['target'])
            y_batch = batch['target']

            if model_is_fresh and i == 0:
                model.partial_fit(X_batch, y_batch, classes=ALL_CLASSES)
            else:
                model.partial_fit(X_batch, y_batch)

            batch_acc = model.score(X_batch, y_batch)
            mlflow.log_metric("batch_accuracy", batch_acc)
            print(f"Batch {i+1} selesai. Akurasi: {batch_acc:.4f}")

    # Log model final di parent run
    print("\nMenyimpan model final ke MLflow...")
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model_artifacts/online_model",
        input_example=X_batch.iloc[:5]
    )

    # Export Run ID ke file
    with open("run_id.txt", "w") as f:
        f.write(parent_run.info.run_id)
    print(f"Run ID: {parent_run.info.run_id}")

print("\nSeluruh proses Online Learning Selesai!")