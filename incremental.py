from sklearn.datasets import load_iris
import pandas as pd
import mlflow
from sklearn.utils import shuffle
from joblib import load, dump
import warnings
warnings.filterwarnings("ignore")

# 1. Siapkan Data Baru (Sisa data dari baris 50 sampai akhir)
iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
data['target'] = iris.target

data_baru = data.iloc[50:]
data_baru = shuffle(data_baru, random_state=42)  # Diacak biar natural

# Kita pecah data baru masuk per 30 baris (simulasi)
batch_size = 30
batches = [data_baru.iloc[i:i + batch_size] for i in range(0, len(data_baru), batch_size)]

# 2. Sambungkan ke MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("Online Training Iris")

# 3. AMBIL "INGATAN" MODEL LAMA DARI MLFLOW
print("Sedang mengambil model lama dari MLflow...")
experiment = mlflow.get_experiment_by_name("Online Training Iris")
runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"])

# ✅ FIX DI SINI
latest_run_id = runs.iloc[0]["run_id"]

artifact_uri = f"runs:/{latest_run_id}/model_artifacts/online_model.joblib"
local_path = mlflow.artifacts.download_artifacts(artifact_uri)
model = load(local_path)

# 4. CEKOKIN DATA BARU (ONLINE LEARNING)
print("\nMensimulasikan masuknya data baru...")
for i, batch in enumerate(batches):
    with mlflow.start_run(run_name=f"2_Update_Batch_{i+1}"):
        mlflow.autolog()
        
        X_batch = batch.drop(columns=['target'])
        y_batch = batch['target']
        
        # INI KUNCINYA: Belajar dari data baru tanpa reset
        model.partial_fit(X_batch, y_batch)
        
        # Cek Akurasi Terbaru
        batch_acc = model.score(X_batch, y_batch)
        mlflow.log_metric("batch_accuracy", batch_acc)
        print(f"Batch {i+1} dipelajari. Akurasi sekarang: {batch_acc:.4f}")
        
        # Simpan kembali model yang sudah lebih pintar ke MLflow
        dump(model, "online_model.joblib")
        mlflow.log_artifact("online_model.joblib", artifact_path="model_artifacts")
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="online_model",
            input_example=X_batch.iloc[:5]
        )

print("\nSeluruh proses Online Learning Selesai!")