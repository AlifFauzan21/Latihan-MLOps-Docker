from sklearn.datasets import load_iris
import pandas as pd
import mlflow
from sklearn.linear_model import SGDClassifier
from joblib import dump

# 1. Siapkan Data (Kita pakai 50 data pertama saja untuk hari ke-1)
iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
data['target'] = iris.target

X_awal = data.iloc[:50].drop(columns=['target'])
y_awal = data.iloc[:50]['target']
classes = data['target'].unique() # Wajib ada untuk partial_fit

# 2. Sambungkan ke MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("Online Training Iris")

# 3. Siapkan Algoritma Dinamis (SGD)
model = SGDClassifier(loss='log_loss', learning_rate='adaptive', eta0=0.01, max_iter=10000, random_state=42)

# 4. Mulai Latih & Catat!
print("Memulai pelatihan awal (Model Bayi)...")
with mlflow.start_run(run_name="1_Inisialisasi_Awal"):
    mlflow.autolog()
    
    # Latih model dengan partial_fit
    model.partial_fit(X_awal, y_awal, classes=classes)
    
    # Cek & Catat Akurasi
    accuracy = model.score(X_awal, y_awal)
    mlflow.log_metric("accuracy", accuracy)
    print(f"Selesai! Akurasi Awal: {accuracy:.4f}")
    
    # Simpan Model ke MLflow
    dump(model, "online_model.joblib")
    mlflow.log_artifact("online_model.joblib", artifact_path="model_artifacts")
    mlflow.sklearn.log_model(sk_model=model, artifact_path="online_model", input_example=X_awal.iloc[:5])