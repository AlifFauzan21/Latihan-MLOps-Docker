import mlflow
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# 1. Sambungkan ke MLflow Server
mlflow.set_tracking_uri("http://127.0.0.1:5000/")

# 2. Cari eksperimen
experiment_name = "Online Training Iris"
print(f"Mencari eksperimen: {experiment_name}...")
experiment = mlflow.get_experiment_by_name(experiment_name)

if experiment is None:
    print("Eksperimen tidak ditemukan. Pastikan server MLflow jalan dan nama eksperimen benar.")
    exit()

# 3. Ambil run terbaru
runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["start_time DESC"]
)

if runs.empty:
    print("Tidak ada run ditemukan di eksperimen ini.")
    exit()

latest_run_id = runs.iloc[0]["run_id"]  # ✅ FIX DI SINI
print(f"Berhasil menemukan Model! Run ID: {latest_run_id}\n")

# 4. Data dummy
data_bunga_baru = {
    "columns": ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"],
    "data": [
        [5.1, 3.5, 1.4, 0.2],
        [6.7, 3.0, 5.2, 2.3],
        [5.9, 3.0, 4.2, 1.5]
    ]
}

# 5. Validasi input
model_uri = f"runs:/{latest_run_id}/online_model"

try:
    from mlflow.models import validate_serving_input
    validate_serving_input(model_uri, {"dataframe_split": data_bunga_baru})
    print("Validasi Format Data: SUKSES!\n")
except Exception as e:
    print(f"Peringatan Validasi: {e}\n")

# 6. Load model
print("Memuat model ke dalam memori...")
model = mlflow.pyfunc.load_model(model_uri)

# 7. Convert ke DataFrame
df_bunga = pd.DataFrame(
    data_bunga_baru["data"],
    columns=data_bunga_baru["columns"]
)

# 8. Prediksi
print("\nMeminta model memprediksi...")
prediksi = model.predict(df_bunga)

# 9. Mapping label
print("\n=== HASIL PREDIKSI ===")
nama_bunga = ["Setosa", "Versicolor", "Virginica"]

for i, hasil in enumerate(prediksi):
    print(f"Bunga Misterius {i+1} diprediksi sebagai: {nama_bunga[int(hasil)]} (Kelas {hasil})")