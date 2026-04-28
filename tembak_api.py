import requests

# Cari baris ini di tembak_api.py
url = "http://127.0.0.1:5004/invocations"
headers = {"Content-Type": "application/json"}

# 1. Siapkan Pesanan (Data Format JSON)
payload = {
    "dataframe_split": {
        "columns": ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"],
        "data": [
            [5.1, 3.5, 1.4, 0.2],  # Bunga A
            [6.7, 3.0, 5.2, 2.3]   # Bunga B
        ]
    }
}

print("Mengirim data pesanan ke Pelayan API (Port 5002)...")

# 2. Tembak data ke API (POST Request)
try:
    response = requests.post(url, json=payload, headers=headers)

    # 3. Tangkap dan tampilkan respons
    if response.status_code == 200:
        hasil_api = response.json().get("predictions")
        nama_bunga = ["Setosa", "Versicolor", "Virginica"]
        
        print("\n[API SERVER MERESPONS:]")
        for i, hasil in enumerate(hasil_api):
            print(f"-> Bunga {i+1} ditebak sebagai: {nama_bunga[int(hasil)]} (Kelas {hasil})")
    else:
        print(f"Error! Pelayan kebingungan. Status: {response.status_code}")
        print(response.text)
except Exception as e:
    print(f"Gagal menghubungi server API: {e}")