import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

# Fitur: [Luas_Hutan_Awal_Ha, Indeks_Vegetasi_NDVI, Aktivitas_Penebangan_Terdeteksi_km2]
data_satelit = {
    'Wilayah_ID': [f'ZONA_{chr(65+i)}' for i in range(10)],
    'Luas_Hutan_Awal_Ha': [5000, 4200, 6100, 3800, 5500, 4900, 6000, 2500, 5300, 4800],
    'Indeks_Vegetasi_NDVI': [0.75, 0.68, 0.80, 0.45, 0.78, 0.70, 0.82, 0.30, 0.76, 0.72],
    'Aktivitas_Penebangan_km2': [0.2, 0.5, 0.1, 4.5, 0.3, 0.4, 0.1, 6.8, 0.2, 0.3]
}

df = pd.DataFrame(data_satelit)

 
X = df[['Luas_Hutan_Awal_Ha', 'Indeks_Vegetasi_NDVI', 'Aktivitas_Penebangan_km2']]
model_ai = IsolationForest(contamination=0.2, random_state=42)
df['Status_Anomali'] = model_ai.fit_predict(X)


df['Keterangan_Zona'] = df['Status_Anomali'].apply(lambda x: '🚨 POTENSI DEFORESTASI TINGGI' if x == -1 else '✅ Hutan Lestari/Aman')

print("=== HASIL ANALISIS AI & SATELIT (DEFORESTASI) ===")
print(df[['Wilayah_ID', 'Indeks_Vegetasi_NDVI', 'Aktivitas_Penebangan_km2', 'Keterangan_Zona']])
print("\n-----------------------------------------------------")
zona_bahaya = df[df['Status_Anomali'] == -1]['Wilayah_ID'].tolist()
print(f"Zona yang Membutuhkan Perhatian & Patroli Darurat: {', '.join(zona_bahaya)}")