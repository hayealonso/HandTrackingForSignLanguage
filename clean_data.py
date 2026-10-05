#Etapa 2 del pipeline: limpieza del dataset.
#Lee el CSV crudo de collect_data.py, descarta filas con etiquetas inválidas o con
#valores faltantes y guarda un CSV limpio listo para entrenar.
#Uso: python clean_data.py

import pandas as pd

CSV_PATH = "landmarks_data.csv"
CLEAN_PATH = "landmarks_data_clean.csv"
VALID_LETTERS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

#Se lee en latin1 porque el CSV original trae una etiqueta corrupta (byte 0x9F, la "Ÿ"
#que generaba un bug ya corregido en collect_data.py). latin1 nunca falla al decodificar
#y para las letras A-Z es idéntico a utf-8.
df = pd.read_csv(CSV_PATH, encoding="latin1")

print("Todas las etiquetas encontradas (incluyendo las inválidas):")
print(df["label"].value_counts())

#Nos quedamos solo con filas cuya etiqueta sea una letra A-Z y que no tengan valores vacíos
labels = df["label"].astype(str).str.upper()
mask_valid = labels.isin(VALID_LETTERS) & df.notna().all(axis=1)

print(f"\nFilas totales: {len(df)}")
print(f"Filas válidas: {mask_valid.sum()}")
print(f"Filas descartadas: {(~mask_valid).sum()}")

df_clean = df[mask_valid].copy()
df_clean["label"] = labels[mask_valid]

print("\nConteo final por letra:")
print(df_clean["label"].value_counts().sort_index())

df_clean.to_csv(CLEAN_PATH, index=False, encoding="utf-8")
print(f"\nGuardado limpio en {CLEAN_PATH}")
