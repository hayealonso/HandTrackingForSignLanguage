import pandas as pd

CSV_PATH = "landmarks_data.csv"
CLEAN_PATH = "landmarks_data_clean.csv"

df = pd.read_csv(CSV_PATH, encoding="latin1")

print("Todas las etiquetas encontradas (incluyendo raras):")
print(df["label"].value_counts())

# Nos quedamos solo con filas cuya etiqueta sea una sola letra A-Z
valid_letters = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
mask_valid = df["label"].astype(str).str.upper().isin(valid_letters)

print(f"\nFilas totales: {len(df)}")
print(f"Filas válidas: {mask_valid.sum()}")
print(f"Filas descartadas: {(~mask_valid).sum()}")

df_clean = df[mask_valid].copy()
df_clean["label"] = df_clean["label"].str.upper()

print("\nConteo final por letra:")
print(df_clean["label"].value_counts().sort_index())

df_clean.to_csv(CLEAN_PATH, index=False, encoding="utf-8")
print(f"\nGuardado limpio en {CLEAN_PATH}")