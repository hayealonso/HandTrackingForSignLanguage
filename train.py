import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV

CSV_PATH = "landmarks_data_clean.csv"
MODEL_PATH = "asl_model.pkl"


def main():
    print("Cargando datos")
    df = pd.read_csv(CSV_PATH)
    print(f"Total de muestras: {len(df)}")
    print(df["label"].value_counts().sort_index())

    X = df.drop(columns=["label"]).values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\nBuscando mejores hiperparámetros (puede tardar un poco)...")
    param_grid = {
        "n_estimators": [200, 400],
        "max_depth": [None, 20, 30],
        "min_samples_leaf": [1, 2],
    }

    grid = GridSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1),
        param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1,
    )
    grid.fit(X_train, y_train)

    print(f"\nMejores parámetros: {grid.best_params_}")
    print(f"Mejor accuracy en cross-validation: {grid.best_score_:.4f}")

    best_model = grid.best_estimator_

    y_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy en set de validación (hold-out): {test_accuracy:.4f}")

    print("\nReporte de clasificación por letra:")
    print(classification_report(y_test, y_pred))

    labels_sorted = sorted(df["label"].unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)
    print("\nMatriz de confusión (filas=real, columnas=predicho):")
    print("     " + " ".join(f"{l:>3}" for l in labels_sorted))
    for i, row in enumerate(cm):
        print(f"{labels_sorted[i]:>3}: " + " ".join(f"{v:>3}" for v in row))

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(best_model, f)
    print(f"\nModelo guardado en {MODEL_PATH}")


if __name__ == "__main__":
    main()