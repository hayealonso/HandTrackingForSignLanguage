#Etapa 3 del pipeline: entrenamiento.
#Entrena el clasificador elegido sobre los landmarks limpios, lo evalúa con un set de
#prueba que no comparte bloques de video con el de entrenamiento y guarda el modelo.
#Uso: python train.py                    (modelo y features por defecto)
#python train.py --model rf --features raw
#python train.py --model mlp --tune  (búsqueda de hiperparámetros)

import argparse
import pickle

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold

from classifiers import CLASSIFIERS, PARAM_GRIDS, make_classifier
from hand_features import FEATURE_SETS, build_features

CSV_PATH = "landmarks_data_clean.csv"
MODEL_PATH = "asl_model.pkl"
PLOT_PATH = "confusion_matrix.png"
DEFAULT_MODEL = "svm"
DEFAULT_FEATURES = "dist"

#Filas consecutivas de una misma letra que se consideran un "bloque" (~3 s de video).
#Los frames seguidos son casi idénticos; si unos quedan en train y sus vecinos en test,
#la accuracy sale inflada. Por eso train y test se separan por bloques, no por filas.
GROUP_BLOCK_SIZE = 50


#Asigna un id de bloque a cada fila: cambia cada GROUP_BLOCK_SIZE filas o cuando cambia la letra
def make_groups(labels, block_size=GROUP_BLOCK_SIZE):
    groups = np.zeros(len(labels), dtype=int)
    group_id, rows_in_block = 0, 0
    for i in range(1, len(labels)):
        rows_in_block += 1
        if labels[i] != labels[i - 1] or rows_in_block >= block_size:
            group_id += 1
            rows_in_block = 0
        groups[i] = group_id
    return groups


#Imprime la matriz de confusión en consola y, si matplotlib está disponible, la guarda como imagen
def report_confusion(y_true, y_pred, labels, plot_path):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print("\nMatriz de confusión (filas = real, columnas = predicho):")
    print("     " + " ".join(f"{l:>3}" for l in labels))
    for i, row in enumerate(cm):
        print(f"{labels[i]:>3}: " + " ".join(f"{v:>3}" for v in row))

    try:
        import matplotlib
        matplotlib.use("Agg")  #backend sin ventana: solo guarda el archivo
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(11, 10))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Matriz de confusión (set de prueba)")
    ax.set_xlabel("Letra predicha")
    ax.set_ylabel("Letra real")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)
    print(f"Matriz de confusión guardada en {plot_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Entrena el clasificador de letras.")
    parser.add_argument("--model", choices=CLASSIFIERS, default=DEFAULT_MODEL,
                        help=f"clasificador a usar (por defecto: {DEFAULT_MODEL})")
    parser.add_argument("--features", choices=FEATURE_SETS, default=DEFAULT_FEATURES,
                        help=f"conjunto de features (por defecto: {DEFAULT_FEATURES})")
    parser.add_argument("--tune", action="store_true",
                        help="buscar hiperparámetros con validación cruzada (más lento)")
    parser.add_argument("--csv", default=CSV_PATH, help="CSV limpio de entrada")
    parser.add_argument("--out", default=MODEL_PATH, help="archivo donde guardar el modelo")
    return parser.parse_args()


def main():
    args = parse_args()

    print("Cargando datos")
    df = pd.read_csv(args.csv)
    print(f"Total de muestras: {len(df)}")
    print(df["label"].value_counts().sort_index())

    y = df["label"].values
    X = build_features(df.drop(columns=["label"]).values, args.features)
    groups = make_groups(y)
    print(f"\nModelo: {args.model} | features: {args.features} ({X.shape[1]} columnas)")

    #Separar ~20% de los bloques para prueba (5 folds -> nos quedamos con el primero)
    #StratifiedGroupKFold mantiene la proporción de letras y nunca parte un bloque en dos
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, test_idx = next(cv.split(X, y, groups))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model = make_classifier(args.model)

    if args.tune:
        print("\nBuscando mejores hiperparámetros (puede tardar un poco)...")
        grid = GridSearchCV(
            model,
            PARAM_GRIDS[args.model],
            cv=StratifiedGroupKFold(n_splits=5),
            scoring="accuracy",
            n_jobs=-1,
            verbose=1,
        )
        grid.fit(X_train, y_train, groups=groups[train_idx])
        print(f"\nMejores parámetros: {grid.best_params_}")
        print(f"Mejor accuracy en validación cruzada: {grid.best_score_:.4f}")
        model = grid.best_estimator_
    else:
        print("\nEntrenando...")
        model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy en el set de prueba (bloques no vistos): {test_accuracy:.4f}")

    print("\nReporte de clasificación por letra:")
    print(classification_report(y_test, y_pred, zero_division=0))

    labels_sorted = sorted(df["label"].unique())
    report_confusion(y_test, y_pred, labels_sorted, PLOT_PATH)

    #Ya evaluado, se reentrena con TODOS los datos para que el modelo final aprenda de todo
    print("\nReentrenando con el dataset completo...")
    final_model = clone(model).fit(X, y)

    #Se guarda un diccionario y no solo el modelo: main.py necesita saber qué features usar
    bundle = {
        "model": final_model,
        "model_name": args.model,
        "feature_set": args.features,
        "test_accuracy": test_accuracy,
    }
    with open(args.out, "wb") as f:
        pickle.dump(bundle, f)
    print(f"\nModelo guardado en {args.out}")


if __name__ == "__main__":
    main()
