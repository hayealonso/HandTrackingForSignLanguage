#Catálogo de clasificadores disponibles para train.py.
#Para agregar uno nuevo basta con sumarlo a CLASSIFIERS (y opcionalmente a PARAM_GRIDS).
#Todos deben implementar predict_proba, porque main.py usa la probabilidad como "confianza".

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

RANDOM_STATE = 42


#Random Forest: el modelo original del proyecto. Robusto y sin escalado, pero el archivo pesa mucho
def _rf():
    return RandomForestClassifier(
        n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE
    )


#Extra Trees: parecido al RF pero con cortes aleatorios; suele generalizar un poco mejor y entrena más rápido
def _et():
    return ExtraTreesClassifier(
        n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE
    )


#SVM con kernel RBF: muy bueno con pocas muestras y features continuas. Necesita escalar los datos
#La SVM no entrega probabilidades por sí sola: CalibratedClassifierCV las estima con validación
#cruzada interna (reemplaza a SVC(probability=True), que scikit-learn marcó como obsoleto en la 1.9)
def _svm():
    return make_pipeline(
        StandardScaler(),
        CalibratedClassifierCV(
            SVC(C=10, gamma="scale", class_weight="balanced", random_state=RANDOM_STATE),
            ensemble=False,
        ),
    )


#k vecinos más cercanos: el más simple de todos, útil como referencia
def _knn():
    return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5, weights="distance"))


#Red neuronal (perceptrón multicapa) de dos capas ocultas: la opción "deep" sin dependencias extra
#early_stopping separa un 10% interno para detener el entrenamiento antes de sobreajustar
def _mlp():
    return make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(256, 128),
            alpha=1e-3,
            max_iter=500,
            early_stopping=True,
            random_state=RANDOM_STATE,
        ),
    )


CLASSIFIERS = {
    "rf": _rf,
    "et": _et,
    "svm": _svm,
    "knn": _knn,
    "mlp": _mlp,
}

#Grillas pequeñas de hiperparámetros para la búsqueda opcional (python train.py --tune)
#Los nombres con "__" apuntan al paso correspondiente dentro del pipeline (p. ej. calibratedclassifiercv__estimator__C)
PARAM_GRIDS = {
    "rf": {"n_estimators": [200, 400], "max_depth": [None, 20, 30], "min_samples_leaf": [1, 2]},
    "et": {"n_estimators": [200, 400], "max_depth": [None, 20, 30], "min_samples_leaf": [1, 2]},
    "svm": {
        "calibratedclassifiercv__estimator__C": [1, 10, 100],
        "calibratedclassifiercv__estimator__gamma": ["scale", 0.01, 0.001],
    },
    "knn": {"kneighborsclassifier__n_neighbors": [3, 5, 9, 15]},
    "mlp": {
        "mlpclassifier__hidden_layer_sizes": [(128,), (256, 128), (256, 128, 64)],
        "mlpclassifier__alpha": [1e-4, 1e-3, 1e-2],
    },
}


#Devuelve una instancia nueva (sin entrenar) del clasificador pedido
def make_classifier(name):
    if name not in CLASSIFIERS:
        raise ValueError(f"Modelo desconocido: {name!r}. Opciones: {', '.join(CLASSIFIERS)}")
    return CLASSIFIERS[name]()
