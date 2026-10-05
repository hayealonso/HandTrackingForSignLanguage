#Funciones compartidas para convertir los landmarks de MediaPipe en features.
#collect_data.py, train.py y main.py importan este módulo para que la normalización
#sea EXACTAMENTE la misma al capturar, al entrenar y al predecir.

import numpy as np

#MediaPipe Hands entrega 21 puntos por mano, cada uno con coordenadas (x, y, z)
NUM_LANDMARKS = 21
NUM_COORDS = NUM_LANDMARKS * 3

#Conjuntos de features disponibles (ver build_features)
FEATURE_SETS = ("raw", "dist")

#Índices (i, j) con i < j de todos los pares de puntos: 21*20/2 = 210 pares
_PAIR_I, _PAIR_J = np.triu_indices(NUM_LANDMARKS, k=1)


#Convierte los 21 landmarks en un vector de 63 valores que no depende de dónde
#está la mano en pantalla ni de qué tan cerca está de la cámara.
#Si handedness es "Left", la mano se refleja en el eje x para que quede igual que
#una mano derecha: así el modelo, entrenado con la derecha, también sirve a zurdos.
def normalize_landmarks(hand_landmarks, handedness=None):
    coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])

    #Centrar respecto a la muñeca (landmark 0)
    coords -= coords[0].copy()

    #Escalar por la distancia máxima entre la muñeca y cualquier otro punto
    scale = np.max(np.linalg.norm(coords, axis=1))
    if scale > 0:
        coords /= scale

    #Espejar la mano izquierda para que quede como una derecha
    if handedness == "Left":
        coords[:, 0] *= -1

    return coords.flatten()  #63 valores: x0,y0,z0,x1,y1,z1,...


#Devuelve "Left" o "Right" para la mano detectada, o None si MediaPipe no lo informa.
#MediaPipe asume que la imagen viene espejada (como una selfie); como nosotros hacemos
#cv2.flip(frame, 1) antes de procesar, la etiqueta coincide con la mano real de la persona.
def get_handedness(results, index=0):
    if not results.multi_handedness:
        return None
    return results.multi_handedness[index].classification[0].label


#Transforma una matriz (n_muestras, 63) en las features que recibe el clasificador:
#-"raw": las 63 coordenadas normalizadas, tal cual se guardan en el CSV
#-"dist": las 63 coordenadas + las 210 distancias entre cada par de puntos.
#Las distancias describen la forma de la mano (qué dedos se tocan, cuáles
#están doblados) y no cambian si la mano gira, lo que da más robustez.
def build_features(X, feature_set="raw"):
    X = np.asarray(X, dtype=np.float64).reshape(-1, NUM_COORDS)

    if feature_set == "raw":
        return X

    if feature_set == "dist":
        pts = X.reshape(-1, NUM_LANDMARKS, 3)
        dists = np.linalg.norm(pts[:, _PAIR_I, :] - pts[:, _PAIR_J, :], axis=-1)
        return np.hstack([X, dists])

    raise ValueError(f"feature_set desconocido: {feature_set!r}. Opciones: {', '.join(FEATURE_SETS)}")
