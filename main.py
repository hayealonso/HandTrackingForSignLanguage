#Etapa 4 del pipeline: reconocimiento en tiempo real.
#Abre la cámara, detecta la mano, predice la letra con el modelo entrenado y arma un
#subtítulo letra por letra. La interfaz muestra la cámara limpia y la vista con landmarks.
#Uso: python main.py [--model asl_model.pkl] [--camera 0]

import argparse
import os
import pickle
from collections import Counter, deque

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from hand_features import build_features, get_handedness, normalize_landmarks

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

MODEL_PATH = "asl_model.pkl"
WINDOW_NAME = "Subtitulos de lengua de senas en tiempo real"

#Parámetros de estabilidad de la predicción
SMOOTHING_WINDOW = 12        #cantidad de predicciones recientes sobre las que se vota la letra
HOLD_FRAMES_TO_CONFIRM = 6   #frames que la letra debe mantenerse para escribirla en el subtítulo
CONFIDENCE_THRESHOLD = 0.7   #probabilidad mínima para aceptar una predicción

#Colores en formato BGR (el que usa OpenCV)
NAVY_BG = (0, 0, 0)
WHITE = (255, 255, 255)
ACCENT = (102, 197, 255)
GRAY_LIGHT = (200, 200, 200)

#Fuentes: se prueba primero Segoe UI (Windows), luego DejaVu (Linux) y Helvetica/Arial (macOS).
#Si no se encuentra ninguna, se usa la fuente por defecto de PIL para que el programa no se caiga.
FONT_CANDIDATES_BOLD = [
    "C:/Windows/Fonts/segoeuib.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
]
FONT_CANDIDATES_REGULAR = [
    "C:/Windows/Fonts/segoeui.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
]


#Carga la primera fuente TTF que exista en el sistema
def load_font(candidates, size):
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size)


font_letter = load_font(FONT_CANDIDATES_BOLD, 48)
font_caption = load_font(FONT_CANDIDATES_REGULAR, 32)
font_small = load_font(FONT_CANDIDATES_REGULAR, 18)
font_label = load_font(FONT_CANDIDATES_BOLD, 22)

#Estilo minimalista blanco para dibujar los landmarks
landmark_style = mp_drawing.DrawingSpec(color=WHITE, thickness=2, circle_radius=3)
connection_style = mp_drawing.DrawingSpec(color=WHITE, thickness=2)


#Carga el modelo entrenado. train.py guarda un diccionario con el modelo y el conjunto de
#features; si el archivo es de la versión antigua (solo el Random Forest), se asume "raw".
def load_model(path):
    if not os.path.exists(path):
        raise SystemExit(f"No se encontró {path}. Entrena un modelo primero con: python train.py")
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    if not isinstance(bundle, dict):
        bundle = {"model": bundle, "model_name": "rf", "feature_set": "raw"}
    return bundle


#Dibuja varios textos con fuentes TTF sobre un frame BGR. Se hace en una sola conversión a PIL
#por frame (convertir la imagen completa es caro, y antes se hacía una vez por cada texto).
#texts es una lista de tuplas (texto, (x, y), fuente, color_bgr)
def draw_texts_pil(frame_bgr, texts):
    img_pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    for text, position, font, color_bgr in texts:
        color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
        draw.text(position, text, font=font, fill=color_rgb)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def parse_args():
    parser = argparse.ArgumentParser(description="Reconocimiento de letras en tiempo real.")
    parser.add_argument("--model", default=MODEL_PATH, help="archivo del modelo entrenado")
    parser.add_argument("--camera", type=int, default=0, help="índice de la cámara (0 = la principal)")
    return parser.parse_args()


def main():
    args = parse_args()
    bundle = load_model(args.model)
    model = bundle["model"]
    feature_set = bundle["feature_set"]
    print(f"Modelo cargado: {bundle['model_name']} (features: {feature_set})")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"No se pudo abrir la cámara {args.camera}. Prueba con --camera 1")
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1600, 900)

    recent_predictions = deque(maxlen=SMOOTHING_WINDOW)  #últimas predicciones para el voto
    stable_letter = None          #letra ganadora del voto en el frame actual
    stable_count = 0              #cuántos frames seguidos lleva ganando esa letra
    last_confirmed_letter = None  #última letra escrita, para no repetirla mientras se mantiene la seña
    caption = ""                  #subtítulo acumulado

    with mp_hands.Hands(
        model_complexity=0,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:
        while True:
            success, frame = cap.read()
            if not success:
                print("Se perdió la señal de la cámara.")
                break

            #Espejar la imagen y pasarla a RGB para MediaPipe
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            current_letter = None
            confidence = 0.0

            #Panel izquierdo: video limpio. Panel derecho: video con landmarks
            panel_clean = frame.copy()
            panel_landmarks = frame.copy()

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(
                    image=panel_landmarks,
                    landmark_list=hand_landmarks,
                    connections=mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=landmark_style,
                    connection_drawing_spec=connection_style,
                )

                #Mismas features que en el entrenamiento; la mano izquierda se espeja a derecha
                handedness = get_handedness(results)
                coords = normalize_landmarks(hand_landmarks, handedness)
                features = build_features(coords, feature_set)

                probs = model.predict_proba(features)[0]
                best_idx = np.argmax(probs)
                confidence = probs[best_idx]
                predicted = model.classes_[best_idx]

                #Solo se acepta la predicción si el modelo está suficientemente seguro
                if confidence >= CONFIDENCE_THRESHOLD:
                    current_letter = predicted

            #Suavizado: la letra "estable" es la más votada entre las últimas predicciones válidas
            recent_predictions.append(current_letter)
            valid_recent = [p for p in recent_predictions if p is not None]
            if valid_recent:
                majority_letter, _ = Counter(valid_recent).most_common(1)[0]
            else:
                majority_letter = None

            if majority_letter == stable_letter and majority_letter is not None:
                stable_count += 1
            else:
                stable_letter = majority_letter
                stable_count = 1

            #Confirmación: si la letra se mantuvo HOLD_FRAMES_TO_CONFIRM frames, se escribe una vez
            if (
                stable_letter is not None
                and stable_count == HOLD_FRAMES_TO_CONFIRM
                and stable_letter != last_confirmed_letter
            ):
                caption += stable_letter
                last_confirmed_letter = stable_letter

            #Si se baja la mano (o no hay predicción segura) se permite repetir la letra: "LL", "RR"
            if current_letter is None:
                last_confirmed_letter = None

            #Combinar ambos paneles lado a lado con un separador
            separator = np.full((frame.shape[0], 4, 3), NAVY_BG, dtype=np.uint8)
            combined = np.hstack([panel_clean, separator, panel_landmarks])
            panel_width = frame.shape[1] + separator.shape[1]

            #Franja superior: barra de progreso hacia la confirmación de la letra
            top_h, bottom_h = 90, 80
            top_bar = np.full((top_h, combined.shape[1], 3), NAVY_BG, dtype=np.uint8)
            bar_x, bar_y, bar_w, bar_h = 20, 65, 220, 10
            cv2.rectangle(top_bar, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), WHITE, 1)
            fill_w = int(min(stable_count / HOLD_FRAMES_TO_CONFIRM, 1.0) * bar_w)
            cv2.rectangle(top_bar, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), ACCENT, -1)

            #Franja inferior: fondo para el subtítulo
            bottom_bar = np.full((bottom_h, combined.shape[1], 3), NAVY_BG, dtype=np.uint8)

            final_frame = np.vstack([top_bar, combined, bottom_bar])

            #Todos los textos se dibujan juntos al final (coordenadas sobre final_frame)
            display_letter = current_letter if current_letter else "-"
            bottom_y = top_h + combined.shape[0]
            final_frame = draw_texts_pil(final_frame, [
                (f"Letra: {display_letter}", (20, 5), font_letter, ACCENT),
                (f"confianza {confidence:.2f}", (280, 32), font_small, GRAY_LIGHT),
                ("Cámara", (12, top_h + 10), font_label, WHITE),
                ("Landmarks", (panel_width + 12, top_h + 10), font_label, WHITE),
                (caption[-50:], (20, bottom_y + 12), font_caption, WHITE),
                ("espacio (spacebar)   borrar (backspace)   limpiar (c)   salir (esc)",
                 (20, bottom_y + 52), font_small, GRAY_LIGHT),
            ])

            cv2.imshow(WINDOW_NAME, final_frame)

            #Controles de teclado
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  #ESC: salir
                break
            elif key == 32:  #espacio: separa palabras y permite repetir la última letra
                caption += " "
                last_confirmed_letter = None
            elif key == 8:  #backspace: borra el último carácter
                caption = caption[:-1]
            elif key == ord("c"):  #c: limpia todo el subtítulo
                caption = ""

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
