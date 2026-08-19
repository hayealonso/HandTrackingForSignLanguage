import pickle
from collections import deque, Counter

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

MODEL_PATH = "asl_model.pkl"
SMOOTHING_WINDOW = 12
HOLD_FRAMES_TO_CONFIRM = 6
CONFIDENCE_THRESHOLD = 0.7

NAVY_BG = (0, 0, 0) 
WHITE = (255, 255, 255) 
ACCENT = (102, 197, 255)
GRAY_LIGHT = (200, 200, 200)

FONT_PATH_BOLD = "C:/Windows/Fonts/segoeuib.ttf"
FONT_PATH_REGULAR = "C:/Windows/Fonts/segoeui.ttf"

font_letter = ImageFont.truetype(FONT_PATH_BOLD, 48)
font_caption = ImageFont.truetype(FONT_PATH_REGULAR, 32)
font_small = ImageFont.truetype(FONT_PATH_REGULAR, 18)
font_label = ImageFont.truetype(FONT_PATH_BOLD, 22)

# Estilo minimalista blanco para los landmarks, igual a la referencia
landmark_style = mp_drawing.DrawingSpec(color=WHITE, thickness=2, circle_radius=3)
connection_style = mp_drawing.DrawingSpec(color=WHITE, thickness=2)


def normalize_landmarks(hand_landmarks):
    coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
    wrist = coords[0].copy()
    coords -= wrist
    scale = np.max(np.linalg.norm(coords, axis=1))
    if scale > 0:
        coords /= scale
    return coords.flatten()


def draw_text_pil(frame_bgr, text, position, font, color_bgr):
    """Dibuja texto con una fuente TTF sobre un frame de OpenCV (BGR) usando PIL."""
    color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
    img_pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    draw.text(position, text, font=font, fill=color_rgb)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def main():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    cap = cv2.VideoCapture(index=0)
    cv2.namedWindow("ASL Subtitulos en Tiempo Real", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ASL Subtitulos en Tiempo Real", 1600, 900)

    recent_predictions = deque(maxlen=SMOOTHING_WINDOW)
    stable_letter = None
    stable_count = 0
    last_confirmed_letter = None
    caption = ""

    with mp_hands.Hands(
        model_complexity=0,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            current_letter = None
            confidence = 0.0

            # Panel izquierdo: video limpio, sin landmarks
            panel_clean = frame.copy()

            # Panel derecho: video con landmarks estilo minimalista
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

                features = normalize_landmarks(hand_landmarks).reshape(1, -1)
                probs = model.predict_proba(features)[0]
                best_idx = np.argmax(probs)
                confidence = probs[best_idx]
                predicted = model.classes_[best_idx]

                if confidence >= CONFIDENCE_THRESHOLD:
                    current_letter = predicted

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

            if (
                stable_letter is not None
                and stable_count == HOLD_FRAMES_TO_CONFIRM
                and stable_letter != last_confirmed_letter
            ):
                caption += stable_letter
                last_confirmed_letter = stable_letter

            if current_letter is None:
                last_confirmed_letter = None

            # --- Etiquetas de cada panel ---
            panel_clean = draw_text_pil(panel_clean, "Camara", (12, 10), font_label, WHITE)
            panel_landmarks = draw_text_pil(panel_landmarks, "Landmarks", (12, 10), font_label, WHITE)

            # --- Combinar ambos paneles lado a lado con un separador ---
            separator = np.full((frame.shape[0], 4, 3), NAVY_BG, dtype=np.uint8)
            combined = np.hstack([panel_clean, separator, panel_landmarks])

            # --- Franja superior: letra actual detectada + barra de progreso ---
            top_bar = np.full((90, combined.shape[1], 3), NAVY_BG, dtype=np.uint8)
            display_letter = current_letter if current_letter else "-"
            top_bar = draw_text_pil(top_bar, f"Letra: {display_letter}", (20, 5), font_letter, ACCENT)
            top_bar = draw_text_pil(
                top_bar, f"confianza {confidence:.2f}", (280, 32), font_small, GRAY_LIGHT
            )

            bar_x, bar_y, bar_w, bar_h = 20, 65, 220, 10
            cv2.rectangle(top_bar, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), WHITE, 1)
            fill_w = int(min(stable_count / HOLD_FRAMES_TO_CONFIRM, 1.0) * bar_w)
            cv2.rectangle(top_bar, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), ACCENT, -1)

            # --- Franja inferior: subtitulo/caption ---
            bottom_bar = np.full((80, combined.shape[1], 3), NAVY_BG, dtype=np.uint8)
            bottom_bar = draw_text_pil(bottom_bar, caption[-50:], (20, 12), font_caption, WHITE)
            bottom_bar = draw_text_pil(
                bottom_bar,
                "espacio (spacebar)   borrar (backspace)   clear (c)   salir (esc)",
                (20, 52),
                font_small,
                GRAY_LIGHT,
            )

            final_frame = np.vstack([top_bar, combined, bottom_bar])

            cv2.imshow("ASL Subtitulos en Tiempo Real", final_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            elif key == 32:
                caption += " "
                last_confirmed_letter = None
            elif key == 8:
                caption = caption[:-1]
            elif key == ord("c"):
                caption = ""

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()