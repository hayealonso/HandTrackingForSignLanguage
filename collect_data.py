import csv
import os

import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

SAMPLES_PER_LETTER = 500
CSV_PATH = "landmarks_data.csv"
FRAME_SKIP = 2  # procesa y guarda 1 de cada N frames

def normalize_landmarks(hand_landmarks):
    """Convierte los 21 landmarks en un vector de 63 valores,
    independiente de la posición y el tamaño de la mano en pantalla."""
    coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])

    # Centrar respecto a la muñeca (landmark 0)
    wrist = coords[0].copy()
    coords -= wrist

    # Escalar por la distancia máxima entre la muñeca y cualquier punto
    scale = np.max(np.linalg.norm(coords, axis=1))
    if scale > 0:
        coords /= scale

    return coords.flatten()  # 63 valores: x0,y0,z0,x1,y1,z1,...


def load_existing_counts():
    counts = {}
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH, "r", newline="") as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            for row in reader:
                label = row[0]
                counts[label] = counts.get(label, 0) + 1
    return counts


def main():
    counts = load_existing_counts()
    current_letter = None
    frame_count = 0

    file_exists = os.path.exists(CSV_PATH)
    csv_file = open(CSV_PATH, "a", newline="")
    writer = csv.writer(csv_file)
    if not file_exists:
        header = ["label"] + [f"c{i}" for i in range(63)]
        writer.writerow(header)

    cap = cv2.VideoCapture(index=0)

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
            frame_count += 1
            process_this_frame = (frame_count % FRAME_SKIP == 0)

            hand_landmarks = None

            if process_this_frame:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(frame_rgb)

                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=hand_landmarks,
                        connections=mp_hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                        connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style(),
                    )

                # Captura automatica si hay una letra seleccionada y no llego al limite
                if current_letter is not None and hand_landmarks is not None:
                    count = counts.get(current_letter, 0)
                    if count < SAMPLES_PER_LETTER:
                        row = [current_letter] + normalize_landmarks(hand_landmarks).tolist()
                        writer.writerow(row)
                        counts[current_letter] = count + 1

            # HUD con instrucciones
            letra_txt = current_letter if current_letter else "-"
            progreso = counts.get(current_letter, 0) if current_letter else 0
            cv2.putText(frame, f"Letra: {letra_txt}  ({progreso}/{SAMPLES_PER_LETTER})",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, "Presiona A-Z para elegir letra, ESC para salir",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv2.imshow("Recoleccion de datos ASL", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key < 256 and chr(key).isalpha():
                current_letter = chr(key).upper()

    cap.release()
    cv2.destroyAllWindows()
    csv_file.close()
    print("Datos guardados en", CSV_PATH)
    print(counts)


if __name__ == "__main__":
    main()