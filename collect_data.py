#Etapa 1 del pipeline: captura de datos.
#Abre la cámara, detecta la mano con MediaPipe y guarda sus landmarks normalizados en
#un CSV, etiquetados con la letra que elijas con el teclado.
#Uso: python collect_data.py [--samples 500] [--camera 0]

import argparse
import csv
import os

import cv2
import mediapipe as mp

from hand_features import NUM_COORDS, get_handedness, normalize_landmarks

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

SAMPLES_PER_LETTER = 500  #muestras a juntar por letra antes de dejar de grabar
CSV_PATH = "landmarks_data.csv"
FRAME_SKIP = 2  #procesa y guarda 1 de cada N frames, para que las muestras no sean casi idénticas

#Códigos de tecla que entrega cv2.waitKey
KEY_ESC = 27
KEY_SPACE = 32
VALID_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


#Cuenta cuántas muestras hay ya por letra, para poder retomar una captura a medias
def load_existing_counts(csv_path):
    counts = {}
    if os.path.exists(csv_path):
        with open(csv_path, "r", newline="", encoding="latin1") as f:
            reader = csv.reader(f)
            next(reader, None)  #saltar el encabezado
            for row in reader:
                label = row[0]
                counts[label] = counts.get(label, 0) + 1
    return counts


def parse_args():
    parser = argparse.ArgumentParser(description="Captura landmarks de la mano etiquetados por letra.")
    parser.add_argument("--samples", type=int, default=SAMPLES_PER_LETTER, help="muestras por letra")
    parser.add_argument("--camera", type=int, default=0, help="índice de la cámara (0 = la principal)")
    parser.add_argument("--csv", default=CSV_PATH, help="archivo CSV de salida")
    return parser.parse_args()


def main():
    args = parse_args()
    counts = load_existing_counts(args.csv)
    current_letter = None
    recording = False
    frame_count = 0

    #Se abre en modo "a" (append) para no borrar lo que ya se había capturado
    file_exists = os.path.exists(args.csv)
    csv_file = open(args.csv, "a", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    if not file_exists:
        header = ["label"] + [f"c{i}" for i in range(NUM_COORDS)]
        writer.writerow(header)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"No se pudo abrir la cámara {args.camera}. Prueba con --camera 1")

    #model_complexity=0 usa el modelo liviano de MediaPipe (más rápido en CPU)
    #max_num_hands=1 porque el alfabeto dactilológico se hace con una sola mano
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

            #Espejar la imagen para que se vea como un espejo (más natural al hacer señas)
            frame = cv2.flip(frame, 1)
            frame_count += 1
            process_this_frame = frame_count % FRAME_SKIP == 0

            if process_this_frame:
                #MediaPipe trabaja en RGB, OpenCV entrega BGR
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

                    #Guardar la muestra solo si se está grabando y la letra no llegó al límite
                    count = counts.get(current_letter, 0)
                    if recording and current_letter is not None and count < args.samples:
                        handedness = get_handedness(results)
                        row = [current_letter] + normalize_landmarks(hand_landmarks, handedness).tolist()
                        writer.writerow(row)
                        counts[current_letter] = count + 1
                        if count + 1 == args.samples:
                            recording = False  #letra completa: se pausa sola

            #HUD con el estado de la captura (cv2.putText no soporta tildes)
            letra_txt = current_letter if current_letter else "-"
            progreso = counts.get(current_letter, 0) if current_letter else 0
            estado = "GRABANDO" if recording else "en pausa"
            color = (0, 0, 255) if recording else (0, 255, 0)
            cv2.putText(frame, f"Letra: {letra_txt}  ({progreso}/{args.samples})  {estado}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame, "A-Z: elegir letra   ESPACIO: grabar/pausar   ESC: salir",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv2.imshow("Recoleccion de datos", frame)

            #Cuando no se presiona nada, waitKey devuelve -1 (255 tras el & 0xFF).
            #Por eso se compara contra una lista explícita de letras en vez de usar isalpha(),
            #que aceptaba chr(255) = "ÿ" y cambiaba la etiqueta en cada frame sin tecla.
            key = cv2.waitKey(1) & 0xFF
            if key == KEY_ESC:
                break
            elif key == KEY_SPACE and current_letter is not None:
                recording = not recording
            elif chr(key).upper() in VALID_LETTERS:
                current_letter = chr(key).upper()
                recording = False  #al cambiar de letra se pausa, para acomodar la mano antes de grabar

    cap.release()
    cv2.destroyAllWindows()
    csv_file.close()
    print("Datos guardados en", args.csv)
    print(dict(sorted(counts.items())))


if __name__ == "__main__":
    main()
