# HandTrackingForSignLanguage

Reconocimiento en tiempo real del alfabeto dactilológico de la lengua de señas mediante detección de landmarks de la mano y clasificación con Random Forest.

## Descripción

Reconocimiento en tiempo real del alfabeto de la lengua de señas a partir de video de cámara web. En lugar de clasificar la imagen completa, el sistema extrae los 21 landmarks de la mano con MediaPipe y entrena un clasificador Random Forest sobre esas coordenadas. Esto reduce el costo computacional, permite ejecución fluida en tiempo real sobre CPU y hace viable entrenar con un dataset acotado.

El objetivo es facilitar la comunicación en entornos digitales —videollamadas, mensajería, formularios— entre personas sordas y personas que no manejan lengua de señas, traduciendo señas a texto sin requerir un intérprete presente.

El dataset fue construido íntegramente por el autor: contiene las letras del abecedario ejecutadas con la mano derecha y capturadas como poses estáticas.
<img width="607" height="603" alt="Captura de pantalla 2026-08-18 225331" src="https://github.com/user-attachments/assets/dfd2674b-8417-4727-9ac4-5f04543e8f82" />
## Cómo funciona

El proyecto se estructura como un pipeline de cuatro etapas:

| Archivo | Etapa | Descripción |
|---|---|---|
| `collect_data.py` | Captura | abre la cámara, detecta tu mano y guarda sus landmarks normalizados en un CSV cada vez que presionas una tecla A-Z, hasta juntar el número de muestras configurado por letra.. |
| `clean_data.py` | Limpieza | lee ese CSV, descarta filas con etiquetas inválidas o corruptas, corrige el encoding, y genera un CSV limpio. |
| `train.py` | Entrenamiento | entrena un Random Forest sobre los landmarks del CSV, prueba distintos hiperparámetros con validación cruzada, mide accuracy en un set de validación separado, y guarda el modelo entrenado. |
| `main.py` | Inferencia | abre la cámara, detecta la mano en tiempo real, predice la letra con el modelo entrenado, aplica suavizado y confirmación por tiempo sostenido para evitar errores, arma los subtítulos letra por letra, y lo muestra todo en una interfaz con vista doble (cámara limpia + landmarks) y texto estilizado. |

## Instalación

Requiere Python 3.10 o superior y una cámara web.

```bash
git clone https://github.com/hayealonso/HandTrackingForSignLanguage.git
cd HandTrackingForSignLanguage
python -m venv .venv
source .venv/bin/activate      # En Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Uso

Para ejecutar el reconocimiento con el modelo ya entrenado:

```bash
python main.py
```

Para reentrenar desde cero con tus propias muestras:

```bash
python collect_data.py    # Captura las señas con la cámara
python clean_data.py      # Procesa el dataset
python train.py           # Entrena y guarda el modelo
```

[Explica cómo se etiquetan las señas durante la captura: ¿se pasa la etiqueta como argumento? ¿se presiona una tecla?]


## Tecnologías

- **MediaPipe Hands** — detección y seguimiento de los 21 landmarks de la mano
- **OpenCV** — captura y procesamiento del video
- **scikit-learn** — clasificador Random Forest
- **pandas / NumPy** — construcción y manipulación del dataset
- **Matplotlib** — visualización de resultados
- **sounddevice** — [salida de audio, si corresponde]

## Limitaciones y trabajo futuro

- Solo reconoce señas estáticas. Las letras que en la lengua de señas involucran movimiento fueron capturadas como pose fija, por lo que el sistema no modela su componente dinámica.
- Entrenado únicamente con la mano derecha; no generaliza a usuarios zurdos ni a señas bimanuales.
- El dataset proviene de una sola persona, lo que limita la generalización a distintas anatomías y estilos de ejecución.
- Extensión natural: incorporar secuencias temporales (LSTM o modelos basados en ventanas de fotogramas) para cubrir señas con movimiento.

## Autor

Alonso Haye Retamal — Estudiante de Ingeniería Civil Eléctrica, Universidad de Chile
