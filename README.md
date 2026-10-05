# Hand Tracking for Sign Language

Traduce el alfabeto dactilológico de la lengua de señas a texto en tiempo real, usando solo una cámara web y la CPU.

<p align="center">
  <img width="607" alt="Interfaz del reconocimiento en tiempo real" src="https://github.com/user-attachments/assets/dfd2674b-8417-4727-9ac4-5f04543e8f82" />
</p>

## ¿De qué se trata?

La idea es facilitar la comunicación en entornos digitales (videollamadas, mensajería, formularios) entre personas sordas y personas que no manejan lengua de señas, convirtiendo las señas en subtítulos sin necesidad de un intérprete presente.

En vez de clasificar la imagen completa, el sistema usa [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) para encontrar los **21 puntos clave (landmarks)** de la mano y entrena un clasificador sobre esas coordenadas. Esto tiene tres ventajas:

- Es liviano: corre fluido en tiempo real sin GPU.
- Necesita pocos datos: unos cientos de muestras por letra bastan.
- El fondo y la ropa casi no influyen, porque el clasificador solo ve la geometría de la mano.

El dataset incluido lo grabé yo mismo: las 26 letras (A-Z) hechas con la mano derecha como poses estáticas, unas 10.800 muestras en total.

## Inicio rápido

Necesitas **Python 3.10, 3.11 o 3.12** (MediaPipe todavía no es compatible con 3.13) y una cámara web.

```bash
git clone https://github.com/hayealonso/HandTrackingForSignLanguage.git
cd HandTrackingForSignLanguage
python -m venv .venv
```

Activa el entorno virtual:

```bash
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate
```

Instala las dependencias y ejecuta:

```bash
pip install -r requirements.txt
python main.py
```

El repositorio ya trae un modelo entrenado (`asl_model.pkl`), así que no hace falta entrenar nada para probarlo.

### Controles

| Tecla | Acción |
|---|---|
| `Espacio` | Agrega un espacio entre palabras |
| `Backspace` | Borra el último carácter |
| `C` | Limpia todo el subtítulo |
| `Esc` | Cierra el programa |

Para escribir una letra, mantén la seña quieta hasta que se llene la barra de progreso. Para repetir una letra (como en "LL"), baja la mano un instante y vuelve a hacer la seña.

## Cómo funciona

El proyecto es un pipeline de cuatro scripts que se ejecutan en orden, más dos módulos de apoyo:

| Archivo | Qué hace |
|---|---|
| `collect_data.py` | **Captura.** Abre la cámara, detecta la mano y guarda sus landmarks normalizados en `landmarks_data.csv`, etiquetados con la letra que elijas. |
| `clean_data.py` | **Limpieza.** Descarta filas con etiquetas inválidas o datos faltantes y genera `landmarks_data_clean.csv`. |
| `train.py` | **Entrenamiento.** Entrena el clasificador, lo evalúa con datos que no vio y guarda el modelo en `asl_model.pkl`. |
| `main.py` | **Reconocimiento.** Predice la letra en tiempo real, suaviza la predicción para evitar parpadeos y arma el subtítulo letra por letra. |
| `hand_features.py` | Normalización de los landmarks y cálculo de features, compartido por todos los scripts. |
| `classifiers.py` | Catálogo de clasificadores disponibles (Random Forest, SVM, red neuronal, etc.). |

### Del video a la letra

1. **Detección.** MediaPipe entrega 21 puntos (x, y, z) de la mano en cada frame.
2. **Normalización.** Las coordenadas se centran en la muñeca y se escalan según el tamaño de la mano, así da igual dónde esté en la imagen o qué tan cerca de la cámara. Si es la mano izquierda, se refleja para que quede como una derecha.
3. **Features.** A las 63 coordenadas se les suman las 210 distancias entre cada par de puntos, que describen la forma de la mano (qué dedos se tocan, cuáles están doblados).
4. **Clasificación.** El modelo entrega una probabilidad por letra; solo se acepta si supera el 70 %.
5. **Estabilización.** Se vota entre las últimas 12 predicciones y la letra se escribe cuando se mantiene estable durante 6 frames.

## Entrenar tu propio modelo

Si quieres agregar tus propias muestras o empezar de cero:

```bash
python collect_data.py   # 1. graba las señas
python clean_data.py     # 2. limpia el dataset
python train.py          # 3. entrena y guarda el modelo
```

**Durante la captura**, presiona la tecla de la letra que quieres grabar (A-Z), acomoda la mano y presiona `Espacio` para empezar a grabar. El script guarda muestras mientras tu mano esté visible y se pausa solo al llegar al límite por letra (500 por defecto). `Espacio` también sirve para pausar, y `Esc` para salir. Si ya existe un CSV, las muestras nuevas se agregan al final sin borrar las anteriores.

### Elegir el clasificador

`train.py` permite elegir el modelo y el conjunto de features:

```bash
python train.py --model svm          # SVM (por defecto)
python train.py --model rf           # Random Forest, el modelo original del proyecto
python train.py --model mlp          # red neuronal (perceptrón multicapa)
python train.py --model et           # Extra Trees
python train.py --model knn          # k vecinos más cercanos
python train.py --features raw       # solo las 63 coordenadas, sin distancias
python train.py --model mlp --tune   # busca los mejores hiperparámetros (más lento)
```

`main.py` detecta solo qué modelo y qué features se usaron, así que no hay que cambiar nada para usarlo. Para agregar un clasificador nuevo basta con sumarlo al diccionario `CLASSIFIERS` en `classifiers.py`.

## Resultados

Para medir la precisión de forma honesta, el set de prueba se separa por **bloques de video**, no por frames sueltos. Como los frames consecutivos son casi idénticos, separarlos al azar deja "copias" de los datos de prueba en el entrenamiento y la precisión sale inflada (entre 99,4 % y 99,9 % para cualquier modelo).

Accuracy promedio con validación cruzada de 5 folds sobre bloques de video no vistos:

| Clasificador | Solo coordenadas (`raw`) | Coordenadas + distancias (`dist`) |
|---|:---:|:---:|
| Random Forest | 95,7 % | 99,1 % |
| Extra Trees | 96,2 % | 99,0 % |
| k vecinos (kNN) | 96,1 % | 99,0 % |
| Red neuronal (MLP) | 97,8 % | 99,1 % |
| **SVM** | 98,4 % | **99,3 %** |

La combinación por defecto (SVM + distancias) comete unas **6 veces menos errores** que la versión original (Random Forest + coordenadas), y el modelo pesa 2 MB en vez de 44 MB.

<p align="center">
  <img src="confusion_matrix.png" width="600" alt="Matriz de confusión del modelo SVM en el set de prueba" />
</p>

Los pocos errores que quedan se concentran en **M, N y T**: las tres son un puño cerrado y solo se diferencian por dónde queda el pulgar entre los dedos, algo que a veces ni MediaPipe logra ver bien.

## Tecnologías

- **[MediaPipe Hands](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)**: detección y seguimiento de los 21 landmarks de la mano.
- **OpenCV**: captura de video e interfaz.
- **scikit-learn**: clasificadores y evaluación.
- **pandas / NumPy**: manejo del dataset.
- **Pillow**: texto con fuentes TTF (con tildes) en la interfaz.
- **Matplotlib**: gráfico de la matriz de confusión.

## Limitaciones y próximos pasos

- **Solo señas estáticas.** Las letras que llevan movimiento (como la J o la Z) se grabaron como una pose fija. Para cubrirlas habría que modelar secuencias de frames, por ejemplo con una LSTM o un Transformer sobre ventanas de landmarks.
- **Un solo signante.** Todo el dataset es de una persona, así que puede costarle con manos de otras proporciones o estilos distintos. Sumar muestras de más gente es probablemente la mejora con más impacto.
- **Una mano a la vez.** La mano izquierda se reconoce reflejándola, pero no hay soporte para señas con dos manos.
- **Sin corrección de texto.** El subtítulo se arma letra por letra; un corrector o modelo de lenguaje podría completar palabras y corregir errores.

## Autor

**Alonso Haye Retamal**, estudiante de Ingeniería Civil Eléctrica, Universidad de Chile.
