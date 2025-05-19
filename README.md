# 🐱🐰 Cats VS Rabbits Classification

### Descripción del proyecto 

Este proyecto tiene como objetivo desarrollar un modelo de aprendizaje automático  (Machine Learning) capaz de clasificar imágenes de forma binaria, identificando si pertenecen a un **gato** o un **conejo**. A través del entrenamiento con un conjunto de imágenes variadas, el modelo busca aprender a reconocer patrones visuales distintivos entre ambas especies, incluso en presencia de diferentes posturas, razas y fondos.

### Descripción del Dataset

El conjunto de datos utilizado fue obtenido de la plataforma Kaggle. Contiene imágenes de **gatos y conejos** en diversas condiciones: múltiples razas, posturas y escenarios visuales, lo que permite entrenar un modelo más robusto y generalizable.

### Características del dataset:

- Contiene **2,000 imágenes** distribuidas en las capetas test, train y validation:
    - `Cat`: 1,000 imágenes de gatos
        - Train: 800 imágenes
        - Test: 10 imágenes
        - Validation: 207 imágenes
    - `Rabbit`: 1,000 imágenes
        - Train: 800 imágenes
        - Test: 5 imágenes
        - Validation: 207 imágenes
- Formato de imagen: `.jpg`
- No contiene etiquetas adicionales como raza, edad o ubicación.
- Todas las imágenes ya han sido redimensionadas a (300, 300).

## Obtener, generar o aumentar un set de datos.
El dataset utilizado fue descargado desde la plataforma Kaggle [(Munir Yadi, “Cat vs Rabbit”)](https://www.kaggle.com/datasets/muniryadi/cat-vs-rabbit) 

Se redistribuyó de forma manual la información contenida en el dataset y realizó una limpieza en los datos que presentaban elementos no relacionados (imágenes con demasiado zoom unicamente de pelaje, imágenes de pasto) para tener un mejor equilibro en los datos para entrenar, validar y probar el modelo de ML

- Entrenamiento: 70%
    - `Cat:` 700 imágenes
    - `Rabbit`: 700 imágenes
- Validación: 15%
    - `Cat:` 150 imágenes
    - `Rabbit`: 150 imágenes
- Prueba: 15%
    - `Cat:` 150 imágenes
    - `Rabbit`: 150 imágenes
    

### Escalamiento y Preprocesado de Datos
Para mejorar el rendimiento y la capacidad de generalización del modelo, se implementaron técnicas de preprocesado utilizando TensorFlow y Keras. 

Las imágenes originales tenían un tamaño de 300x300 píxeles, pero se redimensionaron a 150x150 para poder acelerar el entrenamiento y reducir el uso de memoria (RAM o GPU) además, simplica los cálculos al disminuir la cantidad de píxeles por imagen.
Igualmente preserva suficiente información visual para tareas de clasificación binaria como en este caso es la distinción entre gatos y conejos, ya que estas clases tienen características fácilmente detectables incluso a menor resolución no hace falta tener una alta resolución para poder distinguir entre ambas especies.

- `rescale = 1./255,` : Convierte los valores de píxeles de [0, 255] a [0, 1]
- `rotation_range=20`:  Gira aleatoriamente la imagen en un rango de ±20 grados
- `width_shift_range=0.1`:  Desplaza la imagen horizontalmente hasta un 10%
- `height_shift_range=0.2`: Desplaza la imagen verticalmente hasta un 10%
- `zoom_range=0.15`: Aplica un zoom aleatorio de hasta un 15%
- `horizontal_flip = True:` Invierte horizontalmente la imagen de forma aleatoria
- `brightness_range=[0.5, 1.2]:` Ajusta aleatoriamente el brillo entre 50% y 120%
