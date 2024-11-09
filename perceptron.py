import os
from rembg import remove
from PIL import Image
import io
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import pathlib
from tensorflow.keras.models import load_model # type: ignore
def main():
  
  while True:
    print("1.- Escalar imagenes")
    print("2.- Entrenar ARN")
    print("3.- Predecir imagen")
    print("4.- Salir")
    op = int(input("Elige una opcion:"))
  
    if op == 1:
      inp = str(input("Nombre carpeta de entrada de imagenes:"))
      out = input("Nombre de carpeta de imagenes procesadas:")
      Custom_Image(inp,out)
    elif op == 2: 
      train_and_test()
    elif op == 3: 
      modelo = load_model("modelo.h5")
      predict_image(modelo)
    elif op == 4:
      break   
    else:
      print("Opcion no disponible")
    

# ----- Funcion para entrenar el modelo 
def train_model(img_train,class_train):
  modelo = tf.keras.Sequential([
      tf.keras.layers.Flatten(input_shape=(90, 100)),
      tf.keras.layers.Dense(64, activation="relu"),
      tf.keras.layers.Dense(4), # Cantidad de clases 
      tf.keras.layers.Softmax()
  ])
  modelo.compile(optimizer="sgd",
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=["accuracy"])
  
  modelo.fit(img_train, class_train, epochs=50)
  modelo.save("modelo.h5") # Guarda el modelo entrenado
  return modelo

def predict_image(modelo):
    # Selecciona y carga una imagen para predecir
    nombre_archivo = input("Ingresa la ruta de la imagen a clasificar:")
    imagen = Image.open(nombre_archivo)
    
# Verifica que el archivo tenga una extensión de imagen
    if nombre_archivo.lower().endswith(('.png', '.jpg', '.jpeg')):
        # Ruta completa de la imagen de entrada
        ruta_entrada = os.path.join(nombre_archivo)
        
        # Abre la imagen y la convierte en bytes para procesar con rembg
        with open(ruta_entrada, "rb") as input_file:
            imagen_bytes = input_file.read()
        # Elimina el fondo
        imagen_sin_fondo = remove(imagen_bytes)
        # Convierte la salida de rembg a un objeto PIL Image
        imagen = Image.open(io.BytesIO(imagen_sin_fondo))
        
        if imagen.height > imagen.width: # Ajusta imagen si es vertical
            imagen = imagen.transpose(Image.ROTATE_90) # Rota la imagen a 90 grados

        # Aplica redimensionamiento y convierte a escala de grises
        imagen = imagen.resize((100, 90))
        imagen = imagen.convert("L")
    
    # Convertir la imagen a un arreglo numpy y escalar
    imagen_array = np.array(imagen) / 255.0 
    plt.figure()
    plt.imshow(imagen_array, cmap="gray")
    plt.colorbar()
    plt.grid(False)
    plt.box(False)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.show()
    imagen_array = np.expand_dims(imagen_array, axis=0)  # Expandir dimensiones para que se ajuste a la entrada del modelo
    
    # Hacer la predicción
    prediccion = modelo.predict(imagen_array)
    clase_predicha = np.argmax(prediccion)
    descripcion = ("veinte", "cincuenta", "doscientos", "quinientos")
    
    print(f"La imagen es probablemente un billete de {descripcion[clase_predicha]} pesos.")

def Custom_Image(carpeta_entrada,carpeta_salida):
  
  # Contador para los nombres de las imágenes procesadas
  contador = 0
  # Crea la carpeta de salida si no existe
  ruta_salida_completa = os.path.join("billetes", carpeta_salida)
  os.makedirs(ruta_salida_completa, exist_ok=True)
  # Procesa cada archivo en la carpeta de entrada
  for nombre_archivo in os.listdir(carpeta_entrada):
      # Verifica que el archivo tenga una extensión de imagen
      if nombre_archivo.lower().endswith(('.png', '.jpg', '.jpeg')):
          # Ruta completa de la imagen de entrada
          ruta_entrada = os.path.join(carpeta_entrada, nombre_archivo)
          
          # Abre la imagen y la convierte en bytes para procesar con rembg
         
          with open(ruta_entrada, "rb") as input_file:
              imagen_bytes = input_file.read()
          # Elimina el fondo
          imagen_sin_fondo = remove(imagen_bytes)
          # Convierte la salida de rembg a un objeto PIL Image
          imagen = Image.open(io.BytesIO(imagen_sin_fondo))
          
          if imagen.height > imagen.width: # Ajusta imagen si es vertical
              imagen = imagen.transpose(Image.ROTATE_90)  # Rota la imagen 90 grados en el sentido de las agujas del reloj

          # Aplica redimensionamiento y convierte a escala de grises
          imagen = imagen.resize((100, 90))
          imagen = imagen.convert("L")
          # Define el nuevo nombre para la imagen procesada usando el contador
          nuevo_nombre = f"{contador}.png"  # Cambia la extensión si lo deseas
          # Define la ruta de salida y guarda la imagen procesada
          ruta_salida = os.path.join(ruta_salida_completa, nuevo_nombre)
          imagen.save(ruta_salida)
          contador += 1
          print(f"Procesada {nombre_archivo} y guardada en {ruta_salida}")

  print("Todas las imágenes han sido procesadas.")

def train_and_test():
  # Carpeta principal que contiene las imagenes procesadas
  archivo = "./billetes" 
  ruta = str(pathlib.Path(archivo)) 
  
  # Descripción de clases y su identificador
  descripcion = ("veinte", "cincuenta","doscientos", "quinientos")
  clases = {"veinte":0, "cincuenta":1,"doscientos":2, "quinientos":3}

  # Número de imágenes de cada clase
  num_img_clase = 100

  # Imágenes de Entrenamiento de cada clase: 70
  num_entrena = round(num_img_clase * 0.70)

  # Imágenes de Prueba de cada clase: 30
  num_prueba = round(num_img_clase * 0.30)

  # Creación de arreglos para almacenar datos de Entrenamiento y Prueba para cada clase
  # Las imágenes son de 90 (alto) x 100 (ancho)
  imagenes_entrena = np.empty((num_entrena * len(clases), 90, 100), dtype="uint8")
  clases_entrena = np.empty(num_entrena * len(clases), dtype="uint8")

  imagenes_prueba = np.empty((num_prueba * len(clases), 90, 100), dtype="uint8")
  clases_prueba = np.empty(num_prueba * len(clases), dtype="uint8")

  # Cargar datos de Entrenamiento
  for i in range(num_entrena):
    for clase in clases:
      imagen = Image.open(ruta + "/" + clase + "/" + str(i) + ".png")
      indice_instancia = i + clases[clase] * num_entrena
      imagenes_entrena[indice_instancia] = np.array(imagen)
      clases_entrena[indice_instancia] = clases[clase]
      
  # Cargar datos de Prueba
  for i in range(num_entrena, num_img_clase):
    for clase in clases:
      imagen = Image.open(ruta + "/" + clase + "/" + str(i) + ".png")
      indice_instancia = i + clases[clase] * num_prueba - num_entrena
      imagenes_prueba[indice_instancia] = np.array(imagen)
      clases_prueba[indice_instancia] = clases[clase]

  plt.figure(figsize=(10, 10))
  for i in range(100):
      plt.subplot(10, 10, i + 1)
      # Selección aleatoria de una imagen
      indice = random.randint(0, num_entrena*len(clases) - 1)
      plt.imshow(imagenes_entrena[indice], cmap="gray")
      plt.xlabel(descripcion[clases_entrena[indice]])
      plt.grid(False)
      plt.box(False)
      plt.xticks([])
      plt.yticks([])
  plt.show()

  plt.figure()
  plt.imshow(imagenes_prueba[3], cmap="gray")
  plt.colorbar()
  plt.grid(False)
  plt.box(False)
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  plt.show()

  imagenes_entrena = imagenes_entrena / 255
  imagenes_prueba = imagenes_prueba / 255

  modelo = train_model(imagenes_entrena, clases_entrena)
  exactitud = modelo.evaluate(imagenes_prueba, clases_prueba)
  print("Exactitud (Accuracy) = aciertos_prueba / imagenes_de_prueba):", exactitud)
  predicciones = modelo.predict(imagenes_prueba)
  # Creación de arreglo para almacenar predicciones
  clase_predicha = np.empty(num_prueba * len(clases), dtype = "uint8")

  for instancia in range(num_prueba * len(clases)):
    # almacena clase predicha para una imagen dada
    clase_predicha[instancia] = np.argmax(predicciones[instancia])

  matriz = tf.math.confusion_matrix(clases_prueba, clase_predicha)
  print("Matriz de Confusión:\n", matriz.numpy())


if __name__ == "__main__":
    main()