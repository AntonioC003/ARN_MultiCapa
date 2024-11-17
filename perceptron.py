import os
from rembg import remove
from PIL import Image
import io
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import pathlib

def main():
  inp = str(input("Carpeta de entrada:"))
  out = input("Carpeta de salida:")
  CustomImage(inp,out)
  trainAndtest()

def CustomImage(carpeta_entrada,carpeta_salida):

  # Define las carpetas de entrada y salida
  #carpeta_entrada = "1000" 
  #carpeta_salida = "Mil" 
  # Contador para los nombres de las imágenes procesadas
  contador = 0
  
  folder = 'billetes/'
  outputPath = os.path.join(folder, carpeta_salida)
  
  # Crea la carpeta de salida si no existe
  os.makedirs(outputPath , exist_ok=True)
  
  # Procesa cada archivo en la carpeta de entrada
  for nombre_archivo in os.listdir(carpeta_entrada):
      # Verifica que el archivo tenga una extensión de imagen
      if nombre_archivo.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
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
          ruta_salida = os.path.join(outputPath, nuevo_nombre)
          imagen.save(ruta_salida)
          contador += 1
          print(f"Procesada {nombre_archivo} y guardada en {ruta_salida}")

  print("Todas las imágenes han sido procesadas.")





def trainAndtest():
          
  archivo = "./billetes"
  ruta = str(pathlib.Path(archivo))
  print("Ruta donde están almacenadas las imágenes -> ", ruta)

  # # Descripción de clases y su identificador
  descripcion = ("Veinte", "Cincuenta", "Cien", "Doscientos", "Quinientos", "Mil")
  clases = {"Veinte" : 0, "Cincuenta" : 1, "Cien" : 2, "Doscientos" : 3, "Quinientos" : 4, "Mil" : 5}


  num_img_clase = 100
  
  num_entrena = round(num_img_clase * 0.70)

  num_prueba = round(num_img_clase * 0.30)

  # Creación de arreglos para almacenar datos de Entrenamiento para las 3 clases
  # Las imágenes son de 100 (ancho) x 90 (alto)
  imagenes_entrena = np.empty((num_entrena * len(clases), 90, 100), dtype="uint8")
  clases_entrena = np.empty(num_entrena * len(clases), dtype="uint8")

  # Creación de arreglos para almacenar datos de Prueba para las 3 clases
  imagenes_prueba = np.empty((num_prueba * len(clases), 90, 100), dtype="uint8")
  clases_prueba = np.empty(num_prueba * len(clases), dtype="uint8")

  # Cargar datos de Entrenamiento: imágenes de la 0 al 70
  print(num_entrena)
  for i in range(num_entrena):
    print(i)
    for clase in clases:
      print(i)
      imagen = Image.open(ruta +  "/" + clase + "/" + str(i) + ".png")
      indice_instancia = i + clases[clase] * num_entrena
      imagenes_entrena[indice_instancia] = np.array(imagen)
      clases_entrena[indice_instancia] = clases[clase]
      

  # Cargar datos de Prueba: imágenes de la 490 a la 699
  for i in range(num_entrena, num_img_clase):
    for clase in clases:
      imagen = Image.open(ruta + "/"  + clase + "/" + str(i) + ".png")
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
  
  # Escalar los Pixeles de imagenes [0-255]:
  imagenes_entrena = imagenes_entrena / 255
  imagenes_pruebas = imagenes_entrena / 255
  
  # Se crea Modelo de la Red Neuronal
  modelo = tf.keras.Sequential([ 
    tf.keras.layers.Flatten(input_shape=(20,30)),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(3),
    tf.keras.layers.Softmax()
      ])
 
  # Configuramos Modelo:
  modelo.compile( optimizer="sgd", loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=["accuracy"]) 
  
  # Entrenamos Modelo:
  modelo.fit(imagenes_entrena, clases_entrena, epochs=50)
  
  # Evaluamos Modelo:
  perdida, exactitud = modelo.evaluate(imagenes_prueba, clases_prueba)
  print("Exactitud = aciertos_prueba / imagenes_de_prueba):", exactitud)
  
  # Clasificación de imagenes pruebas:
  predicciones = modelo.predict(imagenes_prueba)
  
  # Arreglo para almacenar Prediciones
  # Creación de arreglo para almacenar predicciones
  clase_predicha = np.empty(num_prueba * len(clases), dtype = "uint8")

  for instancia in range(num_prueba * len(clases)):
    # almacena clase predicha para una imagen dada
    clase_predicha[instancia] = np.argmax(predicciones[instancia])
    if clase_predicha[instancia] == clases_prueba[instancia]:
      print("Probabilidades:", predicciones[instancia],
            "Clase predicha:", clase_predicha[instancia],
            "Clase correcta:", clases_prueba[instancia],
            "La Red Neuronal ACERTÓ")
    else:
      print("Probabilidades:", predicciones[instancia],
            "Clase predicha:", clase_predicha[instancia],
            "Clase correcta:", clases_prueba[instancia],
            "La Red Neuronal ERRÓ")
      
    matriz = tf.math.confusion_matrix(clases_prueba, clase_predicha)
    print("Matriz de Confusión:\n", matriz.numpy())

if __name__ == "__main__":
    main()